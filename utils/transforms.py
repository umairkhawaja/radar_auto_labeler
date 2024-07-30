import numpy as np
import open3d as o3d

def clear_z_one_pose(pose):
    output_pose = np.zeros((6,))
    state_vector = TransformMatrix4dToVector6d(pose)
    output_pose[:2] = state_vector[:2]
    output_pose[5] = state_vector[5]
    return TransformVector6dToMatrix4d(output_pose)

def clear_z(input_pose):
    output_poses = np.zeros((input_pose.shape[0], 4, 4))
    for i, pose in enumerate(input_pose):
        output_poses[i, :, :] = clear_z_one_pose(pose)
    return output_poses

def TransformMatrix4dToVector6d(H):
    x, y, z = H[0, 3], H[1, 3], H[2, 3]

    roll = np.arctan2(H[2, 1], H[2, 2])
    pitch = np.arctan2(-H[2, 0], np.sqrt(H[0, 0] * H[0, 0] + H[1, 0] * H[1, 0]))
    yaw = np.arctan2(H[1, 0], H[0, 0])

    v = np.array([x, y, z, roll, pitch, yaw], dtype=np.float64)

    return v


def TransformVector6dToMatrix4d(v):
    x, y, z, roll, pitch, yaw = v[0], v[1], v[2], v[3], v[4], v[5]

    cos_roll = np.cos(roll)
    sin_roll = np.sin(roll)
    cos_pitch = np.cos(pitch)
    sin_pitch = np.sin(pitch)
    cos_yaw = np.cos(yaw)
    sin_yaw = np.sin(yaw)

    # Construct the rotation matrix using Rodrigues' rotation formula.
    R_yaw = np.array([[cos_yaw, -sin_yaw, 0],
                      [sin_yaw, cos_yaw, 0],
                      [0, 0, 1]], dtype=np.float64)

    R_pitch = np.array([[cos_pitch, 0, sin_pitch],
                        [0, 1, 0],
                        [-sin_pitch, 0, cos_pitch]], dtype=np.float64)

    R_roll = np.array([[1, 0, 0],
                       [0, cos_roll, -sin_roll],
                       [0, sin_roll, cos_roll]], dtype=np.float64)

    R = np.dot(np.dot(R_yaw, R_pitch), R_roll)

    # Translation vector.
    T = np.array([x, y, z], dtype=np.float64)

    # Combine the rotation and translation to form the homogeneous matrix.
    T_matrix = np.eye(4)
    T_matrix[:3, :3] = R
    T_matrix[:3, 3] = T

    return T_matrix

def transform_points(transform, dpoints):
    # N x [x y z rcs v v_comp]
    result_dpoints = np.zeros_like(dpoints)

    points_h = np.ones((dpoints.shape[0], 4))
    points_h[:, :3] = dpoints[:, :3]
    points_h = (points_h @ transform.T)
    result_dpoints[:, :3] = points_h[:, :3]

    return result_dpoints


def transform_doppler_points(transform, dpoints):
    # N x [x y z rcs v v_comp]
    result_dpoints = np.zeros_like(dpoints)

    points_h = np.ones((dpoints.shape[0], 4))
    points_h[:, :3] = dpoints[:, :3]
    points_h = (points_h @ transform.T)
    result_dpoints[:, :3] = points_h[:, :3]

    if dpoints.shape[1] > 3:
        result_dpoints[:, 3:] = dpoints[:, 3:]

    return result_dpoints


def polar_to_cartesian(coordinates):
    # Convert degrees to radians
    azimuth = coordinates[:, 1]
    elevation = np.pi/2 - coordinates[:, 2]

    # Calculate Cartesian coordinates
    x = coordinates[:, 0] * np.sin(elevation) * np.cos(azimuth)
    y = coordinates[:, 0] * np.sin(elevation) * np.sin(azimuth)
    z = coordinates[:, 0] * np.cos(elevation)

    cartesian_coordinates = np.column_stack((x, y, z))
    return cartesian_coordinates


def transform_points_to_car_frame(points, calibs):
    transformed_points = []
    for i in range(points.shape[0]):
        point = points[i, :]
        sensor_idx = int(point[6])
        result_point = np.copy(point)
        point_homogeneous = np.ones(4)
        point_homogeneous[:3] = point[:3]
        T = calibs[sensor_idx]
        result_point[:3] = T.dot(point_homogeneous)[:3]
        transformed_points.append(result_point)
        
    return np.array(transformed_points)


def convert_to_open3d_pcd(np_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points[:, :3])
    return pcd

def align_pointclouds(source, target):
    threshold = 0.5
    trans_init = np.eye(4)
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, trans_init,
        o3d.pipelines.registration.TransformationEstimationPointToPoint())
    return reg_p2p.transformation

def find_overlapping_points(pcd, threshold=0.1):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    overlapping_indices = set()
    points = np.asarray(pcd.points)
    for i, point in enumerate(points):
        [_, idx, _] = pcd_tree.search_radius_vector_3d(point, threshold)
        if len(idx) > 1:
            overlapping_indices.add(i)
    return overlapping_indices

# Map overlapping indices back to the original point clouds
def get_original_indices(pcd_merged, pcd, overlapping_indices):
    merged_points = np.asarray(pcd_merged.points)
    pcd_points = np.asarray(pcd.points)
    tree = o3d.geometry.KDTreeFlann(pcd)
    original_indices = set()
    for idx in overlapping_indices:
        [_, idxs, _] = tree.search_radius_vector_3d(merged_points[idx], 0.1)
        original_indices.update(idxs)
    return original_indices