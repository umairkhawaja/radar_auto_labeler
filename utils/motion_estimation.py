import numpy as np

import repackage
repackage.up()

from utils.ransac_solver import RANSACSolver
from utils.transforms import transform_doppler_points


def transform_points(points, transformation_matrix):
    """
    Transform the points using the provided transformation matrix.
    
    Args:
        points (np.ndarray): Point cloud data of shape (N, 6).
        transformation_matrix (np.ndarray): Transformation matrix of shape (4, 4).

    Returns:
        np.ndarray: Transformed points of shape (N, 6).
    """
    # Convert points to homogeneous coordinates
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    
    # Apply the transformation matrix
    transformed_points = points_homogeneous @ transformation_matrix.T
    
    # Replace the x, y, z coordinates with the transformed ones
    points[:, :3] = transformed_points[:, :3]
    
    return points

def aggregate_points(pcs, calibs, transform_to_car_frame):
    """
    Aggregate point clouds and optionally transform to car frame.

    Args:
        pcs (list of np.ndarray): List of point clouds, each with shape (N, 6).
        calibs (list of np.ndarray): List of calibration matrices, each with shape (4, 4).
        transform_to_car_frame (bool): Whether to transform points to car frame.

    Returns:
        np.ndarray: Aggregated points with appended sensor index, shape (M, 7).
    """
    pcs_aggregated = []

    for i, pc in enumerate(pcs):
        if transform_to_car_frame:
            # Transform to car frame
            transform_points(pc, calibs[i])
        
        # Append sensor index to each point
        sensor_indices = np.full((pc.shape[0], 1), i)
        pc_with_index = np.hstack((pc, sensor_indices))

        # Aggregate points
        pcs_aggregated.append(pc_with_index)
    
    return np.vstack(pcs_aggregated)

def aggregate_to_car_frame(pcs, calibs):
    """
    Aggregate points and transform to car frame.

    Args:
        pcs (list of np.ndarray): List of point clouds, each with shape (N, 6).
        calibs (list of np.ndarray): List of calibration matrices, each with shape (4, 4).

    Returns:
        np.ndarray: Aggregated points in car frame with appended sensor index, shape (M, 7).
    """
    return aggregate_points(pcs, calibs, transform_to_car_frame=True)

def aggregate_to_sensor_frame(pcs):
    """
    Aggregate points without transforming to car frame.

    Args:
        pcs (list of np.ndarray): List of point clouds, each with shape (N, 6).

    Returns:
        np.ndarray: Aggregated points in sensor frame with appended sensor index, shape (M, 7).
    """
    dummy_calibs = [np.eye(4) for _ in pcs]  # No transformation, use identity matrices
    return aggregate_points(pcs, dummy_calibs, transform_to_car_frame=False)


def fit_model(R, v_d, index1, index2):
    A = np.vstack((R[index1, :], R[index2, :]))
    b = np.array([v_d[index1], v_d[index2]])
    
    # Solve the linear system using NumPy's least squares solver
    model = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]

    return model

def compute_ransac_inliers(R, v_d, points, residual_threshold, max_iterations):
    np.random.seed(1)
    best_inlier_indices = []

    for _ in range(max_iterations):
        # Randomly select two points to form a model
        index1 = np.random.randint(0, R.shape[0])
        index2 = np.random.randint(0, R.shape[0])

        # Ensure distinct indices
        while index2 == index1:
            index2 = np.random.randint(0, R.shape[0])

        # Fit a model using the selected points
        model = fit_model(R, v_d, index1, index2)

        # Count inliers
        residuals = np.abs(R @ model - v_d)
        inlier_indices = np.where(residuals < residual_threshold)[0]

        # Update best_inliers if this model is better
        if len(inlier_indices) > len(best_inlier_indices):
            best_inlier_indices = inlier_indices

    return best_inlier_indices.tolist()

def build_lsq_problem(points, calib_list):
    N = len(points)
    A = np.zeros((N, 2))
    b = np.zeros((N, 1))

    for i, point in enumerate(points):
        point_vector = point[:6]
        sensor_id = int(point[-1]) if len(calib_list) > 1 else 0  # Handle single sensor case
        calib = calib_list[sensor_id]

        xs, ys = calib[0, 3], calib[1, 3]
        sensor_angle = np.arctan2(calib[1, 0], calib[0, 0])
        theta = np.arctan2(point_vector[1], point_vector[0]) + sensor_angle

        M_j = np.array([[np.cos(theta), np.sin(theta)]])
        v_d_j = -point_vector[4]

        S_j = np.array([[-ys, 1.0], [xs, 0.0]])
        R_j = M_j @ S_j

        A[i, :] = R_j
        b[i, 0] = v_d_j

    return A, b

def estimate_velocity_from_multiple_sensors(points, calib_list, use_ransac, th):
    A, b = build_lsq_problem(points, calib_list)

    if use_ransac:
        max_iterations = 100
        ransac_inliers = compute_ransac_inliers(A, b.flatten(), points, th, max_iterations)
        A_filtered = A[ransac_inliers, :]
        b_filtered = b[ransac_inliers]

        A = A_filtered
        b = b_filtered

    # Solve the least squares problem
    lsq_result = np.linalg.lstsq(A.T @ A, A.T @ b, rcond=None)[0]
    return lsq_result.flatten()

def get_inliers_from_known_velocity(point_list, calib_list, w, v, th):
    A, b = build_lsq_problem(point_list, calib_list)
    lsq_result = np.array([w, v])
    
    residuals = np.abs(A @ lsq_result - b.flatten())
    inlier_indices = np.where(residuals < th)[0]

    return inlier_indices.tolist()

# Aggregation functions (translated from C++)
def transform_points(points, transformation_matrix):
    points_homogeneous = np.hstack((points[:, :3], np.ones((points.shape[0], 1))))
    transformed_points = points_homogeneous @ transformation_matrix.T
    points[:, :3] = transformed_points[:, :3]
    return points

def aggregate_points(pcs, calibs, transform_to_car_frame):
    pcs_aggregated = []
    for i, pc in enumerate(pcs):
        if transform_to_car_frame:
            transform_points(pc, calibs[i])
        sensor_indices = np.full((pc.shape[0], 1), i)
        pc_with_index = np.hstack((pc, sensor_indices))
        pcs_aggregated.append(pc_with_index)
    return np.vstack(pcs_aggregated)

def aggregate_to_car_frame(pcs, calibs):
    return aggregate_points(pcs, calibs, transform_to_car_frame=True)

def aggregate_to_sensor_frame(pcs):
    dummy_calibs = [np.eye(4) for _ in pcs]
    return aggregate_points(pcs, dummy_calibs, transform_to_car_frame=False)


## Using Dani's Motion Estimation
def process_point_clouds(point_clouds_dict, calib_dict, poses_dict, use_ransac=True, threshold=0.5):
    inliers_dict = {name: [] for name in point_clouds_dict}
    scene_pointclouds = {name: [] for name in point_clouds_dict}

    for scene_name in point_clouds_dict:
        scans_pcls = point_clouds_dict[scene_name]
        scans_calibs = calib_dict[scene_name]
        scans_poses = poses_dict[scene_name]
        inliers_dict[scene_name] = []


        for sensor_pcls, sensor_calibs, pose in zip(scans_pcls, scans_calibs, scans_poses):
            # Aggregate point clouds to sensor frame
            aggregated_points = aggregate_to_sensor_frame(sensor_pcls)

            # Estimate velocity using RANSAC
            velocity_estimate = estimate_velocity_from_multiple_sensors(aggregated_points, sensor_calibs, use_ransac, threshold)
            estimated_w, estimated_v = velocity_estimate
            
            # Get inliers based on estimated velocity
            inliers = get_inliers_from_known_velocity(aggregated_points, sensor_calibs, estimated_w, estimated_v, threshold)
            inliers_dict[scene_name].append(inliers)

            merged_pcls = []
            for pcl, calib in zip(sensor_pcls, sensor_calibs):
                ego_pcl = transform_doppler_points(calib, pcl)
                global_pcl = transform_doppler_points(pose, ego_pcl)
                merged_pcls.append(global_pcl)
            scene_pointclouds[scene_name].append(np.vstack(merged_pcls))

    
    return inliers_dict, scene_pointclouds

### AutoPlace Dynamic Point Removal

def remove_dynamic_points(scene_pointclouds, scene_calibs, scene_poses, sensors, filter_sensors=False, dpr_thresh=0.15, save_vis=False):
    ransac_solver = RANSACSolver(threshold=dpr_thresh, max_iter=10, outdir='output_dpr')

    dpr_masks = {scene_name: [] for scene_name in scene_pointclouds}
    global_scene_pointclouds = {scene_name: [] for scene_name in scene_pointclouds}


    for scene_name in scene_pointclouds:
        scans_pcl = scene_pointclouds[scene_name]
        scans_calibs = scene_calibs[scene_name]
        poses = scene_poses[scene_name]
        
        for index, (sensor_pcls, sensor_calibs, pose) in enumerate(zip(scans_pcl, scans_calibs, poses)):
            merged_pcl_global = []
            merged_pcl_dpr_mask = []

            for sensor, pcl, calib in zip(sensors, sensor_pcls, sensor_calibs):
                if filter_sensors and sensor in ['RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']:
                    continue

                if pcl.shape[0] > 1:
                    info = [
                        [scene_name, index],
                        sensor,
                        pcl.shape[0]
                    ]
                    best_mask, _, _ = ransac_solver.ransac_nusc(info=info, pcl=pcl, vis=(save_vis and sensor == 'RADAR_FRONT'))

                    ego_points_sensor = transform_doppler_points(calib, pcl)
                    global_points_sensor = transform_doppler_points(pose, ego_points_sensor)
                    merged_pcl_global.append(global_points_sensor)
                    merged_pcl_dpr_mask.append(best_mask)
            
            dpr_masks[scene_name].append(np.hstack(merged_pcl_dpr_mask))
            global_scene_pointclouds[scene_name].append(np.vstack(merged_pcl_global))

    return dpr_masks, global_scene_pointclouds
