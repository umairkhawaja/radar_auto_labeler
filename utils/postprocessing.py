import numpy as np
import open3d as o3d

def convert_to_open3d_pcd(np_points):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_points[:, :3])
    return pcd

def align_pointclouds(source, target, threshold=0.5):
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

def euclidean_distance(pose1, pose2):
    """
    Calculate the Euclidean distance between two poses.
    
    Args:
    pose1: np.array, shape (3,) - First pose (x, y, z).
    pose2: np.array, shape (3,) - Second pose (x, y, z).
    
    Returns:
    float - Euclidean distance between pose1 and pose2.
    """
    return np.linalg.norm(pose1 - pose2)

def has_matching_pose(pose, other_poses, threshold):
    """
    Check if there is a matching pose within a given threshold.
    
    Args:
    pose: np.array, shape (3,) - Pose to match.
    other_poses: list of np.array, shape (3,) - List of other poses to match against.
    threshold: float - Distance threshold for matching poses.
    
    Returns:
    bool - True if a matching pose is found, False otherwise.
    """
    for other_pose in other_poses:
        if euclidean_distance(pose, other_pose) < threshold:
            return True
    return False

def create_filtered_maps(scene_poses_dict, scene_pointclouds_dict, threshold=0.1):
    """
    Create filtered maps and poses by adding point clouds to the maps only if the current pose has a match
    in the set of poses from the other scenes within a given distance threshold.
    
    Args:
    scene_poses_dict: dict - Dictionary where keys are scene names and values are lists of np.array with shape (3,)
    scene_pointclouds_dict: dict - Dictionary where keys are scene names and values are lists of np.array with shape (N, 3)
    threshold: float - Distance threshold for matching poses.
    
    Returns:
    filtered_maps_dict: dict - Dictionary where keys are scene names and values are concatenated point cloud maps.
    filtered_scene_poses_dict: dict - Dictionary of filtered poses for each scene.
    """
    filtered_maps_dict = {scene: [] for scene in scene_poses_dict}
    filtered_scene_poses_dict = {scene: [] for scene in scene_poses_dict}
    
    for scene1, poses1 in scene_poses_dict.items():
        for pose1, pc1 in zip(poses1, scene_pointclouds_dict[scene1]):
            matching_found = False
            for scene2, poses2 in scene_poses_dict.items():
                if scene1 != scene2 and has_matching_pose(pose1, poses2, threshold):
                    matching_found = True
                    break
            if matching_found:
                filtered_maps_dict[scene1].append(pc1)
                filtered_scene_poses_dict[scene1].append(pose1)
    
    for scene in filtered_maps_dict:
        if filtered_maps_dict[scene]:
            filtered_maps_dict[scene] = np.concatenate(filtered_maps_dict[scene], axis=0)
        else:
            filtered_maps_dict[scene] = np.array([])  # Return an empty array if no matching poses were found

    return filtered_maps_dict, filtered_scene_poses_dict


def filter_points_within_radius(pose, pointcloud, radius=100.0):
    """
    Filter points in a point cloud that are within a specified radius around a given pose.
    
    Args:
    pose: np.array, shape (3,) - The center pose (x, y, z).
    pointcloud: np.array, shape (N, 3) - The point cloud to filter.
    radius: float - The radius within which points will be retained (default is 100 meters).
    
    Returns:
    np.array - Filtered point cloud with points within the specified radius.
    """
    distances = np.linalg.norm(pointcloud[:,:3] - pose[:3,3], axis=1)
    return pointcloud[distances <= radius]

def create_filtered_maps_with_radius(scene_poses_dict, scene_pointclouds_dict, threshold=1, radius=100.0):
    """
    Create filtered maps and poses by adding point clouds to the maps only if the current pose has a match
    in the set of poses from the other scenes within a given distance threshold. Points in point clouds are
    filtered to be within a specified radius around each pose.
    
    Args:
    scene_poses_dict: dict - Dictionary where keys are scene names and values are lists of np.array with shape (3,)
    scene_pointclouds_dict: dict - Dictionary where keys are scene names and values are lists of np.array with shape (N, 3)
    threshold: float - Distance threshold for matching poses.
    radius: float - Radius around each pose within which points will be retained (default is 100 meters).
    
    Returns:
    filtered_maps_dict: dict - Dictionary where keys are scene names and values are concatenated point cloud maps.
    filtered_scene_poses_dict: dict - Dictionary of filtered poses for each scene.
    """
    filtered_maps_dict = {scene: [] for scene in scene_poses_dict}
    filtered_scene_poses_dict = {scene: [] for scene in scene_poses_dict}
    
    for scene1, poses1 in scene_poses_dict.items():
        for pose1, pc1 in zip(poses1, scene_pointclouds_dict[scene1]):
            matching_found = False
            for scene2, poses2 in scene_poses_dict.items():
                if scene1 != scene2 and has_matching_pose(pose1, poses2, threshold):
                    matching_found = True
                    break
            if matching_found:
                filtered_pc1 = filter_points_within_radius(pose1, pc1, radius)
                if filtered_pc1.size > 0:
                    filtered_maps_dict[scene1].append(filtered_pc1)
                    filtered_scene_poses_dict[scene1].append(pose1)
    
    for scene in filtered_maps_dict:
        if filtered_maps_dict[scene]:
            filtered_maps_dict[scene] = np.concatenate(filtered_maps_dict[scene], axis=0)
        else:
            filtered_maps_dict[scene] = np.array([])  # Return an empty array if no matching poses were found

    return filtered_maps_dict, filtered_scene_poses_dict


def create_filtered_indices(poses, reference_poses, threshold=0.1):
    """
    Create filtered indices by adding point clouds to the maps only if the current pose has a match
    in the set of reference poses within a given distance threshold.
    
    Args:
    poses: list of np.array, shape (4, 4) - List of pose matrices.
    reference_poses: list of np.array, shape (4, 4) - List of reference pose matrices.
    threshold: float - Distance threshold for matching poses.
    
    Returns:
    list - List of indices to keep.
    """
    indices_to_keep = []
    
    for idx, pose in enumerate(poses):
        if has_matching_pose(pose, reference_poses, threshold):
            indices_to_keep.append(idx)

    return indices_to_keep

