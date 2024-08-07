import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree
from collections import defaultdict
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def transfer_labels(labels, radar_points):
    """
    Transfers labels from the labels array to the radar points based on the nearest neighbor search.

    Parameters:
    labels (np.ndarray): An Nx4 array where each row is (x, y, z, label).
    radar_points (np.ndarray): An Mx3 array where each row is (x, y, z).

    Returns:
    np.ndarray: An Mx4 array where each row is (x, y, z, label) with labels from the nearest point in the labels array.
    """
    # Extract the coordinates (x, y, z) from the labels array
    label_coords = labels[:, :3]

    print(np.unique(labels[:, 3], return_counts=True))

    # Build a KDTree for efficient nearest neighbor search
    tree = KDTree(label_coords)

    # Find the nearest neighbors for each radar point
    distances, indices = tree.query(radar_points)

    # Create an array to store the radar points with their corresponding labels
    radar_points_with_labels = np.zeros((radar_points.shape[0], radar_points.shape[1] + 1))

    # Copy the radar points to the new array
    radar_points_with_labels[:, :3] = radar_points

    # Assign the corresponding labels from the labels array
    radar_points_with_labels[:, 3] = labels[indices, 3]

    print(np.unique(radar_points_with_labels[:, 3], return_counts=True))


    return radar_points_with_labels


class VoxelHashMap:
    def __init__(self, voxel_size):
        self.voxel_size = voxel_size
        self.voxel_map = defaultdict(int)  # Using a default dictionary to store voxel counts

    def point_to_voxel_key(self, point):
        ix = int(np.floor(point[0] / self.voxel_size))
        iy = int(np.floor(point[1] / self.voxel_size))
        iz = int(np.floor(point[2] / self.voxel_size))
        return (ix, iy, iz)
    
    def update_with_scan(self, point_cloud):
        for point in point_cloud:
            voxel_key = self.point_to_voxel_key(point)
            self.voxel_map[voxel_key] += 1
    
    def get_normalized_scores(self):
        max_count = max(self.voxel_map.values()) if self.voxel_map else 1
        normalized_scores = {k: 1- (v / max_count) for k, v in self.voxel_map.items()}
        return normalized_scores

    def query_point_score(self, point):
        voxel_key = self.point_to_voxel_key(point)
        return self.voxel_map.get(voxel_key, 0) / max(self.voxel_map.values(), default=1)

    def assign_scores_to_pointcloud(self, point_cloud):
        scores = []
        for point in point_cloud:
            score = self.query_point_score(point)
            scores.append(1 - score) # Undo the 1-normalizd score for consisten visualization
        return np.hstack((point_cloud, np.array(scores).reshape(-1, 1)))
    
def create_voxel_map(radar_scans, voxel_size):
    # Initialize voxel hash map
    voxel_hash_map = VoxelHashMap(voxel_size)

    # Process each radar scan
    for frame_id, point_cloud in radar_scans.items():
        voxel_hash_map.update_with_scan(point_cloud)
    return voxel_hash_map

def get_sps_labels(map, scan_points):
    labeled_map_points = map[:, :3]
    labeled_map_labels = map[:, -1]

    sps_labels = []
    for point in scan_points[:, :3]:
        distances = np.linalg.norm(labeled_map_points - point, axis=1)
        closest_point_idx = np.argmin(distances)
        sps_labels.append(labeled_map_labels[closest_point_idx])
    sps_labels = np.array(sps_labels)
    return sps_labels