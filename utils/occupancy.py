from tqdm import tqdm
import octomap
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable, get_cmap
from utils.transforms import transform_doppler_points

def build_octomap(dl, resolution=0.1):
    # Initialize the OctoMap with the specified resolution
    poses = dl.global_poses
    num_readings = dl.num_readings
    octree = octomap.OcTree(resolution)


    for i in tqdm(range(num_readings)):
        pointclouds = dl[i][0]
        calibs = dl[i][1]
        pose = poses[i]

        if type(pointclouds) == list:
            for pointcloud, calib in zip(pointclouds, calibs):
                ego_pointcloud = transform_doppler_points(calib, pointcloud)
                global_pointcloud = transform_doppler_points(pose, ego_pointcloud)

                sensor_origin = pose[:3, 3]
                octree.insertPointCloud(global_pointcloud, sensor_origin)

    return octree


def extract_high_occupancy_points(octree, threshold):
    """
    Extract points from the OctoMap with occupancy greater than the specified threshold.

    Args:
        octomap_file (str): Path to the OctoMap file.
        threshold (float): The occupancy probability threshold.

    Returns:
        numpy.ndarray: The extracted point cloud as an array of shape (N, 3).
    """
    # List to store the filtered points
    points = []

    # Iterate over all leaf nodes in the OctoMap
    for node in octree.begin_leafs():
        occupancy = node.getOccupancy()
        if occupancy > threshold:
            x, y, z = node.getCoordinate()
            points.append([x, y, z])

    # Convert the list to a numpy array
    pointcloud = np.array(points)
    return pointcloud


def plot_occupancy_score_dist(omap):
    occupancy_dist = []
    for node in omap.begin_leafs():
        x,y,z = node.getCoordinate()
        occupancy = node.getOccupancy()
        occupancy_dist.append(occupancy)
        # print(f"Node at ({x:0.2f}, {y:0.2f}, {z:0.2f}) has occupancy probability {occupancy:0.2f}")

    plt.figure(figsize=(12, 6))
    plt.hist(occupancy_dist, bins=30, color='skyblue', edgecolor='black')
    plt.title('Distribution of Occupancy Values')
    plt.xlabel('Occupancy Score')
    plt.ylabel('Frequency')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.show()


def plot_octomap(octree, occupancy_thresh=0.5):
    # Retrieve all occupied nodes and their occupancy scores
    occupied_nodes = []
    empty_nodes = []
    occupancy_scores = []

    # Iterate through all leaf nodes in the octree
    for node in octree.begin_leafs():
        occupancy = node.getOccupancy()
        x, y, z = node.getCoordinate()
        if occupancy > occupancy_thresh:
            occupied_nodes.append([x, y, z])
            occupancy_scores.append(occupancy)
        else:
            empty_nodes.append([x, y, z])
            

    # Convert to numpy array for easier handling
    occupied_nodes = np.array(occupied_nodes)
    empty_nodes = np.array(empty_nodes)
    occupancy_scores = np.array(occupancy_scores)


    # Normalize occupancy scores for colormap
    norm = Normalize(vmin=occupancy_scores.min(), vmax=occupancy_scores.max())
    cmap = get_cmap('RdYlGn')  # Red-Yellow-Green colormap
    colors = cmap(norm(occupancy_scores))

    # Create Open3D point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(occupied_nodes)
    pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])  # Open3D uses RGB colors
    resolution = 0.5

    empty_pcd = o3d.geometry.PointCloud()
    empty_pcd.points = o3d.utility.Vector3dVector(empty_nodes)

    # Create voxel grids for occupied and empty spaces
    occupied_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(
    pcd, voxel_size=resolution
    )
    empty_voxels = o3d.geometry.VoxelGrid.create_from_point_cloud(
    empty_pcd, voxel_size=resolution
    )

    # Set colors for the voxel grids
    # occupied_voxels.paint_uniform_color([1.0, 0, 0])
    # empty_voxels.paint_uniform_color([0.5, 0.5, 0.5])

    # Create a visualizer
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Add geometries to the visualizer
    # vis.add_geometry(pcd)
    vis.add_geometry(occupied_voxels)
    vis.add_geometry(empty_voxels)

    # Run the visualizer
    vis.run()
    vis.destroy_window()


def load_octomap(path):
    with open(path, 'rb') as f:
        return octomap.OcTree(f.read())


def save_octomap(path, tree):
    with open(path, 'wb') as f:
        f.write(tree.writeBinary())