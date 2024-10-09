import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt

import os
import cv2
import os.path as osp
from typing import List
from nuscenes.utils.data_classes import RadarPointCloud, LidarPointCloud
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from matplotlib.colors import LinearSegmentedColormap
from utils.labelling import get_sps_labels
from utils.motion_estimation import RANSACSolver


def plot_maps(scene_maps, poses, size=0.5, zoom_level=3):
    plt.figure(figsize=(15, 15))

    colors = ['cyan', 'magenta', 'yellow', 'black', 'purple', 'brown']
    markers = ['o', 'v', 's', 'P', 'X', 'D']
    positions = {name: np.array(pose)[:, :3,3] for name,pose in poses.items()}

    for idx, (scene_name, map) in enumerate(scene_maps.items()):

        plt.scatter(map[:, 0], map[:, 1], s=size, label=scene_name, alpha=0.5)


        if scene_name in positions:
            pose = np.array(positions[scene_name])
            plt.plot(pose[:, 0], pose[:, 1], label=f'{scene_name} Trajectory', color=colors[idx % len(colors)])
            # plt.scatter(pose[:, 0], pose[:, 1], c=colors[idx % len(colors)], marker=markers[idx % len(markers)])
            
            for i in range(1, len(pose)):
                plt.arrow(pose[i-1, 0], pose[i-1, 1], pose[i, 0] - pose[i-1, 0], pose[i, 1] - pose[i-1, 1], 
                          head_width=0.5, head_length=0.5, fc=colors[idx % len(colors)], ec=colors[idx % len(colors)])

    plt.xlabel('X')
    plt.ylabel('Y')

    if zoom_level != -1:
        all_pos = np.concatenate(list(positions.values()))
        x_mean = np.mean(all_pos[:,0])
        y_mean = np.mean(all_pos[:,1])
        x_std = np.std(all_pos[:,0])
        y_std = np.std(all_pos[:,1])
        
        std_dev_range = zoom_level

        x_limits = [x_mean - std_dev_range*x_std, x_mean + std_dev_range*x_std]
        y_limits = [y_mean - std_dev_range*y_std, y_mean + std_dev_range*y_std]

        plt.xlim(x_limits)
        plt.ylim(y_limits)

    plt.title("Overlapped Maps and Trajectories")
    if len(scene_maps) <= 3:
        plt.legend()
    plt.grid(True)
    plt.show()



def plot_voxel_hash_map_open3d(normalized_scores, voxel_hash_map, voxel_size):
    points = []
    colors = []
    cmap = plt.cm.get_cmap('RdYlGn')

    for (ix, iy, iz), score in normalized_scores.items():
        x = (ix + 0.5) * voxel_size  # Center of the voxel
        y = (iy + 0.5) * voxel_size
        z = (iz + 0.5) * voxel_size
        if voxel_hash_map.voxel_map[(ix, iy, iz)] < 1:
            color = [1, 1, 1]  # White for voxels with no points
        else:
            color = cmap(score)[:3]  # Get RGB color from colormap
        points.append([x, y, z])
        colors.append(color)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound())
    
    o3d.visualization.draw_geometries([voxel_grid])

    # Create colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb1.set_label('Static Score')

    plt.show()


def plot_voxel_map(voxel_hash_map, voxel_size):
    # Retrieve the normalized scores
    normalized_scores = voxel_hash_map.get_normalized_scores()

    plot_voxel_hash_map_open3d(normalized_scores, voxel_hash_map, voxel_size)


# def map_pointcloud_to_image(nusc,
#                             sps_map,
#                             pointsensor_token: str,
#                             camera_token: str,
#                             min_dist: float = 1.0):
#     """
#     Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
#     plane.
#     :param pointsensor_token: Lidar/radar sample_data token.
#     :param camera_token: Camera sample_data token.
#     :param min_dist: Distance from the camera below which points are discarded.
#     :param render_intensity: Whether to render lidar intensity instead of point depth.
#     :param show_lidarseg: Whether to render lidar intensity instead of point depth.
#     :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes. If None
#         or the list is empty, all classes will be displayed.
#     :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
#                                     predictions for the sample.
#     :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
#         to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
#         If show_lidarseg is True, show_panoptic will be set to False.
#     :return (pointcloud <np.float: 2, n)>, coloring <np.float: n>, image <Image>).
#     """
#     ransac_solver = RANSACSolver(threshold=0.15, max_iter=10, outdir='output_dpr')

#     cam = nusc.get('sample_data', camera_token)
#     pointsensor = nusc.get('sample_data', pointsensor_token)

#     pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
#     pc = RadarPointCloud.from_file(pcl_path)
#     im = Image.open(osp.join(nusc.dataroot, cam['filename']))

#     ransac_input_pcl = pc.points.T
#     info = [
#         ['', -1],
#         'RADAR',
#         ransac_input_pcl.shape[0]
#     ]
#     best_mask, _, _ = ransac_solver.ransac_nusc(info=info, pcl=ransac_input_pcl, vis=False)
    

#     # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
#     # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
#     cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
#     pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
#     pc.translate(np.array(cs_record['translation']))


#     # Second step: transform from ego to the global frame.
#     poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
#     pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
#     pc.translate(np.array(poserecord['translation']))

#     global_points = pc.points.T
#     static_global_points = global_points[best_mask]
#     dpr_global_points = global_points[~best_mask]

#     sps_score = get_sps_labels(sps_map, static_global_points)

#     # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
#     poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
#     pc.translate(-np.array(poserecord['translation']))
#     pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)
    
#     # Fourth step: transform from ego into the camera.
#     cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
#     pc.translate(-np.array(cs_record['translation']))
#     pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)


#     # Fifth step: actually take a "picture" of the point cloud.
#     # Grab the depths (camera frame z axis points away from the camera).
#     depths = pc.points[2, :]
#     coloring = sps_score

#     # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
#     points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

#     # Remove points that are either outside or behind the camera. Leave a margin of 1 pixel for aesthetic reasons.
#     # Also make sure points are at least 1m in front of the camera to avoid seeing the lidar points on the camera
#     # casing for non-keyframes which are slightly out of sync.
#     mask = np.ones(depths.shape[0], dtype=bool)
#     mask = np.logical_and(mask, depths > min_dist)
#     mask = np.logical_and(mask, points[0, :] > 1)
#     mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
#     mask = np.logical_and(mask, points[1, :] > 1)
#     mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)
#     points = points[:, mask]
#     coloring = coloring[mask]

#     ## Add dummy points for fixing coloring scale
#     points = np.hstack([points, np.array([[1,1,1], [2,2,1]]).T])
#     coloring = np.hstack([coloring, np.array([0]), np.array([1])])
#     return points, coloring, im

# def render_pointcloud_in_image(nusc,
#                                sps_map,
#                                 sample_token: str,
#                                 dot_size: int = 5,
#                                 pointsensor_channel: str = 'LIDAR_TOP',
#                                 camera_channel: str = 'CAM_FRONT',
#                                 out_path: str = None,
#                                 ax = None,
#                                 verbose: bool = True):
#     """
#     Scatter-plots a pointcloud on top of image.
#     :param sample_token: Sample token.
#     :param dot_size: Scatter plot dot size.
#     :param pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
#     :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
#     :param out_path: Optional path to save the rendered figure to disk.
#     :param render_intensity: Whether to render lidar intensity instead of point depth.
#     :param show_lidarseg: Whether to render lidarseg labels instead of point depth.
#     :param filter_lidarseg_labels: Only show lidar points which belong to the given list of classes.
#     :param ax: Axes onto which to render.
#     :param show_lidarseg_legend: Whether to display the legend for the lidarseg labels in the frame.
#     :param verbose: Whether to display the image in a window.
#     :param lidarseg_preds_bin_path: A path to the .bin file which contains the user's lidar segmentation
#                                     predictions for the sample.
#     :param show_panoptic: When set to True, the lidar data is colored with the panoptic labels. When set
#         to False, the colors of the lidar data represent the distance from the center of the ego vehicle.
#         If show_lidarseg is True, show_panoptic will be set to False.
#     """
#     from mpl_toolkits.axes_grid1 import make_axes_locatable
    
#     sample_record = nusc.get('sample', sample_token)


#     # Here we just grab the front camera and the point sensor.
#     pointsensor_token = nusc.get('sample_data', sample_record['data'][pointsensor_channel])['token'] # sample_record['data'][pointsensor_channel]
#     camera_token = nusc.get('sample_data', sample_record['data'][camera_channel])['token'] # sample_record['data'][camera_channel]

#     points, coloring, im = map_pointcloud_to_image(nusc, sps_map, pointsensor_token, camera_token)

#     ## Enforce colouring
#         # Define the custom colormap
#     cdict = {
#         'red':   ((0.0, 1.0, 1.0),
#                   (0.5, 1.0, 1.0),
#                   (1.0, 0.0, 0.0)),

#         'green': ((0.0, 0.0, 0.0),
#                   (0.5, 1.0, 1.0),
#                   (1.0, 1.0, 1.0)),

#         'blue':  ((0.0, 0.0, 0.0),
#                   (0.5, 0.0, 0.0),
#                   (1.0, 0.0, 0.0))
#     }
#     gyr_cmap = LinearSegmentedColormap('GYR', cdict)

#     # Normalize the coloring array to [0, 1]
#     norm = plt.Normalize(vmin=0, vmax=1)
#     coloring_normalized = norm(coloring)

#     # Apply the colormap to the normalized coloring array
#     coloring_rgb = gyr_cmap(coloring_normalized)

#     # Init axes.
#     if ax is None:
#         fig, ax = plt.subplots(1, 1, figsize=(9, 16))
#         fig.canvas.set_window_title(sample_token)
#     else:  # Set title on if rendering as part of render_sample.
#         ax.set_title(camera_channel)
#     ax.imshow(im)
#     scatter = ax.scatter(points[0, :], points[1, :], c=coloring, s=dot_size, cmap='RdYlGn')
#     divider = make_axes_locatable(ax)
#     cax = divider.append_axes("right", size="5%", pad=0.05)

#     ax.axis('off')
#     cbar = plt.colorbar(scatter, cax=cax)
#     cbar.set_label("Stability (Max: 1)")

#     if out_path is not None:
#         plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
#     if verbose:
#         plt.show()


def map_pointcloud_to_image(nusc,
                            sps_map,
                            pointsensor_token: str,
                            camera_token: str,
                            min_dist: float = 1.0):
    """
    Given a point sensor (lidar/radar) token and camera sample_data token, load pointcloud and map it to the image
    plane.

    Returns:
        points_static (np.ndarray): Projected image points for static points.
        coloring_static (np.ndarray): SPS scores for static points.
        points_dynamic (np.ndarray): Projected image points for dynamic points.
        im (Image): The camera image.
    """
    ransac_solver = RANSACSolver(threshold=0.15, max_iter=10, outdir='output_dpr')

    cam = nusc.get('sample_data', camera_token)
    pointsensor = nusc.get('sample_data', pointsensor_token)

    pcl_path = osp.join(nusc.dataroot, pointsensor['filename'])
    pc = RadarPointCloud.from_file(pcl_path)
    im = Image.open(osp.join(nusc.dataroot, cam['filename']))

    ransac_input_pcl = pc.points.T
    info = [
        ['', -1],
        'RADAR',
        ransac_input_pcl.shape[0]
    ]
    best_mask, _, _ = ransac_solver.ransac_nusc(info=info, pcl=ransac_input_pcl, vis=False)

    # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
    # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
    cs_record = nusc.get('calibrated_sensor', pointsensor['calibrated_sensor_token'])
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix)
    pc.translate(np.array(cs_record['translation']))

    # Second step: transform from ego to the global frame.
    poserecord = nusc.get('ego_pose', pointsensor['ego_pose_token'])
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix)
    pc.translate(np.array(poserecord['translation']))

    # Extract global points and create an index array for all points.
    global_points = pc.points.T  # Shape (N_points, 4)
    N_points = global_points.shape[0]
    total_points_indices = np.arange(N_points)

    # Extract static global points for computing SPS scores.
    static_global_points = global_points[best_mask]
    sps_score = get_sps_labels(sps_map, static_global_points)

    # Create an array of SPS scores aligned with all points.
    sps_scores_full = np.zeros(N_points)
    sps_scores_full[best_mask] = sps_score

    # Third step: transform from global into the ego vehicle frame for the timestamp of the image.
    poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
    pc.translate(-np.array(poserecord['translation']))
    pc.rotate(Quaternion(poserecord['rotation']).rotation_matrix.T)

    # Fourth step: transform from ego into the camera.
    cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
    pc.translate(-np.array(cs_record['translation']))
    pc.rotate(Quaternion(cs_record['rotation']).rotation_matrix.T)

    # Fifth step: project the point cloud onto the image plane.
    depths = pc.points[2, :]  # Length N_points
    points = view_points(pc.points[:3, :], np.array(cs_record['camera_intrinsic']), normalize=True)

    # Create a mask to filter out points outside the image.
    mask = np.ones(depths.shape[0], dtype=bool)
    mask = np.logical_and(mask, depths > min_dist)
    mask = np.logical_and(mask, points[0, :] > 1)
    mask = np.logical_and(mask, points[0, :] < im.size[0] - 1)
    mask = np.logical_and(mask, points[1, :] > 1)
    mask = np.logical_and(mask, points[1, :] < im.size[1] - 1)

    # Apply the mask to points, best_mask, and sps_scores_full.
    points = points[:, mask]
    best_mask = best_mask[mask]
    sps_scores_full = sps_scores_full[mask]

    # Separate static and dynamic points.
    static_indices = np.where(best_mask)[0]
    dynamic_indices = np.where(~best_mask)[0]

    points_static = points[:, static_indices]
    points_dynamic = points[:, dynamic_indices]
    coloring_static = sps_scores_full[static_indices]

    return points_static, coloring_static, points_dynamic, im



def render_pointcloud_in_image(nusc,
                               sps_map,
                               sample_token: str,
                               dot_size: int = 5,
                               pointsensor_channel: str = 'LIDAR_TOP',
                               camera_channel: str = 'CAM_FRONT',
                               out_path: str = None,
                               ax=None,
                               verbose: bool = True):
    """
    Scatter-plots a pointcloud on top of an image, with static points colored by SPS scores and dynamic points
    as blue crosses.

    Args:
        nusc: NuScenes dataset object.
        sps_map: Stability scores map.
        sample_token: Sample token.
        dot_size: Scatter plot dot size.
        pointsensor_channel: RADAR or LIDAR channel name, e.g. 'LIDAR_TOP'.
        camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        out_path: Optional path to save the rendered figure to disk.
        ax: Axes onto which to render.
        verbose: Whether to display the image in a window.

    Returns:
        None
    """
    from mpl_toolkits.axes_grid1 import make_axes_locatable
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    sample_record = nusc.get('sample', sample_token)

    # Get the sensor and camera tokens.
    pointsensor_token = nusc.get('sample_data', sample_record['data'][pointsensor_channel])['token']
    camera_token = nusc.get('sample_data', sample_record['data'][camera_channel])['token']

    # Get the points and image using the updated map_pointcloud_to_image function.
    points_static, coloring_static, points_dynamic, im = map_pointcloud_to_image(
        nusc, sps_map, pointsensor_token, camera_token
    )

    # Define the custom colormap.
    cdict = {
        'red':   ((0.0, 1.0, 1.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 0.0, 0.0)),
        'green': ((0.0, 0.0, 0.0),
                  (0.5, 1.0, 1.0),
                  (1.0, 1.0, 1.0)),
        'blue':  ((0.0, 0.0, 0.0),
                  (0.5, 0.0, 0.0),
                  (1.0, 0.0, 0.0))
    }
    gyr_cmap = LinearSegmentedColormap('GYR', cdict)

    # Normalize the coloring array to [0, 1].
    norm = plt.Normalize(vmin=0, vmax=1)
    coloring_normalized = norm(coloring_static)

    # Apply the colormap to the normalized coloring array.
    coloring_rgb = gyr_cmap(coloring_normalized)

    # Initialize axes.
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(9, 16))
        fig.canvas.set_window_title(sample_token)
    else:
        ax.set_title(camera_channel)
    ax.imshow(im)

    # Plot static points with coloring.
    scatter = ax.scatter(points_static[0, :], points_static[1, :],
                         c=coloring_static, s=dot_size, cmap=gyr_cmap)

    # Plot dynamic points as blue crosses.
    ax.scatter(points_dynamic[0, :], points_dynamic[1, :],
               c='blue', marker='x', s=dot_size)

    # Create colorbar for static points.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(scatter, cax=cax)
    cbar.set_label("Stability (Max: 1)")

    ax.axis('off')

    if out_path is not None:
        plt.savefig(out_path, bbox_inches='tight', pad_inches=0, dpi=200)
    if verbose:
        plt.show()


def setup_directories(base_dir, scene_name):
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(os.path.join(base_dir, scene_name), exist_ok=True)
    os.makedirs(os.path.join(base_dir, scene_name, 'frames'), exist_ok=True)

def save_plots_for_scene(nusc, base_dir, scene_name, sample_token, sps_map, num_sweeps=5):
    sample = nusc.get('sample', sample_token)
    frame_index = 0

    while sample_token:
        fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 8), dpi=300)
        plot_path = os.path.join(base_dir, scene_name, 'frames', f'{frame_index}.png')
        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
        
        top_pos = [0.05, 0.52, 0.9, 0.46]  # [left, bottom, width, height]
        bottom_pos = [0.05, 0.02, 0.9, 0.46]  # [left, bottom, width, height]

        nusc.render_sample_data(sample['data']['RADAR_FRONT'], nsweeps=num_sweeps, underlay_map=True, with_anns=True, ax=ax_top)
        # top_pos = ax_top.get_position()
        # ax_top.set_position([top_pos.x0, top_pos.y0, top_pos.width, top_pos.height])
        ax_top.set_position(top_pos)

        render_pointcloud_in_image(nusc, sps_map, sample['token'], pointsensor_channel='RADAR_FRONT', ax=ax_bottom)
        # bottom_pos = ax_bottom.get_position()
        # ax_bottom.set_position([bottom_pos.x0, bottom_pos.y0, bottom_pos.width, bottom_pos.height])
        ax_bottom.set_position(bottom_pos)
        
        ax_top.axis('off')
        ax_bottom.axis('off')

        # ax_top.set_title(f"Scene {scene_name} | KeyFrame {frame_index}", fontsize=12)
        plt.tight_layout()
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Move to next sample
        sample_token = sample['next']
        if sample_token:
            sample = nusc.get('sample', sample_token)
        frame_index += 1

def create_video_from_plots(base_dir, scene_name):
    plot_dir = os.path.join(base_dir, scene_name, 'frames')
    num_frames = len(os.listdir(plot_dir))
    sample_image = cv2.imread(os.path.join(plot_dir, '0.png'))

    
    height, width, _ = sample_image.shape
    
    video_height = height # 2 * size[1] # height1 + height2
    video_width = width
    
    out_video_path = os.path.join(base_dir, scene_name, f'{scene_name}.mp4')
    out = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), 5, (video_width, video_height))

    
    for i in range(num_frames):
        img = cv2.resize(cv2.imread(os.path.join(plot_dir, f'{i}.png')), (video_width, video_height))
        out.write(img)
    out.release()
