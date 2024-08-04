import numpy as np
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud
from nuscenes.utils.geometry_utils import view_points, transform_matrix
from pyquaternion import Quaternion
from matplotlib.animation import FuncAnimation, PillowWriter
import os.path as osp


def points_in_box(corners, points: np.ndarray, wlh_factor: float = 1.0):
    """
    Checks whether points are inside the box.

    Picks one corner as reference (p1) and computes the vector to a target point (v).
    Then for each of the 3 axes, project v onto the axis and compare the length.
    Inspired by: https://math.stackexchange.com/a/1552579
    :param box: <Box>.
    :param points: <np.float: 3, n>.
    :param wlh_factor: Inflates or deflates the box.
    :return: <np.bool: n, >.
    """

    p1 = corners[:, 0]
    p_x = corners[:, 4]
    p_y = corners[:, 1]
    p_z = corners[:, 3]

    i = p_x - p1
    j = p_y - p1
    k = p_z - p1

    v = points - p1.reshape((-1, 1))

    iv = np.dot(i, v)
    jv = np.dot(j, v)
    kv = np.dot(k, v)

    mask_x = np.logical_and(0 <= iv, iv <= np.dot(i, i))
    mask_y = np.logical_and(0 <= jv, jv <= np.dot(j, j))
    mask_z = np.logical_and(0 <= kv, kv <= np.dot(k, k))
    mask = np.logical_and(np.logical_and(mask_x, mask_y), mask_z)

    return mask


def process_scene(nusc: NuScenes, scene_token: str, distance_threshold: float = 1.0, zoom_level: int = 5):
    scene_record = nusc.get('scene', scene_token)
    sample_token = scene_record['first_sample_token']
    frames = []
    radar_map = []

    while sample_token != '':
        sample = nusc.get('sample', sample_token)
        radar_sample_data_tokens = [sample['data'][c] for c in sensors]  # Use the appropriate radar sensor channel
        lidar_sample_data_token = sample['data']['LIDAR_TOP']  # Use the appropriate LiDAR sensor channel

        # Project LiDAR points to global frame
        lidar_sd_record = nusc.get('sample_data', lidar_sample_data_token)
        pcl_path = osp.join(nusc.dataroot, lidar_sd_record['filename'])
        pc = LidarPointCloud.from_file(pcl_path)
        
        cs_record = nusc.get('calibrated_sensor', lidar_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', lidar_sd_record['ego_pose_token'])
        
        global_from_car = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']))
        car_from_sensor = transform_matrix(cs_record['translation'], Quaternion(cs_record['rotation']))
        
        lidar_sensor_to_global = np.dot(global_from_car, car_from_sensor)
        lidar_points = view_points(pc.points[:3, :], lidar_sensor_to_global, normalize=False)

        # Get radar points in global frame
        all_points = []
        for token in radar_sample_data_tokens:
            radar_sd_record = nusc.get('sample_data', token)
            radar_sample_rec = nusc.get('sample', radar_sd_record['sample_token'])
            radar_pc, times = RadarPointCloud.from_file_multisweep(nusc, radar_sample_rec, radar_sd_record['channel'], 'RADAR_FRONT')
            all_points.append(radar_pc.points)
        all_points = np.hstack(all_points)
        
        radar_pc_all = RadarPointCloud(all_points)
        
        
        radar_cs_record = nusc.get('calibrated_sensor', radar_sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', radar_sd_record['ego_pose_token'])
        
        global_from_car = transform_matrix(pose_record['translation'], Quaternion(pose_record['rotation']))
        car_from_sensor = transform_matrix(radar_cs_record['translation'], Quaternion(radar_cs_record['rotation']))
        
        sensor_to_global = np.dot(global_from_car, car_from_sensor)
        radar_points = view_points(radar_pc_all.points[:3, :], sensor_to_global, normalize=False)

        # Render radar and lidar points
        fig, ax = plt.subplots(1, 1, figsize=(16, 8))
        ax.scatter(radar_points[0, :], radar_points[1, :], c='blue', s=1, label='Radar Points')
        ax.scatter(lidar_points[0, :], lidar_points[1, :], c='red', s=1, label='LiDAR Points')

        _, boxes, _ = nusc.get_sample_data(lidar_sample_data_token, use_flat_vehicle_coordinates=False)
        # Transform boxes to the global frame
        for box in boxes:
            c = 'black' #np.array(nusc.get_color(box.name)) / 255.0
            box.render(ax, view=lidar_sensor_to_global, colors=(c, c, c))
            # Add class labels
            corners = view_points(box.corners(), lidar_sensor_to_global, False)
            ax.text(corners[0, 0], corners[1, 0], box.name, color='green')
            
            # Check which radar points lie within this box
            in_box_mask = points_in_box(corners, radar_points)            
            stable_labels = 1 - in_box_mask.astype(int)
            radar_pc_sps = np.vstack([radar_points[:3, :], stable_labels])
            radar_map.append(radar_pc_sps.T)
            radar_points_in_box = radar_points[:, in_box_mask]
            ax.scatter(radar_points_in_box[0, :], radar_points_in_box[1, :], c='yellow', s=1)

        # ax.plot(0, 0, 'x', color='red')
        # ax.set_xlim(-40, 40)
        # ax.set_ylim(-40, 40)

        if zoom_level != -1:
            all_pos = np.hstack([radar_points[:3,:], lidar_points[:3,:]]).T
            x_mean = np.mean(all_pos[:,0])
            y_mean = np.mean(all_pos[:,1])
            x_std = np.std(all_pos[:,0])
            y_std = np.std(all_pos[:,1])
            
            std_dev_range = zoom_level

            x_limits = [x_mean - std_dev_range*x_std, x_mean + std_dev_range*x_std]
            y_limits = [y_mean - std_dev_range*y_std, y_mean + std_dev_range*y_std]

            ax.set_xlim(x_limits)
            ax.set_ylim(y_limits)

        ax.set_title('Radar and LiDAR Points in Global Frame with LiDAR detections')
        ax.legend()
        ax.set_aspect('equal')
        ax.axis('off')
        ax.set_xticks([])
        ax.set_yticks([])

        fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
        ax.margins(0,0)

        fig.canvas.draw()
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(frame)
        plt.close(fig)
        
        # plt.show()

        # Associate radar and lidar points
        associations = []
        for i, radar_point in enumerate(radar_points.T):
            distances = np.linalg.norm(lidar_points.T - radar_point, axis=1)
            min_distance = np.min(distances)
            if min_distance < distance_threshold:
                lidar_index = np.argmin(distances)
                associations.append((i, lidar_index))

        print(f'Frame {sample_token}: {len(associations)} associations found')

        sample_token = sample['next']
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    ax.margins(0,0)
    im = ax.imshow(frames[0])

    def update(frame):
        im.set_array(frame)
        return [im]
    
    ani = FuncAnimation(fig, update, frames=frames, blit=True)
    plt.close(fig)
    return ani, radar_map

# Example usage
scene_token = dataset_sequence.scene['token']
ani, radar_map = process_scene(nuscenes_exp['trainval'], scene_token, zoom_level=5)
# writer = PillowWriter(fps=2)
# ani.save('scene_animation.gif', writer=writer)
radar_map = np.concatenate(radar_map)

np.save('scene-0345_lidar_labels.npy', radar_map)