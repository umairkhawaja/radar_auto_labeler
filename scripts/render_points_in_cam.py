from datasets.nuscenes import *
from utils.visualization import map_pointcloud_to_image, render_pointcloud_in_image
import os
import cv2
import numpy as np
from moviepy.editor import ImageSequenceClip
import open3d as o3d
import matplotlib.pyplot as plt
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from pyquaternion import Quaternion
from nuscenes.utils.geometry_utils import view_points, transform_matrix

import os
import cv2
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud


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
        img = cv2.imread(os.path.join(plot_dir, f'{i}.png'))
        out.write(img)
    out.release()


data_dir = "/shared/data/nuScenes/"
sensors = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {
    vname : NuScenes(dataroot=data_dir, version=version, verbose=False)\
    for vname,version in versions.items()
}

import pandas as pd
sps_df = pd.read_json('nuscenes_scenes_df.json')
sps_df.head()



from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
plt.ioff()


ref_frame = 'global'
num_sweeps = 5
ref_sensor = None
apply_dpr = True
filter_points = False
dpr_thresh = 0.5

for i,row in tqdm(sps_df.iterrows(), total=len(sps_df)):

    ref_scene_name = row['scene_name']
    ref_split = row['split']
    closest_scenes = row['closest_scenes_data']
    seq = int(ref_scene_name.split("-")[-1])

    dataset_sequence = NuScenesMultipleRadarMultiSweeps(
        data_dir=data_dir,
        nusc=nuscenes_exp[ref_split],
        sequence=seq,
        sensors=sensors,
        nsweeps=num_sweeps,
        ref_frame=ref_frame,
        ref_sensor=ref_sensor,
        apply_dpr=apply_dpr,
        filter_points=filter_points,
        ransac_threshold=dpr_thresh

    )


    labelled_map_path = f"labelled_maps/{ref_scene_name}.asc"
    sps_map = np.loadtxt(labelled_map_path, delimiter=' ', skiprows=1)
    sample_token = dataset_sequence.scene['first_sample_token']
    base_dir = 'output_videos'

    setup_directories(base_dir, ref_scene_name)
    save_plots_for_scene(nuscenes_exp[ref_split], base_dir, ref_scene_name, sample_token, sps_map)
    create_video_from_plots(base_dir, ref_scene_name)