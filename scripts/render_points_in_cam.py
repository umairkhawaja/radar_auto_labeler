from datasets.nuscenes import *
from utils.visualization import map_pointcloud_to_image, render_pointcloud_in_image
import os
import cv2
import numpy as np
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