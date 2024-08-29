import repackage
repackage.up()
import concurrent.futures
import open3d as o3d
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from utils.postprocessing import *
from autolabeler import AutoLabeler
import pandas as pd
import os.path as osp
import numpy as np
import open3d as o3d
from pathlib import Path
from nuscenes import NuScenes
from multiprocessing import cpu_count, Pool, set_start_method
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')
plt.ioff()

NUM_WORKERS = min(cpu_count(), 4)  # Adjust this number based on your machine's capabilities
DF_PATH = '../sps_nuscenes_more_matches_df.json'
sps_df = pd.read_json(DF_PATH)


ICP_FILTERING = True
SEARCH_IN_RADIUS = True
RADIUS = 2
USE_LIDAR_LABELS = False
USE_OCCUPANCY_PRIORS = True
FILTER_BY_POSES = False
FILTER_BY_RADIUS = False
FILTER_OUT_OF_BOUNDS = False
USE_COMBINED_MAP = False



ref_frame = 'global'
num_sweeps = 5
ref_sensor = 'RADAR_FRONT'
apply_dpr = False
filter_points = False
dpr_thresh = 0.0
voxel_size = 0.1


DATA_DIR = "/shared/data/nuScenes/"
SENSORS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}

nuscenes_exp = {
    vname: NuScenes(dataroot=DATA_DIR, version=version, verbose=False)
    for vname, version in versions.items()
}


# Define a function for processing each row
def process_row(row):
    ref_scene_name = row['scene_name']
    ref_split = row['split']
    closest_scenes = row['closest_scenes_data']

    row_dls = {ref_scene_name: NuScenesMultipleRadarMultiSweeps(
        data_dir=DATA_DIR,
        nusc=nuscenes_exp[ref_split],
        sequence=int(ref_scene_name.split("-")[-1]),
        sensors=SENSORS,
        nsweeps=num_sweeps,
        ref_frame=ref_frame,
        ref_sensor=ref_sensor,
        apply_dpr=apply_dpr,
        filter_points=filter_points,
        ransac_threshold=dpr_thresh
    )}

    for matched_scene, data in closest_scenes.items():
        row_dls[matched_scene] = NuScenesMultipleRadarMultiSweeps(
            data_dir=DATA_DIR,
            nusc=nuscenes_exp[data['split']],
            sequence=int(matched_scene.split("-")[-1]),
            sensors=SENSORS,
            nsweeps=num_sweeps,
            ref_frame=ref_frame,
            ref_sensor=ref_sensor,
            apply_dpr=apply_dpr,
            filter_points=filter_points,
            ransac_threshold=dpr_thresh
        )

    # Process and downsample point clouds
    pointclouds = {name : [dl[i][0] for i in range(dl.num_readings)] for name, dl in row_dls.items()}
    maps = {name : np.vstack(pcl) for name, pcl in pointclouds.items()}
    # downsampled_pointclouds = {
    #     name: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcls[:,:3])).voxel_down_sample(voxel_size=0.05).points
    #     for name, pcls in maps.items()
    # }
    
    # Consolidate downsampled point clouds
    # scene_map = {name: pcls for name, pcls in downsampled_pointclouds.items()}
    scene_map = maps
    scene_pose = {name: dl.global_poses for name, dl in row_dls.items()}
    
    return pointclouds, scene_map, scene_pose, row_dls

# Parallel processing using ThreadPoolExecutor
scene_pointclouds = {}
scene_maps = {}
scene_poses = {}
dataloaders = {}

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_row = {executor.submit(process_row, row): i for i, row in sps_df.iterrows()}
    
    for future in concurrent.futures.as_completed(future_to_row):
        i = future_to_row[future]
        # try:
        pointclouds, scene_map, scene_pose, row_dls = future.result()
        # If using downsampled o3d pcl
        # scene_map = {name: np.asarray(o3d_map) for name, o3d_map in scene_map.items()}
        scene_pointclouds.update(pointclouds)
        scene_maps.update(scene_map)
        scene_poses.update(scene_pose)
        dataloaders.update(row_dls)
    # except Exception as e:
        # print(f"Row {i} generated an exception: {e}")

if FILTER_BY_POSES:
    scene_maps, scene_poses = create_filtered_maps(scene_poses, scene_pointclouds, threshold=1)
elif FILTER_BY_RADIUS:
    scene_maps, scene_poses = create_filtered_maps_with_radius(scene_poses, scene_pointclouds, threshold=1)


if ICP_FILTERING:
    pcd_dict = {key: convert_to_open3d_pcd(val) for key, val in scene_maps.items()}
    pcd_merged = o3d.geometry.PointCloud()
    for pcd in pcd_dict.values():
        pcd_merged += pcd

    keys = list(pcd_dict.keys())
    transforms = {}
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            trans_key = f"{keys[i]}_{keys[j]}"
            transforms[trans_key] = align_pointclouds(pcd_dict[keys[i]], pcd_dict[keys[j]])

    for key, trans in transforms.items():
        src, tgt = key.split('_')
        pcd_dict[src].transform(trans)

    overlapping_indices = find_overlapping_points(pcd_merged)

    cropped_pcd_dict = {}
    cropped_indices_dict = {}
    for key, pcd in pcd_dict.items():
        original_indices = get_original_indices(pcd_merged, pcd, overlapping_indices)
        overlapping_points = np.asarray(pcd.points)[list(original_indices), :]
        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(overlapping_points)
        cropped_pcd_dict[key] = cropped_pcd
        cropped_indices_dict[key] = list(original_indices)

    cropped_scene_maps = {key: np.asarray(val.points) for key, val in cropped_pcd_dict.items()}
    scene_maps = cropped_scene_maps

if USE_LIDAR_LABELS:
    lidar_labels = {name: np.load(f'inputs/lidar_labels/{name}_lidar_labels.npy') for name, map in scene_maps.items()}
else:
    lidar_labels = None

if USE_OCCUPANCY_PRIORS:
    scene_voxel_maps = {name: create_voxel_map(scans[:,:3], voxel_size) for name, scans in scene_pointclouds.items()}
else:
    scene_voxel_maps = None

scene_octomaps = {name: build_octomap(dl) for name, dl in dataloaders.items()}

sps_labeler = AutoLabeler(
    scene_maps=scene_maps, ref_map_id=list(scene_poses.keys())[0], scene_poses=scene_poses,
    scene_octomaps=scene_octomaps, lidar_labels=lidar_labels,
    dynamic_priors=scene_voxel_maps, use_octomaps=True,
    search_in_radius=SEARCH_IN_RADIUS, radius=RADIUS, use_combined_map=USE_COMBINED_MAP,
    downsample=True, voxel_size=voxel_size, filter_out_of_bounds=FILTER_OUT_OF_BOUNDS
)

sps_labeler.label_maps()



lmap = sps_labeler.labeled_environment_map
points = lmap[:, :3]
stable_probs = lmap[:,-1]
filename = f"boston-seaport_sps_nodpr.asc"
np.savetxt(filename,
            np.hstack([points,stable_probs.reshape(-1,1)]),
            fmt='%.6f', delimiter=' ',
            header='x y z stable_prob',
            comments='')
