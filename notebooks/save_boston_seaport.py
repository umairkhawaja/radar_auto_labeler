import repackage
repackage.up()

from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from utils.postprocessing import *
from utils.ransac_solver import RANSACSolver
from utils.motion_estimation import *
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
from utils.visualization import map_pointcloud_to_image, render_pointcloud_in_image, plot_maps
from utils.motion_estimation import remove_dynamic_points
matplotlib.use('Agg')
plt.ioff()
import concurrent.futures
# from multiprocess.pool import Pool



NUM_WORKERS = min(cpu_count(), 12)  # Adjust this number based on your machine's capabilities
DF_PATH = '../sps_nuscenes_more_matches_df.json' # Sticking to this since odom/loc benchmarking was done on this to compare experiments
sps_df = pd.read_json(DF_PATH)


DATA_DIR = "/shared/data/nuScenes/"
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {
    vname: NuScenes(dataroot=DATA_DIR, version=version, verbose=False)
    for vname, version in versions.items()
}


REF_FRAME = None
REF_SENSOR = None
FILTER_POINTS = False

NUM_SWEEPS = 5
APPLY_DPR = True
DPR_THRESH = 0.15
OCTOMAP_RESOLUTION = 0.15 # For dividing space, for lidar 0.1 is suitable but since radar is sparse a larger value might be better
VOXEL_SIZE = 0.01

ICP_FILTERING = True
SEARCH_IN_RADIUS = True
RADIUS = 2
USE_LIDAR_LABELS = False
USE_OCCUPANCY_PRIORS = True
FILTER_BY_POSES = False
FILTER_BY_RADIUS = False
FILTER_OUT_OF_BOUNDS = False
USE_COMBINED_MAP = False


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
        nsweeps=NUM_SWEEPS,
        ref_frame=REF_FRAME,
        ref_sensor=REF_SENSOR,
        apply_dpr=False,
        filter_points=FILTER_POINTS,
        ransac_threshold=-1,
        reformat_pcl=False
    )}

    for matched_scene, data in closest_scenes.items():
        row_dls[matched_scene] = NuScenesMultipleRadarMultiSweeps(
            data_dir=DATA_DIR,
            nusc=nuscenes_exp[data['split']],
            sequence=int(matched_scene.split("-")[-1]),
            sensors=SENSORS,
            nsweeps=NUM_SWEEPS,
            ref_frame=REF_FRAME,
            ref_sensor=REF_SENSOR,
            apply_dpr=False,
            filter_points=FILTER_POINTS,
            ransac_threshold=-1,
            reformat_pcl=False
        )

    # Process and downsample point clouds
    scene_pointclouds = {name : [dl[i][0] for i in range(dl.num_readings)] for name,dl in dataloaders.items()}
    scene_calibs = {name : [dl[i][1] for i in range(dl.num_readings)] for name,dl in dataloaders.items()}
    scene_poses = {name: dl.global_poses for name,dl in dataloaders.items()}

    # downsampled_pointclouds = {
    #     name: o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcls[:,:3])).voxel_down_sample(voxel_size=0.05).points
    #     for name, pcls in maps.items()
    # }
    # Consolidate downsampled point clouds
    # scene_map = {name: pcls for name, pcls in downsampled_pointclouds.items()}

    ## Applying DPR
    scene_pointclouds_dpr_masks, global_scene_pointclouds = remove_dynamic_points(scene_pointclouds, scene_calibs, scene_poses, SENSORS, filter_sensors=True, dpr_thresh=DPR_THRESH, save_vis=False)

    scene_dpr_masks = {name : np.hstack(masks) for name,masks in scene_pointclouds_dpr_masks.items()}
    scene_maps = {name : np.vstack(pcls) for name,pcls in global_scene_pointclouds.items()}

    static_scene_maps = {}
    dynamic_scene_maps = {}
    dpr_sps_maps = {}

    for name in scene_maps:
        indices = scene_dpr_masks[name]
        map_pcl = scene_maps[name]
        filtered_map_pcl = map_pcl[indices]
        static_scene_maps[name] = filtered_map_pcl
        dynamic_scene_maps[name] = map_pcl[~indices]

        dpr_sps_pcl = np.hstack((map_pcl, np.array(indices).astype(np.int).reshape(-1, 1)))
        dpr_sps_maps[name] = dpr_sps_pcl

    dynamic_scene_maps = filter_maps_icp(dynamic_scene_maps, alignment_thresh=0.5, overlapping_thresh=0.25)

    
    return global_scene_pointclouds, static_scene_maps, dynamic_scene_maps, scene_poses, row_dls

# Parallel processing using ThreadPoolExecutor
scene_pointclouds = {}
dynamic_scene_maps = {}
scene_maps = {}
scene_poses = {}
dataloaders = {}

with concurrent.futures.ThreadPoolExecutor() as executor:
    future_to_row = {executor.submit(process_row, row): i for i, row in sps_df.iterrows()}
    
    for future in concurrent.futures.as_completed(future_to_row):
        i = future_to_row[future]
        # try:
        pointclouds, scene_map, dynamic_scene_map, scene_pose, row_dls = future.result()
        # If using downsampled o3d pcl
        # scene_map = {name: np.asarray(o3d_map) for name, o3d_map in scene_map.items()}
        scene_pointclouds.update(pointclouds)
        scene_maps.update(scene_map)
        dynamic_scene_maps.update(dynamic_scene_map)
        scene_poses.update(scene_pose)
        dataloaders.update(row_dls)
    # except Exception as e:
        # print(f"Row {i} generated an exception: {e}")

if FILTER_BY_POSES:
    scene_maps, scene_poses = create_filtered_maps(scene_poses, scene_pointclouds, threshold=1)

if FILTER_BY_RADIUS:
    scene_maps, scene_poses = create_filtered_maps_with_radius(scene_poses, scene_pointclouds, threshold=1)

if ICP_FILTERING:
    scene_maps = filter_maps_icp(scene_maps)

if USE_LIDAR_LABELS:
    lidar_labels = {name: np.load(f'inputs/lidar_labels/{name}_lidar_labels.npy') for name, map in scene_maps.items()}
else:
    lidar_labels = None

if USE_OCCUPANCY_PRIORS:
    scene_scans = {name: {i:scene_pointclouds[name][i][:,:3] for i in range(len(dataloader))} for name,dataloader in dataloaders.items()}
    scene_voxel_maps = {name: create_voxel_map(scans, VOXEL_SIZE) for name, scans in scene_scans.items()}
else:
    scene_voxel_maps = None

scene_octomaps = {name: build_octomap(dl, resolution=OCTOMAP_RESOLUTION) for name,dl in dataloaders.items()}


sps_labeler = AutoLabeler(
    scene_maps=scene_maps, ref_map_id=list(scene_poses.keys())[0], scene_poses=scene_poses,
    scene_octomaps=scene_octomaps, lidar_labels=lidar_labels,
    dynamic_priors=scene_voxel_maps, use_octomaps=True,
    search_in_radius=SEARCH_IN_RADIUS, radius=RADIUS, use_combined_map=USE_COMBINED_MAP,
    downsample=True, voxel_size=VOXEL_SIZE, filter_out_of_bounds=FILTER_OUT_OF_BOUNDS
)

sps_labeler.label_maps()


complete_sps_labelled_map = sps_labeler.labeled_environment_map

dynamic_sps_scene_points = {}
for name in dynamic_scene_maps:
    dyn_sps_pcl = np.hstack((dynamic_scene_maps[name][:,:3], np.zeros(len(dynamic_scene_maps[name])).astype(np.int).reshape(-1, 1)))
    dynamic_sps_scene_points[name] = dyn_sps_pcl


for scene_name, dyn_points in dynamic_sps_scene_points.items():
    complete_sps_labelled_map = np.vstack([complete_sps_labelled_map, dyn_points])


labeled_map = complete_sps_labelled_map
points = labeled_map[:, :3]
stable_probs = labeled_map[:, -1]

plt.figure(figsize=(10, 10))
plt.scatter(points[:, 0], points[:, 1], c=stable_probs, cmap='RdYlGn', s=0.05)
plt.colorbar(label='Stability')
plt.xlabel('X')
plt.ylabel('Y')
plt.savefig(f'boston-seaport_sps.jpg', dpi=300)

filename = f"boston-seaport_sps.asc"
np.savetxt(filename,
            np.hstack([points,stable_probs.reshape(-1,1)]),
            fmt='%.6f', delimiter=' ',
            header='x y z stable_prob',
            comments='')
