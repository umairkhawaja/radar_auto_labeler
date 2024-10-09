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
# from multiprocess.pool import Pool



NUM_WORKERS = min(cpu_count(), 4)  # Adjust this number based on your machine's capabilities
DF_PATH = 'sps_nuscenes_more_matches_df.json' # Sticking to this since odom/loc benchmarking was done on this to compare experiments
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
DPR_THRESH = 0.5 # 0.15
OCTOMAP_RESOLUTION = 0.15 # For dividing space, for lidar 0.1 is suitable but since radar is sparse a larger value might be better
VOXEL_SIZE = 0.01

ICP_FILTERING = True
SEARCH_IN_RADIUS = True
RADIUS = 1
USE_LIDAR_LABELS = False
USE_OCCUPANCY_PRIORS = False
FILTER_BY_POSES = False
FILTER_BY_RADIUS = False
FILTER_OUT_OF_BOUNDS = False
USE_COMBINED_MAP = True


SENSORS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

OUTPUT_DIR = f'output_sw{NUM_SWEEPS}-dpr{DPR_THRESH if APPLY_DPR else 0}-r{RADIUS}'
OCTOMAP_DIR = osp.join(OUTPUT_DIR, 'octomaps')
LABELS_DIR = osp.join(OUTPUT_DIR, 'labelled_maps')
PLOTS_DIR = osp.join(OUTPUT_DIR, 'plots')
[Path(d).mkdir(parents=True, exist_ok=True) for d in [OUTPUT_DIR, OCTOMAP_DIR, LABELS_DIR, PLOTS_DIR]]


def process_scene(i, row):
    print(i)

    ref_scene_name = row['scene_name']
    ref_split = row['split']
    closest_scenes = row['closest_scenes_data']

    dataloaders = {ref_scene_name: NuScenesMultipleRadarMultiSweeps(
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
        combine_velocity_components=False

    )}

    for matched_scene, data in closest_scenes.items():
        dataloaders[matched_scene] = NuScenesMultipleRadarMultiSweeps(
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
            combine_velocity_components=False

        )

    scene_pointclouds = {name : [dl[i][0] for i in range(dl.num_readings)] for name,dl in dataloaders.items()}
    scene_calibs = {name : [dl[i][1] for i in range(dl.num_readings)] for name,dl in dataloaders.items()}
    scene_poses = {name: dl.global_poses for name,dl in dataloaders.items()}

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
    
    
    if FILTER_BY_POSES:
        scene_maps, scene_poses = create_filtered_maps(scene_poses, global_scene_pointclouds, threshold=1)
    
    if FILTER_BY_RADIUS:
        scene_maps, scene_poses = create_filtered_maps_with_radius(scene_poses, global_scene_pointclouds, threshold=1)

    if ICP_FILTERING:
        scene_maps = filter_maps_icp(static_scene_maps)

    if USE_LIDAR_LABELS:
        lidar_labels = {name: np.load(f'inputs/lidar_labels/{name}_lidar_labels.npy') for name, map in scene_maps.items()}
    else:
        lidar_labels = None

    if USE_OCCUPANCY_PRIORS:
        scene_scans = {name: {i:global_scene_pointclouds[name][i][:,:3] for i in range(len(dataloader))} for name,dataloader in dataloaders.items()}
        scene_voxel_maps = {name: create_voxel_map(scans, VOXEL_SIZE) for name, scans in scene_scans.items()}
    else:
        scene_voxel_maps = None

    
    scene_octomaps = {name: build_octomap(dl, resolution=OCTOMAP_RESOLUTION) for name,dl in dataloaders.items()}
    for name, omap in scene_octomaps.items():
        path = osp.join(OCTOMAP_DIR, f'{name}.ot')
        save_octomap(path, omap)

    

    sps_labeler = AutoLabeler(
        scene_maps=scene_maps, ref_map_id=ref_scene_name, scene_poses=scene_poses,
        scene_octomaps=scene_octomaps, lidar_labels=lidar_labels,
        dynamic_priors=scene_voxel_maps, use_octomaps=True,
        search_in_radius=SEARCH_IN_RADIUS, radius=RADIUS, use_combined_map=USE_COMBINED_MAP,
        downsample=True, voxel_size=VOXEL_SIZE, filter_out_of_bounds=FILTER_OUT_OF_BOUNDS
    )

    sps_labeler.label_maps()
        
        
    for map_name, lmap in sps_labeler.labelled_maps.items():
        # points = lmap[:, :3]
        # stable_probs = lmap[:, -1]

        # dyn_points = np.hstack((dynamic_scene_maps[name][:,:3], np.zeros(len(dynamic_scene_maps[name])).astype(np.int).reshape(-1, 1)))
        dyn_points = np.hstack((dynamic_scene_maps[name], np.ones(len(dynamic_scene_maps[name])).astype(np.int).reshape(-1, 1)))

        # points = np.vstack([points, dyn_points[:,:3]])
        # stable_probs = np.hstack([stable_probs, dyn_points[:,-1]])
        # combined_labelled_map = np.hstack([points, stable_probs.reshape(-1, 1)])

        combined_labelled_map = np.vstack([lmap, dyn_points])

        save_path = osp.join(PLOTS_DIR, 'labeled_maps')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = osp.join(save_path, f'{name}.jpg')
        sps_labeler.save_bev_plot(combined_labelled_map, save_path, title=f"{name} | DPR and Autolabeler labelled map", size=0.5)

        filename = f"{LABELS_DIR}/{map_name}.asc"
        np.savetxt(filename,
                #    np.hstack([points, stable_probs.reshape(-1, 1)]),
                   combined_labelled_map,
                   fmt='%.6f', delimiter=' ',
                #    header='x y z stable_prob',
                   header='x y z RCS v_x v_y cv_x cv_y stable_prob',
                   comments='')

if __name__ == '__main__':
    # with Pool(NUM_WORKERS) as pool:
    #     pool.starmap(process_scene, sps_df.iterrows())
    for i, row in sps_df.iterrows():
        process_scene(i, row)