from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from autolabeler import AutoLabeler
import pandas as pd
import os.path as osp
import nuscenes
from pathlib import Path
from nuscenes import NuScenes


ICP_FILTERING = True
SEARCH_IN_RADIUS=True
RADIUS=1
USE_LIDAR_LABELS = False
USE_OCCUPANCY_PRIORS = True

DF_PATH  = 'nuscenes_scenes_df.json'
DATA_DIR = "/home/umair/workspace/datasets/nuscenes_radar/"
SENSORS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

OUTPUT_DIR = 'output'
OCTOMAP_DIR = osp.join(OUTPUT_DIR, 'octomaps')
LABELS_DIR = osp.join(OUTPUT_DIR, 'labelled_maps')
PLOTS_DIR = osp.join(OUTPUT_DIR, 'plots')
[Path(d).mkdir(parents=True, exist_ok=True) for d in [OUTPUT_DIR, OCTOMAP_DIR, LABELS_DIR, PLOTS_DIR]]


versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {
    vname : NuScenes(dataroot=DATA_DIR, version=version, verbose=False)\
    for vname,version in versions.items()
}
sps_df = pd.read_json(DF_PATH)


ref_frame = 'global'
num_sweeps = 1
ref_sensor = None
apply_dpr = True
filter_points = False
dpr_thresh = 0.5
voxel_size = 1


for i,row in sps_df.iterrows():
    ref_scene_name = row['scene_name']
    ref_split = row['split']
    closest_scenes = row['closest_scenes_data']

    ## Create Dataloader for each scene in a record
    dataloaders = {ref_scene_name : NuScenesMultipleRadarMultiSweeps(
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
        dataloaders[matched_scene] = NuScenesMultipleRadarMultiSweeps(
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

    scene_maps = {name : np.vstack([dl[i][0] for i in range(dl.num_readings)]) for name,dl in dataloaders.items()}
    scene_poses = {name: dl.global_poses for name,dl in dataloaders.items()}


    if ICP_FILTERING:
        # Convert to Open3d
        pcd_dict = {key: convert_to_open3d_pcd(val) for key, val in scene_maps.items()}

        # Merge point clouds
        pcd_merged = o3d.geometry.PointCloud()
        for pcd in pcd_dict.values():
            pcd_merged += pcd

        
        # Align point clouds (pairwise alignment)
        keys = list(pcd_dict.keys())
        transforms = {}
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                trans_key = f"{keys[i]}_{keys[j]}"
                transforms[trans_key] = align_pointclouds(pcd_dict[keys[i]], pcd_dict[keys[j]])


        # Apply transformations
        for key, trans in transforms.items():
            src, tgt = key.split('_')
            pcd_dict[src].transform(trans)

        # Find overlapping points in the merged point cloud
        overlapping_indices = find_overlapping_points(pcd_merged)

        # Crop individual point clouds to keep only overlapping points and track indices
        cropped_pcd_dict = {}
        cropped_indices_dict = {}
        for key, pcd in pcd_dict.items():
            original_indices = get_original_indices(pcd_merged, pcd, overlapping_indices)
            overlapping_points = np.asarray(pcd.points)[list(original_indices), :]
            cropped_pcd = o3d.geometry.PointCloud()
            cropped_pcd.points = o3d.utility.Vector3dVector(overlapping_points)
            cropped_pcd_dict[key] = cropped_pcd
            cropped_indices_dict[key] = list(original_indices)

        # Convert Open3D cropped point clouds back to numpy arrays
        cropped_scene_maps = {key: np.asarray(val.points) for key, val in cropped_pcd_dict.items()}
        scene_maps = cropped_scene_maps
    
    if USE_LIDAR_LABELS:
        lidar_labels = {name: np.load(f'inputs/lidar_labels/{name}_lidar_labels.npy') for name,map in scene_maps.items()}
    else:
        lidar_labels = None

    
    if USE_OCCUPANCY_PRIORS:
        ## Create Dataloader for each scene in a record
        voxel_prior_dataloaders = {ref_scene_name : NuScenesMultipleRadarMultiSweeps(
            data_dir=DATA_DIR,
            nusc=nuscenes_exp[ref_split],
            sequence=int(ref_scene_name.split("-")[-1]),
            sensors=SENSORS,
            nsweeps=num_sweeps,
            ref_frame='global',
            ref_sensor=ref_sensor,
            apply_dpr=True,
            filter_points=False,
            ransac_threshold=dpr_thresh

        )}

        for matched_scene, data in closest_scenes.items():
            voxel_prior_dataloaders[matched_scene] = NuScenesMultipleRadarMultiSweeps(
            data_dir=DATA_DIR,
            nusc=nuscenes_exp[data['split']],
            sequence=int(matched_scene.split("-")[-1]),
            sensors=SENSORS,
            nsweeps=num_sweeps,
            ref_frame='global',
            ref_sensor=ref_sensor,
            apply_dpr=True,
            filter_points=False,
            ransac_threshold=dpr_thresh
        )

        scene_scans = {name: {i:dataloader[i][0][:,:3] for i in range(len(dataloader))} for name,dataloader in voxel_prior_dataloaders.items()}
        scene_voxel_maps = {name : create_voxel_map(scans, voxel_size) for name,scans in scene_scans.items()}
    else:
        scene_voxel_maps = None

    
    # Build Octomaps
    scene_octomaps = {name: build_octomap(dl) for name,dl in dataloaders.items()}
    for name, omap in scene_octomaps.items():
        path = osp.join(OCTOMAP_DIR, f'{name}.ot')
        save_octomap(path, omap)

    # Setup AutoLabeler

    sps_labeler = AutoLabeler(
        scene_maps=scene_maps, ref_map_id=ref_scene_name,
        scene_octomaps=scene_octomaps, lidar_labels=lidar_labels, 
        dynamic_priors=scene_voxel_maps,  use_octomaps=True,
        search_in_radius=SEARCH_IN_RADIUS, radius=RADIUS,
        downsample=True, voxel_size=0.5
    )

    sps_labeler.label_maps()

    for name, labeled_map in sps_labeler.labelled_maps.items():
        save_path = osp.join(PLOTS_DIR, 'labeled_maps')
        Path(save_path).mkdir(parents=True, exist_ok=True)
        save_path = osp.join(save_path, f'{name}.jpg')
        sps_labeler.save_bev_plot(labeled_map, save_path, title="Labelled Map in Bird's Eye View", size=1)
    
    
    for map_name, lmap in sps_labeler.labelled_maps.items():
        points = lmap[:, :3]
        stable_probs = lmap[:,-1]
        filename = f"{LABELS_DIR}/{map_name}.asc"
        np.savetxt(filename,
                np.hstack([points,stable_probs.reshape(-1,1)]),
                fmt='%.6f', delimiter=' ',
                header='x y z stable_prob',
                comments='')