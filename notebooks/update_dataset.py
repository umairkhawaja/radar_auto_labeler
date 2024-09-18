import repackage
repackage.up()
import os
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from utils.postprocessing import *
from utils.ransac_solver import RANSACSolver
from utils.motion_estimation import *
from autolabeler import AutoLabeler
import pandas as pd
import numpy as np
from pathlib import Path
from nuscenes import NuScenes
from multiprocessing import Pool, cpu_count
import open3d as o3d

DATA_DIR = "/shared/data/nuScenes/"
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {vname: NuScenes(dataroot=DATA_DIR, version=version, verbose=False) for vname, version in versions.items()}

LABELLED_MAPS_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes/maps/'
DPR_MAPS_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes/maps_dpr/'
LOCAL_DPR_MAPS_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes/local_maps_dpr/'
[Path(d).mkdir(exist_ok=True, parents=True) for d in [DPR_MAPS_DIR, LOCAL_DPR_MAPS_DIR]]

DPR_THRESH=0.15
NUM_SWEEPS = 5
SENSORS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

DF_PATH = '../sps_nuscenes_more_matches_df.json'
sps_df = pd.read_json(DF_PATH)

def get_updated_data(scene_name, dataloader, labelled_map):

    scene_pointclouds = [dataloader[i][0] for i in range(dataloader.num_readings)]
    scene_calibs = [dataloader[i][1] for i in range(dataloader.num_readings)]
    scene_global_poses = dataloader.global_poses
    scene_local_poses = dataloader.local_poses

    scene_label_statistics = {}
    
    ransac_solver = RANSACSolver(threshold=DPR_THRESH, max_iter=10, outdir='output_dpr')
    scene_dpr_masks = []
    for index, (sensor_pcls, sensor_calibs) in enumerate(zip(scene_pointclouds, scene_calibs)):
        sensor_dpr_masks = {}
        for sensor, pcl, calib in zip(SENSORS, sensor_pcls, sensor_calibs):
            if sensor in ['RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']:
                ## Removing these points since Autoplace does not use these sensors either
                # DPR results get worse if we use them
                sensor_dpr_masks[sensor] = np.zeros(len(pcl)).astype(bool)
                continue
            
            if pcl.shape[0] > 1:
                info = [
                    [scene_name, index],
                    sensor,
                    pcl.shape[0]
                ]
                best_mask, _, _ = ransac_solver.ransac_nusc(info=info, pcl=pcl, vis=(False and sensor == 'RADAR_FRONT'))
                sensor_dpr_masks[sensor] = best_mask
        scene_dpr_masks.append(sensor_dpr_masks)
    
    ego_scene_pcls = []
    local_scene_pcls = []
    global_scene_pcls = []
    scene_sensor_ids = []
    merged_scene_dpr_mask = []

    for frame_dpr_masks, frame_pcls, frame_global_pose, frame_local_pose, frame_calibs in zip(scene_dpr_masks, scene_pointclouds, scene_global_poses, scene_local_poses, scene_calibs):
        frame_ego_pcl = []
        frame_local_pcl = []
        frame_global_pcl = []
        frame_sensor_ids = []
        merged_frame_dpr_masks = []

        for sensor, sensor_pcl, sensor_calib in zip(SENSORS, frame_pcls, frame_calibs):
            dpr_mask = frame_dpr_masks[sensor]
            merged_frame_dpr_masks.append(dpr_mask)

            static_sensor_pcl = sensor_pcl[dpr_mask, :]

            num_dynamic_points = len(sensor_pcl) - len(static_sensor_pcl)
            if 'scene_dynamic_points' in scene_label_statistics:
                scene_label_statistics['scene_dynamic_points']+=num_dynamic_points
            else:
                scene_label_statistics['scene_dynamic_points'] = num_dynamic_points

            ego_pcl = transform_doppler_points(sensor_calib, static_sensor_pcl)
            ego_local_pcl = transform_doppler_points(frame_local_pose, ego_pcl)
            global_pcl = transform_doppler_points(frame_global_pose, ego_pcl)
            
            frame_ego_pcl.append(ego_pcl)
            frame_local_pcl.append(ego_local_pcl)
            frame_global_pcl.append(global_pcl)
            frame_sensor_ids.append([sensor] * len(static_sensor_pcl))
        
        ego_scene_pcls.append(np.vstack(frame_ego_pcl))
        local_scene_pcls.append(np.vstack(frame_local_pcl))
        global_scene_pcls.append(np.vstack(frame_global_pcl))
        scene_sensor_ids.append(np.hstack(frame_sensor_ids))
        merged_scene_dpr_mask.append(np.hstack(merged_frame_dpr_masks))

    
    static_labels = []
    for pcl in global_scene_pcls:
        stable_scores = get_sps_labels(labelled_map, pcl)
        static_labels.append(stable_scores)


    scene_global_static_labelled_map = []
    scene_local_static_labelled_map = []

    for i in range(len(static_labels)):
        local_labelled_scan = np.hstack([local_scene_pcls[i], static_labels[i].reshape(-1 ,1) ])
        global_labelled_scan = np.hstack([global_scene_pcls[i], static_labels[i].reshape(-1 ,1)])

        scene_local_static_labelled_map.append(local_labelled_scan)
        scene_global_static_labelled_map.append(global_labelled_scan)

    scene_local_static_labelled_map = np.vstack(scene_local_static_labelled_map)
    scene_global_static_labelled_map = np.vstack(scene_global_static_labelled_map)


    num_unstable_points = np.sum(scene_local_static_labelled_map[:, -1] <= 0.5)
    num_stable_points = np.sum(scene_local_static_labelled_map[:, -1] > 0.5)

    scene_label_statistics['scene_unstable_points'] = num_unstable_points
    scene_label_statistics['scene_stable_points'] = num_stable_points

    updated_data_dict = {
        'labelled_local_static_map' : scene_local_static_labelled_map,
        'labelled_global_static_map' : scene_local_static_labelled_map,

        'static_ego_frames' : ego_scene_pcls,
        'static_frame_labels' : static_labels,

        'scene_label_statistics' : scene_label_statistics,
    }

    return updated_data_dict

    

def process_scene(scene_name, split):
    try:
        old_labelled_map = np.loadtxt(os.path.join(LABELLED_MAPS_DIR, scene_name + ".asc"), skiprows=1)
        seq = int(scene_name.split("-")[-1])

        dataloader = NuScenesMultipleRadarMultiSweeps(
            data_dir=DATA_DIR,
            nusc=nuscenes_exp[split],
            sequence=seq,
            sensors=SENSORS,
            nsweeps=NUM_SWEEPS,
            ref_frame=None,
            ref_sensor=None,
            apply_dpr=False,
            filter_points=False,
            ransac_threshold=-1,
            combine_velocity_components=False
        )
        
        update_data_dict = get_updated_data(scene_name, dataloader, old_labelled_map)

        local_lmap = update_data_dict['labelled_local_static_map']
        global_lmap = update_data_dict['labelled_global_static_map']
        ego_frames = update_data_dict['static_ego_frames']
        scan_stablility_scores = update_data_dict['static_frame_labels']
        
        local_map_file = os.path.join(LOCAL_DPR_MAPS_DIR, f'{scene_name}.asc')
        global_map_file = os.path.join(DPR_MAPS_DIR, f'{scene_name}.asc')
        
        np.savetxt(local_map_file, local_lmap, fmt='%.6f', delimiter=' ', header='x y z RCS v_x v_y cv_x cv_y stable_prob', comments='')
        np.savetxt(global_map_file, global_lmap, fmt='%.6f', delimiter=' ', header='x y z RCS v_x v_y cv_x cv_y stable_prob', comments='')

        scans_dir = f'/home/umair/workspace/radar_sps_datasets/nuscenes/sequence/{scene_name}/scans/'
        scans_dpr_dir = f'/home/umair/workspace/radar_sps_datasets/nuscenes/sequence/{scene_name}/scans_dpr/'
        labels_dpr_dir = f'/home/umair/workspace/radar_sps_datasets/nuscenes/sequence/{scene_name}/labels_dpr/'

        [Path(p).mkdir(parents=True, exist_ok=True) for p in [scans_dpr_dir, labels_dpr_dir]]
        
        scan_files = sorted(os.listdir(scans_dir), key=lambda f: float(''.join(filter(str.isdigit, f.split('.npy')[0]))))
        
        for name, scan, label in zip(scan_files, ego_frames, scan_stablility_scores):
            np.save(os.path.join(scans_dpr_dir, f'{name}'), scan)
            np.save(os.path.join(labels_dpr_dir, f'{name}'), label)

        return {**{'scene_name': scene_name}, **update_data_dict['scene_label_statistics']}
    except Exception as e:
        print(f"Error processing scene {scene_name}: {e}")
        exit(0)
        return None

if __name__ == '__main__':
    all_scenes_data = [(row['scene_name'], row['split']) for i, row in sps_df.iterrows()]
    for row in sps_df['closest_scenes_data']:
        all_scenes_data.extend([(scene, data['split']) for scene, data in row.items()])

    with Pool(8) as pool:
        results = pool.starmap(process_scene, all_scenes_data)

    # Save the scene label statistics
    all_scenes_label_statistics = [result for result in results if result is not None]
    pd.DataFrame(all_scenes_label_statistics).to_csv(f'{LABELLED_MAPS_DIR.replace("maps/", "")}/scene_label_statistics.csv', index=False)