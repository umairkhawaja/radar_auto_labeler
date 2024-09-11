import repackage
repackage.up()
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
from tqdm import tqdm
from nuscenes.nuscenes import NuScenes
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.labelling import get_sps_labels
from utils.transforms import *
import os
import shutil
import numpy as np  # Added missing numpy import

# Load the DataFrame
sps_df = pd.read_json('../sps_nuscenes_more_matches_df.json')

# Configurations
ref_frame = None
num_sweeps = 5
ref_sensor = None
apply_dpr = False
filter_points = False
dpr_thresh = 0.75
data_dir = "/shared/data/nuScenes/"
EXP_NAME = 'output_sw5-dpr0.15-r1_combined_maps_more_matches'
BASE_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes'

if os.path.exists(BASE_DIR):
    shutil.rmtree(BASE_DIR)
    os.mkdir(BASE_DIR)
    
src_maps_dir = f"/home/umair/workspace/radar_auto_labeler/{EXP_NAME}/labelled_maps/"
dst_maps_dir = os.path.join(BASE_DIR, "maps")
shutil.copytree(src_maps_dir, dst_maps_dir)
sps_maps_dir = f"../{EXP_NAME}/labelled_maps/"

for i,row in sps_df.iterrows():
    ref_scene_name = row['scene_name']
    closest_scenes = row['closest_scenes']
    src_path = os.path.join(dst_maps_dir, f'{ref_scene_name}.asc')
    
    for scene in closest_scenes:
        dst_path = os.path.join(dst_maps_dir, f'{scene}.asc')
        shutil.copyfile(src_path, dst_path)


sensors = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}

nuscenes_exp = {
    vname : NuScenes(dataroot=data_dir, version=version, verbose=False)\
    for vname,version in versions.items()
}

def extract_data(dataset):
    num_frames = len(dataset)
    local_poses = dataset.local_poses
    global_poses = dataset.global_poses
    ego_timestamps = dataset.timestamps
    all_data = []

    for i in range(num_frames):
        frame_dict = {}
        (pointclouds, sps_scores), calibs, sensors, timestamps = dataset[i]

        for sensor, calib, ts, pcl, scores in zip(sensors, calibs, timestamps, pointclouds, sps_scores):
            frame_dict[sensor] = {
                'calib': calib,
                'timestamp': ts,
                'pointcloud': pcl,
                'stability_scores': scores
            }
        frame_dict['ego_pose'] = global_poses[i]
        frame_dict['ego_local_pose'] = local_poses[i]
        frame_dict['ego_timestamp'] = ego_timestamps[i]
        all_data.append(frame_dict)
    return all_data

def save_sensor_data(data, sequence_name):
    sequence_dir = os.path.join(BASE_DIR, "sequence", sequence_name)
    scans_dir = os.path.join(sequence_dir, "scans")
    poses_dir = os.path.join(sequence_dir, "poses")
    local_poses_dir = os.path.join(sequence_dir, "local_poses")
    labels_dir = os.path.join(sequence_dir, "labels")
    map_transform_dir = os.path.join(sequence_dir, "map_transform")

    os.makedirs(scans_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(local_poses_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    os.makedirs(map_transform_dir, exist_ok=True)

    for idx, sensor_data in enumerate(data):
        combined_pointclouds = []
        combined_sps_scores = []
        for sensor in sensor_data:
            if sensor == 'ego_pose' or sensor == 'ego_timestamp' or sensor == 'ego_local_pose':
                continue

            calib_matrix = sensor_data[sensor]['calib']
            pointclouds = sensor_data[sensor]['pointcloud']
            
            transformed_points = transform_doppler_points(calib_matrix, pointclouds)
            combined_pointclouds.append(transformed_points)
            combined_sps_scores.append(sensor_data[sensor]['stability_scores'])

        combined_pointcloud = np.vstack(combined_pointclouds)
        combined_sps_scores = np.hstack(combined_sps_scores)

        assert(combined_sps_scores.shape[0] == combined_pointcloud.shape[0])
        
        ego_timestamp = str(np.mean(np.array(sensor_data['ego_timestamp'])))
        scan_file = os.path.join(scans_dir, f"{ego_timestamp}.npy")
        pose_file = os.path.join(poses_dir, f"{ego_timestamp}.txt")
        local_pose_file = os.path.join(local_poses_dir, f"{ego_timestamp}.txt")
        label_file = os.path.join(labels_dir, f"{ego_timestamp}.npy")

        np.save(scan_file, combined_pointcloud)
        np.savetxt(pose_file, sensor_data['ego_pose'], delimiter=',')
        np.savetxt(local_pose_file, sensor_data['ego_local_pose'], delimiter=',')
        np.save(label_file, combined_sps_scores)

        map_transform_file = os.path.join(map_transform_dir, "map_transform.txt")
        if not os.path.exists(map_transform_file):
            dummy_transform = np.eye(4)
            np.savetxt(map_transform_file, dummy_transform, delimiter=',')

def process_scene(scene_data):
    scene_name, split = scene_data
    seq = int(scene_name.split("-")[-1])
    
    dataset_sequence = NuScenesMultipleRadarMultiSweeps(
        data_dir=data_dir,
        nusc=nuscenes_exp[split],
        sequence=seq,
        sensors=sensors,
        nsweeps=num_sweeps,
        ref_frame=ref_frame,
        ref_sensor=ref_sensor,
        sps_thresh=0.0,
        return_sps_scores=True,
        sps_labels_dir=sps_maps_dir,
        apply_dpr=apply_dpr,
        filter_points=filter_points,
        ransac_threshold=dpr_thresh,
        reformat_pcl=False
    )
    
    data = extract_data(dataset_sequence)
    save_sensor_data(data, scene_name)
    
    return data

def process_all_scenes(df, num_workers=4):
    scenes_to_process = []
    
    for _, row in df.iterrows():
        ref_scene_name = row['scene_name']
        ref_split = row['split']
        closest_scenes = row['closest_scenes_data']
        
        scenes_to_process.append((ref_scene_name, ref_split))
        for closest_scene, data in closest_scenes.items():
            scenes_to_process.append((closest_scene, data['split']))
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {executor.submit(process_scene, scene_data): scene_data for scene_data in scenes_to_process}
        results = []
        
        for future in tqdm(as_completed(futures), total=len(scenes_to_process)):
            result = future.result()
            results.append(result)

    return results

if __name__ == "__main__":
    all_data = process_all_scenes(sps_df, num_workers=8)