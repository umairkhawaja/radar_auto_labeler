import pickle
import os.path as osp
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import shutil
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
import repackage
repackage.up()
from utils.labelling import get_sps_labels
from utils.transforms import *
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps

# Load scene data
sps_df = pd.read_json('nuscenes_scenes_df.json')
sps_df.head()

# Define constants and settings
ref_frame = None
num_sweeps = 5
ref_sensor = None
apply_dpr = True
filter_points = False
dpr_thresh = 0.75

exp_name = "output_sw5-dpr0.0-r1"
base_dir = "/home/umair/workspace/radar_sps_datasets/nuscenes"

data_dir = "/shared/data/nuScenes/"
sps_maps_dir = f"/home/umair/workspace/radar_auto_labeler/{exp_name}/labelled_maps/"
src_maps_dir = sps_maps_dir

sensors = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {
    vname: NuScenes(dataroot=data_dir, version=version, verbose=False)
    for vname, version in versions.items()
}


def combine_maps(sps_maps_dir):
    """
    Combines all .asc maps from the given directory into a single map and writes it to disk.
    
    Parameters:
    - sps_maps_dir: The directory containing the .asc map files.
    """
    combined_data = []

    # Iterate through all .asc files in the directory
    for file_name in os.listdir(sps_maps_dir):
        if file_name.endswith(".asc"):
            file_path = os.path.join(sps_maps_dir, file_name)
            # Read the .asc file
            data = np.loadtxt(file_path, skiprows=1)
            combined_data.append(data)
    
    # Combine all the data into a single array
    combined_data = np.vstack(combined_data)
    
    # Determine the output file path one level up from sps_maps_dir
    parent_dir = os.path.abspath(os.path.join(sps_maps_dir, os.pardir))
    output_file = os.path.join(parent_dir, "combined_map.asc")
    
    # Write the combined data to a new .asc file
    np.savetxt(output_file, combined_data, fmt="%.6f")
    print(f"Combined map saved to {output_file}")
    return combined_data


# Function to extract data from dataset
def extract_data(dataset):
    num_frames = len(dataset)
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
        frame_dict['ego_timestamp'] = ego_timestamps[i]
        all_data.append(frame_dict)
    return all_data

# Function to plot point clouds
def plot_pointclouds(sensor_data, ego_pose):
    all_points = []
    all_scores = []

    for sensor, data in sensor_data.items():
        if sensor == 'ego_pose':
            continue
        calib_matrix = data['calib']
        pointcloud = data['pointcloud']
        stability_scores = data['stability_scores']

        ego_frame_points = transform_doppler_points(calib_matrix, pointcloud)
        global_frame_points = transform_doppler_points(ego_pose, ego_frame_points)

        all_points.append(global_frame_points)
        all_scores.append(stability_scores)

    all_points = np.vstack(all_points)
    all_scores = np.concatenate(all_scores)

    fig = plt.figure(figsize=(16, 16))
    ax = fig.add_subplot(111)
    sc = ax.scatter(all_points[:, 0], all_points[:, 1], c=all_scores, cmap='RdYlGn', s=0.5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.colorbar(sc)
    plt.title('Pointclouds from All Sensors in Global Frame')
    plt.show()

# Function to concatenate sensor data
def concatenate_sensor_data(sensor_data_array):
    concatenated_data = {}

    for sensor in sensor_data_array[0]:
        if sensor == 'ego_pose':
            concatenated_data[sensor] = sensor_data_array[0][sensor]
        else:
            concatenated_data[sensor] = {
                'calib': sensor_data_array[0][sensor]['calib'],
                'timestamp': [],
                'pointcloud': [],
                'stability_scores': []
            }

    for data in sensor_data_array:
        for sensor, values in data.items():
            if sensor == 'ego_pose':
                continue
            concatenated_data[sensor]['timestamp'].append(values['timestamp'])
            concatenated_data[sensor]['pointcloud'].append(values['pointcloud'])
            concatenated_data[sensor]['stability_scores'].append(values['stability_scores'])

    for sensor in concatenated_data:
        if sensor == 'ego_pose':
            continue
        concatenated_data[sensor]['timestamp'] = np.array(concatenated_data[sensor]['timestamp'])
        concatenated_data[sensor]['pointcloud'] = np.vstack(concatenated_data[sensor]['pointcloud'])
        concatenated_data[sensor]['stability_scores'] = np.hstack(concatenated_data[sensor]['stability_scores'])

    return concatenated_data

# Function to save sensor data to disk
def save_sensor_data(data, sequence_name, base_dir):
    sequence_dir = os.path.join(base_dir, "sequence", sequence_name)
    scans_dir = os.path.join(sequence_dir, "scans")
    poses_dir = os.path.join(sequence_dir, "poses")

    os.makedirs(scans_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)

    for idx, sensor_data in enumerate(data):
        combined_pointclouds = []
        for sensor in sensor_data:
            if sensor == 'ego_pose' or sensor == 'ego_timestamp':
                continue

            calib_matrix = sensor_data[sensor]['calib']
            pointclouds = sensor_data[sensor]['pointcloud']

            transformed_points = transform_doppler_points(calib_matrix, pointclouds)
            combined_pointclouds.append(transformed_points)

        combined_pointcloud = np.vstack(combined_pointclouds)

        ego_timestamp = str(np.mean(np.array(sensor_data['ego_timestamp'])))
        scan_file = os.path.join(scans_dir, f"{ego_timestamp}.npy")
        pose_file = os.path.join(poses_dir, f"{ego_timestamp}.txt")

        np.save(scan_file, combined_pointcloud)
        np.savetxt(pose_file, sensor_data['ego_pose'], delimiter=',')

        map_transform_file = os.path.join(sequence_dir, "map_transform.txt")
        if not os.path.exists(map_transform_file):
            dummy_transform = np.eye(4)
            np.savetxt(map_transform_file, dummy_transform, delimiter=',')

# Function to process each scene
def process_scene(row):
    ref_scene_name = row['scene_name']
    ref_split = row['split']
    seq = int(ref_scene_name.split("-")[-1])

    dataset_sequence = NuScenesMultipleRadarMultiSweeps(
        data_dir=data_dir,
        nusc=nuscenes_exp[ref_split],
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
        ransac_threshold=dpr_thresh
    )

    data = extract_data(dataset_sequence)
    save_sensor_data(data, ref_scene_name, base_dir)

# Function to create train, val, test split and write to disk
def create_splits(sps_df, base_dir):
    train, temp = train_test_split(sps_df, test_size=0.3, random_state=42)
    val, test = train_test_split(temp, test_size=(1/3), random_state=42)

    def write_split(split, filename):
        with open(os.path.join(base_dir, filename), 'w') as f:
            for scene_name in split['scene_name']:
                f.write(f"{scene_name}\n")

    write_split(train, 'train.txt')
    write_split(val, 'val.txt')
    write_split(test, 'test.txt')

# Main function to handle parallel processing
def main():
    dst_maps_dir = os.path.join(base_dir, "maps")
    full_map_path = os.path.abspath(os.path.join(sps_maps_dir, os.pardir))
    src_full_map_path = os.path.join(full_map_path, 'combined_map.asc')
    combine_maps(sps_maps_dir)
    dst_full_map_path = os.path.join(base_dir, 'base_map.asc')
    shutil.copytree(src_maps_dir, dst_maps_dir)
    shutil.copyfile(src_full_map_path, dst_full_map_path)

    create_splits(sps_df, base_dir)

    with Pool() as pool:
        pool.map(process_scene, [row for _, row in sps_df.iterrows()])

# Execute main function
if __name__ == "__main__":
    main()