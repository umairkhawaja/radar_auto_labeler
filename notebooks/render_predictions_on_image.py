import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import matplotlib.pyplot as plt
import cv2  # For video creation
from nuscenes import NuScenes
import repackage

# Ensure utils module is accessible
repackage.up()
from utils.visualization import (
    map_pointcloud_to_image,
    render_pointcloud_in_image,
    plot_maps,
)

# Constants and Configurations
DATA_DIR = "/home/umair/workspace/radar_sps_datasets/nuscenes/"
NUSCENES_DATA_DIR = "/shared/data/nuScenes/"
VERSIONS = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
DF_PATH = '../sps_nuscenes_more_matches_df.json'

# Paths for SPS
SPS_BASE_PREDICTION_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes/predictions'
SPS_BASE_OUTPUT_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes_predictions/sps'
SPS_BASE_TENSORBOARD_DIR = '/home/umair/workspace/SPS/tb_logs'
SPS_THRESHOLD = 0.84

# Paths for RIT
RIT_BASE_PREDICTION_DIR = '/home/umair/workspace/rit-master/test_output/'
RIT_BASE_OUTPUT_DIR = '/home/umair/workspace/radar_sps_datasets/nuscenes_predictions/rit'
RIT_BASE_TENSORBOARD_DIR = '/home/umair/workspace/rit-master/runs/'
RIT_THRESHOLD = 0.5

# Experiment Names
SPS_EXP_NAMES = [
    # 'Nuscenes-ICPMaps-NoAugs-NoRadarFeatures',
    'Nuscenes-ICPMaps-NoAugs-RadarFeatures',
    # 'Nuscenes-ICPMaps-Augs-RadarFeatures',
    'Nuscenes-MapsDPR-NoAugs-RadarFeatures',
    'Nuscenes-MapsDPR-NoAugs-RadarFeatures-DPRScans',
]

RIT_EXP_NAMES = [
    # 'tversky-nosubmaps-noaugs',
    # 'tversky-nosubmaps-augs',
    'ce-nosubmaps-noaugs',
    # 'ce-nosubmaps-augs',
    'ce-dprsubmaps-noaugs',
    'ce-dprsubmaps-noaugs-dprscans',
    'tversky-dprsubmaps-noaugs-dprscans',
]

# Test Scenes
TEST_SCENES = [
    'scene-0665', 'scene-0218', 'scene-0219', 'scene-0695', 'scene-0901',
    'scene-0530', 'scene-0570', 'scene-0464', 'scene-0573', 'scene-0515',
    'scene-0292', 'scene-0678', 'scene-0487', 'scene-0472', 'scene-0679',
    'scene-0643', 'scene-0911', 'scene-1033', 'scene-0039', 'scene-0984',
    'scene-0147', 'scene-0359', 'scene-0372', 'scene-0126', 'scene-0968',
    'scene-0484', 'scene-0461', 'scene-0669', 'scene-0588', 'scene-0286',
    'scene-0701', 'scene-0703', 'scene-0533', 'scene-0143', 'scene-0281',
    'scene-0459', 'scene-0585', 'scene-0285', 'scene-0526',
]

# Initialize NuScenes Instances
def initialize_nuscenes(data_dir, versions):
    return {
        vname: NuScenes(dataroot=data_dir, version=vdir, verbose=False)
        for vname, vdir in versions.items()
    }

# Load SPS DataFrame
def load_sps_dataframe(df_path):
    return pd.read_json(df_path)

# Generate Experiments List
def generate_experiments(sps_exp_names, rit_exp_names):
    experiments = []

    # SPS Experiments
    for exp_name in sps_exp_names:
        sps_tb_log_dir = os.path.join(SPS_BASE_TENSORBOARD_DIR, exp_name)
        tensorboard_log_file = find_latest_tensorboard_log(sps_tb_log_dir, subdir='version_')

        experiment = {
            'name': 'sps',
            'exp_name': exp_name,
            'output_dir': os.path.join(SPS_BASE_OUTPUT_DIR, exp_name),
            'prediction_dir': os.path.join(SPS_BASE_PREDICTION_DIR, exp_name),
            'threshold': SPS_THRESHOLD,
            'tensorboard_log_dir': tensorboard_log_file,
            'metric_name_mapping': {
                'loss_train_epoch': 'train_loss_epoch',
                'val_loss_epoch': 'val_loss_epoch',
                'train_loss': 'train_loss_step',
                'val_loss_step': 'val_loss_step',
                'val_r2_epoch': 'val_r2_epoch',
                'val_r2_step': 'val_r2_epoch',
            }
        }
        experiments.append(experiment)

    # RIT Experiments
    for exp_name in rit_exp_names:
        rit_tb_log_dir = RIT_BASE_TENSORBOARD_DIR
        tensorboard_log_file = find_latest_rit_tensorboard_log(rit_tb_log_dir, exp_name)

        experiment = {
            'name': 'rit',
            'exp_name': exp_name,
            'output_dir': os.path.join(RIT_BASE_OUTPUT_DIR, exp_name),
            'prediction_dir': os.path.join(RIT_BASE_PREDICTION_DIR, exp_name),
            'threshold': RIT_THRESHOLD,
            'tensorboard_log_dir': tensorboard_log_file,
            'metric_name_mapping': {
                'loss_train_epoch': 'train_loss_epoch',
                'loss_val_epoch': 'val_loss_epoch',
                'loss_train_batch': 'train_loss_step',
                'loss_val_batch': 'val_loss_step',
                'iou_moving_train_batch': 'train_iou_unstable_step',
                'iou_moving_val_batch': 'val_iou_unstable_step',
                'iou_moving_val_epoch': 'val_iou_unstable_epoch',
                'iou_static_val_epoch': 'val_iou_stable_epoch',
            }
        }
        experiments.append(experiment)

    return experiments

# Helper Function to Find Latest TensorBoard Log for SPS
def find_latest_tensorboard_log(tb_log_dir, subdir='version_'):
    if os.path.exists(tb_log_dir):
        versions = [d for d in os.listdir(tb_log_dir) if d.startswith(subdir)]
        if versions:
            latest_version = sorted(versions, key=lambda x: int(x.split('_')[1]))[-1]
            tensorboard_log_dir = os.path.join(tb_log_dir, latest_version)
            event_files = [f for f in os.listdir(tensorboard_log_dir) if f.startswith('events.out.tfevents')]
            if event_files:
                return os.path.join(tensorboard_log_dir, event_files[0])
    return None

# Helper Function to Find Latest TensorBoard Log for RIT
def find_latest_rit_tensorboard_log(tb_log_dir, exp_name):
    event_files = []
    if os.path.exists(tb_log_dir):
        for root, dirs, files in os.walk(tb_log_dir):
            for file in files:
                if file.startswith('events.out.tfevents') and exp_name in root:
                    event_files.append(os.path.join(root, file))
    if event_files:
        return sorted(event_files)[-1]
    return None

# Parse RIT Predictions
def parse_rit_predictions(dataset_dir, predictions_dir, scene_name):
    src_pose_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'local_poses')
    dst_pose_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'poses')
    scans_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'scans')

    scan_files = sorted(
        [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith('.npy')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.npy')[0])))
    )
    src_pose_paths = sorted(
        [os.path.join(src_pose_dir, f) for f in os.listdir(src_pose_dir) if f.endswith('.txt')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.txt')[0])))
    )
    dst_pose_paths = sorted(
        [os.path.join(dst_pose_dir, f) for f in os.listdir(dst_pose_dir) if f.endswith('.txt')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.txt')[0])))
    )

    src_poses = [np.loadtxt(pth, delimiter=',') for pth in src_pose_paths]
    dst_poses = [np.loadtxt(pth, delimiter=',') for pth in dst_pose_paths]

    all_scans = []
    for i, scan_file in enumerate(scan_files):
        predicted_pcl = np.load(os.path.join(predictions_dir, scene_name, f'{i}.npy'))
        global_predicted_pcl = update_pointcloud_pose(predicted_pcl, src_poses[i], dst_poses[i])
        all_scans.append(global_predicted_pcl)

    output_map = np.vstack(all_scans)
    return output_map, all_scans

# Parse SPS Predictions
def parse_sps_predictions(dataset_dir, predictions_dir, scene_name, update_pose=False):
    src_pose_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'local_poses')
    dst_pose_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'poses')

    scans_dir = os.path.join(dataset_dir, 'sequence', scene_name, 'scans')

    scan_files = sorted(
        [os.path.join(scans_dir, f) for f in os.listdir(scans_dir) if f.endswith('.npy')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.npy')[0])))
    )
    src_pose_paths = sorted(
        [os.path.join(src_pose_dir, f) for f in os.listdir(src_pose_dir) if f.endswith('.txt')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.txt')[0])))
    )
    dst_pose_paths = sorted(
        [os.path.join(dst_pose_dir, f) for f in os.listdir(dst_pose_dir) if f.endswith('.txt')],
        key=lambda f: float(''.join(filter(lambda x: x.isdigit() or x == '.', os.path.basename(f).split('.txt')[0])))
    )

    src_poses = [np.loadtxt(pth, delimiter=',') for pth in src_pose_paths]
    dst_poses = [np.loadtxt(pth, delimiter=',') for pth in dst_pose_paths]

    all_scans = []
    for i, scan_file in enumerate(scan_files):
        predicted_pcl = np.load(os.path.join(predictions_dir, scene_name, 'scans', f'{i}.npy'))[:, [0, 1, 2, 4]]
        if update_pose:
            predicted_pcl = update_pointcloud_pose(predicted_pcl, src_poses[i], dst_poses[i])
        all_scans.append(predicted_pcl)

    output_map = np.vstack(all_scans)
    return output_map, all_scans

# Update Point Cloud Pose
def update_pointcloud_pose(pointcloud, from_pose, to_pose):
    transformation = to_pose @ np.linalg.inv(from_pose)
    num_points = pointcloud.shape[0]
    xyz = pointcloud[:, :3]
    labels = pointcloud[:, 3].reshape(num_points, 1)
    xyz_homogeneous = np.hstack([xyz, np.ones((num_points, 1))]).T
    transformed_xyz = (transformation @ xyz_homogeneous).T[:, :3]
    transformed_point_clouds = np.hstack([transformed_xyz, labels])
    return transformed_point_clouds

# Create Video from Frames
def create_video(frames_dir, video_path, fps=10):
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    if not frame_files:
        print(f"No frames found in {frames_dir}. Skipping video creation.")
        return

    # Read the first frame to get the frame size
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Failed to read the first frame {first_frame_path}. Skipping video creation.")
        return
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Skipping frame {frame_path} as it could not be read.")

    video.release()
    print(f"Video saved to {video_path}")

# Process a Single Scene for a Given Experiment
def process_scene(nusc_exp, experiment, scene_name, sps_df):
    try:
        row = sps_df[sps_df.scene_name == scene_name]
        if not row.empty:
            row = row.iloc[0]
            scene_split = row['split']
            scene_token = row.get('scene_token', '')
        else:
            mask = sps_df.closest_scenes.apply(lambda x: scene_name in x)
            if not mask.any():
                print(f"Scene {scene_name} not found in SPS DataFrame.")
                return
            row = sps_df[mask].iloc[0]
            scene_split = row['closest_scenes_data'][scene_name]['split']
            scene_token = row['closest_scenes_data'][scene_name]['scene_token']

        if not scene_token:
            print(f"No scene_token found for scene {scene_name}. Skipping.")
            return
        nusc = nusc_exp[scene_split]
        scene = nusc.get('scene', scene_token)
        sample_token = scene['first_sample_token']
        sample = nusc.get('sample', sample_token)

        gt_lmap_path = os.path.join('/home/umair/workspace/radar_sps_datasets/nuscenes/maps', f'{scene_name}.asc')
        if not os.path.exists(gt_lmap_path):
            print(f"Ground truth map not found for {scene_name} at {gt_lmap_path}.")
            return
        gt_lmap = np.loadtxt(gt_lmap_path, skiprows=1)

        # Parse Predictions
        if experiment['name'] == 'sps':
            pred_lmap, _ = parse_sps_predictions(
                dataset_dir=DATA_DIR,
                predictions_dir=experiment['prediction_dir'],
                scene_name=scene_name,
                update_pose=True
            )
        elif experiment['name'] == 'rit':
            pred_lmap, _ = parse_rit_predictions(
                dataset_dir=DATA_DIR,
                predictions_dir=experiment['prediction_dir'],
                scene_name=scene_name
            )
        else:
            print(f"Unknown experiment type: {experiment['name']}")
            return

        pred_lmap[:, -1] = 1 - pred_lmap[:, -1]  # Adjust labels to match ground truth

        # Create Plots Directory
        plots_dir = os.path.join(experiment['prediction_dir'], 'plots', scene_name)
        Path(plots_dir).mkdir(parents=True, exist_ok=True)

        frame_index = 0
        viewpoint = [('RADAR_FRONT', 'CAM_FRONT'), ('RADAR_BACK_LEFT', 'CAM_BACK')]
        vidx = 0  # Index for viewpoint

        model_label = f"{experiment['name'].upper()}_{experiment['exp_name']}"

        while sample_token:
            fig, (ax_top, ax_bottom) = plt.subplots(2, 1, figsize=(16, 8), dpi=300)
            plt.subplots_adjust(left=0.05, right=0.95, top=0.90, bottom=0.10)  # Adjusted to accommodate labels

            # Add Labels
            fig.text(0.5, 0.95, f"Scene: {scene_name} | Experiment: {experiment['exp_name']}", ha='center', fontsize=14, fontweight='bold')
            fig.text(0.1, 0.92, "Top is GT", fontsize=12, ha='left')
            fig.text(0.1, 0.05, f"Below is prediction from {model_label}", fontsize=12, ha='left')

            # Render Ground Truth
            render_pointcloud_in_image(
                nusc,
                gt_lmap,
                sample['token'],
                ax=ax_top,
                pointsensor_channel=viewpoint[vidx][0],
                camera_channel=viewpoint[vidx][1]
            )
            ax_top.set_title("Ground Truth", fontsize=12)
            ax_top.axis('off')

            # Render Prediction
            render_pointcloud_in_image(
                nusc,
                pred_lmap,
                sample['token'],
                ax=ax_bottom,
                pointsensor_channel=viewpoint[vidx][0],
                camera_channel=viewpoint[vidx][1]
            )
            ax_bottom.set_title(f"Prediction ({model_label})", fontsize=12)
            ax_bottom.axis('off')

            plt.tight_layout(rect=[0, 0.05, 1, 0.90])  # Adjust layout to prevent overlap with labels

            frame_path = os.path.join(plots_dir, f'{frame_index}.jpg')
            fig.savefig(frame_path, dpi=300)
            plt.close(fig)

            # Move to Next Sample
            sample_token = sample.get('next', '')
            if sample_token:
                sample = nusc.get('sample', sample_token)
            frame_index += 1

        # Create Video from Frames
        video_path = os.path.join(experiment['prediction_dir'], 'videos', scene_name)
        Path(os.path.dirname(video_path)).mkdir(parents=True, exist_ok=True)
        video_path += '.mp4'
        create_video(plots_dir, video_path, fps=10)

        print(f"Processed scene {scene_name} for experiment {experiment['exp_name']}.")

    except Exception as e:
        print(f"Error processing scene {scene_name} for experiment {experiment['exp_name']}: {e}")

# Process a Single Experiment
def process_experiment(nuscenes_exp, experiment, test_scenes, sps_df):
    for scene_name in test_scenes:
        process_scene(nuscenes_exp, experiment, scene_name, sps_df)

# Create Video from Frames
def create_video(frames_dir, video_path, fps=10):
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.endswith('.jpg')],
        key=lambda f: int(os.path.splitext(f)[0])
    )
    if not frame_files:
        print(f"No frames found in {frames_dir}. Skipping video creation.")
        return

    # Read the first frame to get the frame size
    first_frame_path = os.path.join(frames_dir, frame_files[0])
    frame = cv2.imread(first_frame_path)
    if frame is None:
        print(f"Failed to read the first frame {first_frame_path}. Skipping video creation.")
        return
    height, width, layers = frame.shape

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Be sure to use lower case
    video = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        frame = cv2.imread(frame_path)
        if frame is not None:
            video.write(frame)
        else:
            print(f"Warning: Skipping frame {frame_path} as it could not be read.")

    video.release()
    print(f"Video saved to {video_path}")

# Main Function
def main(max_workers=4):
    # Initialize NuScenes
    nuscenes_exp = initialize_nuscenes(NUSCENES_DATA_DIR, VERSIONS)

    # Load SPS DataFrame
    sps_df = load_sps_dataframe(DF_PATH)

    # Generate Experiments
    experiments = generate_experiments(SPS_EXP_NAMES, RIT_EXP_NAMES)

    # Create Output Directories if Needed
    for experiment in experiments:
        Path(experiment['output_dir']).mkdir(parents=True, exist_ok=True)

    # Parallel Processing of Experiments
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_experiment = {
            executor.submit(process_experiment, nuscenes_exp, experiment, TEST_SCENES, sps_df): experiment
            for experiment in experiments
        }
        for future in as_completed(future_to_experiment):
            experiment = future_to_experiment[future]
            try:
                future.result()
            except Exception as exc:
                print(f"Experiment {experiment['exp_name']} generated an exception: {exc}")

if __name__ == "__main__":
    main(max_workers=1)  # Adjust max_workers as needed