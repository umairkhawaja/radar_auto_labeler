import repackage
repackage.up()

import multiprocessing
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from utils.postprocessing import *
from utils.ransac_solver import RANSACSolver
from utils.motion_estimation import remove_dynamic_points
from autolabeler import AutoLabeler
import pandas as pd
import os
import os.path as osp
import numpy as np
import open3d as o3d
from pathlib import Path
from nuscenes import NuScenes
from multiprocessing import cpu_count
import matplotlib
import matplotlib.pyplot as plt
from utils.visualization import map_pointcloud_to_image, render_pointcloud_in_image, plot_maps
matplotlib.use('Agg')
plt.ioff()
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import octomap  # Ensure that the octomap Python bindings are installed

# Adjust this number based on your machine's capabilities
NUM_WORKERS = min(cpu_count(), 32)

# Paths and configuration
DF_PATH = '../sps_nuscenes_more_matches_df.json'
sps_df = pd.read_json(DF_PATH)

DATA_DIR = "/shared/data/nuScenes/"
versions = {'trainval': 'v1.0-trainval', 'test': 'v1.0-test'}
nuscenes_exp = {
    vname: NuScenes(dataroot=DATA_DIR, version=version, verbose=False)
    for vname, version in versions.items()
}

# Configuration parameters
REF_FRAME = None
REF_SENSOR = None
FILTER_POINTS = False

NUM_SWEEPS = 5
APPLY_DPR = True
DPR_THRESH = 0.15
OCTOMAP_RESOLUTION = 0.15
VOXEL_SIZE = 0.01

ICP_FILTERING = False
SEARCH_IN_RADIUS = True
RADIUS = 1
USE_LIDAR_LABELS = False
USE_OCCUPANCY_PRIORS = True
FILTER_BY_POSES = False
FILTER_BY_RADIUS = False
FILTER_OUT_OF_BOUNDS = False
USE_COMBINED_MAP = True
USE_N_MAP_CORRESPONDENCES = 1

MAP_NAME = 'boston-seaport_sps_df_1correspondence'

SENSORS = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

# Ensure checkpoint directories exist
os.makedirs('tmp_labelled_maps', exist_ok=True)
os.makedirs('checkpoints/scene_scans_voxel_maps', exist_ok=True)
os.makedirs('octomap_tmp', exist_ok=True)
os.makedirs('processed_scene_data', exist_ok=True)  # Directory to save processed scene data


# Modify process_row to only process and save scene data without labeling
def process_row(row):
    ref_scene_name = row['scene_name']
    ref_split = row['split']

    # Check if this scene has already been processed
    scene_data_file = f'processed_scene_data/{ref_scene_name}_data.pkl'
    if os.path.exists(scene_data_file):
        print(f"Skipping {ref_scene_name}, already processed.")
        return ref_scene_name  # Return the scene name to indicate completion

    try:
        closest_scenes = row['closest_scenes_data']

        # Initialize data loaders for the reference scene and its closest scenes
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

        for i, (matched_scene, data) in enumerate(closest_scenes.items()):
            if i+1 > USE_N_MAP_CORRESPONDENCES:
                break

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
        pointclouds = {name: [dl[i][0] for i in range(dl.num_readings)] for name, dl in row_dls.items()}
        calibs = {name: [dl[i][1] for i in range(dl.num_readings)] for name, dl in row_dls.items()}
        poses = {name: dl.global_poses for name, dl in row_dls.items()}

        # Applying DPR
        scene_pointclouds_dpr_masks, global_scene_pointclouds = remove_dynamic_points(
            pointclouds, calibs, poses, SENSORS, filter_sensors=True, dpr_thresh=DPR_THRESH, save_vis=False)

        scene_dpr_masks = {name: np.hstack(masks) for name, masks in scene_pointclouds_dpr_masks.items()}
        global_scene_maps = {name: np.vstack(pcls) for name, pcls in global_scene_pointclouds.items()}

        static_scene_maps = {}
        dynamic_scene_maps = {}

        for name in global_scene_maps:
            indices = scene_dpr_masks[name]
            map_pcl = global_scene_maps[name]
            filtered_map_pcl = map_pcl[indices]
            static_scene_maps[name] = filtered_map_pcl
            dynamic_scene_maps[name] = map_pcl[~indices]

        # Filtering dynamic scene maps using ICP (if implemented)
        dynamic_scene_maps = filter_maps_icp(dynamic_scene_maps, alignment_thresh=0.5, overlapping_thresh=0.25)

        # Process scene scans and voxel maps
        dataloader_lengths = {name: len(dl) for name, dl in row_dls.items()}

        args_list = []
        for name in row_dls.keys():
            args = (name, dataloader_lengths[name], global_scene_pointclouds[name], VOXEL_SIZE)
            args_list.append(args)

        # Process scene scans and voxel maps
        for args in args_list:
            process_scene_scan_and_voxel_map(args)

        # Process scene octomaps
        args_list = []
        for name, dl in row_dls.items():
            # Extract poses and point clouds from the dataloader
            poses_scene = dl.global_poses
            num_readings = dl.num_readings
            point_clouds = []
            calibs_scene = []
            for i in range(len(dl)):
                batch = dl[i]
                point_clouds.append(batch[0])
                calibs_scene.append(batch[1])
            args = (name, poses_scene, point_clouds, calibs_scene, num_readings, OCTOMAP_RESOLUTION, 'octomap_tmp')
            args_list.append(args)

        # Process scene octomaps
        for args in args_list:
            process_scene_octomap(args)

        # Save all necessary data for labeling
        processed_data = {
            'static_scene_maps': static_scene_maps,
            'dynamic_scene_maps': dynamic_scene_maps,
            'global_scene_pointclouds': global_scene_pointclouds,
            'poses': poses,
            'dataloaders': row_dls,
        }

        # Save the processed data to disk
        with open(scene_data_file, 'wb') as f:
            pickle.dump(processed_data, f)

        print(f"Processed and saved data for {ref_scene_name}")
        return ref_scene_name

    except Exception as e:
        print(f"Error processing {ref_scene_name}: {e}")
        return None  # Indicate failure

# Function to process scene scans and voxel maps
def process_scene_scan_and_voxel_map(args):
    name, dataloader_length, scene_pointclouds_name, VOXEL_SIZE = args
    checkpoint_file = f'checkpoints/scene_scans_voxel_maps/{name}_voxel_map.pkl'

    # Check if this scene has already been processed
    if os.path.exists(checkpoint_file):
        print(f"Skipping voxel map for {name}, already processed.")
        return name  # Return the scene name to indicate completion

    try:
        # Build scene_scans for this name
        scans = {i: scene_pointclouds_name[i][:, :3] for i in range(dataloader_length)}
        # Build scene_voxel_map for this name
        voxel_map = create_voxel_map(scans, VOXEL_SIZE)

        # Save to disk
        with open(checkpoint_file, 'wb') as f:
            pickle.dump({'scans': scans, 'voxel_map': voxel_map}, f)

        print(f"Processed voxel map for {name}")
        return name

    except Exception as e:
        print(f"Failed to process voxel map for {name}: {e}")
        return None

# Worker function for processing scene octomaps
def process_scene_octomap(args):
    name, poses, point_clouds, calibs, num_readings, OCTOMAP_RESOLUTION, tmp_dir = args
    octomap_file = os.path.join(tmp_dir, f"{name}_octomap.bt")

    # Check if octomap already exists
    if os.path.exists(octomap_file):
        print(f"Skipping octomap for {name}, already processed.")
        return name

    try:
        # Build the octomap for the scene
        tree = octomap.OcTree(OCTOMAP_RESOLUTION)
        for idx in range(num_readings):
            pcl = point_clouds[idx]
            pose = poses[idx]
            # Transform point cloud to global coordinates
            pcl_global = (pose @ np.hstack((pcl[:, :3], np.ones((pcl.shape[0], 1)))).T).T[:, :3]
            # Insert point cloud into octomap
            tree.insertPointCloud(octomap.Pointcloud(pcl_global.astype(np.float32)), octomap.pose6d(0, 0, 0, 0, 0, 0))
        # Save the octomap to a file
        tree.writeBinary(octomap_file)
        print(f"Processed octomap for {name}")
        return name
    except Exception as e:
        print(f"Failed to process octomap for {name}: {e}")
        return None

def main():
    # Collect already processed scenes
    processed_scenes = set([fname.split('_data.pkl')[0] for fname in os.listdir('processed_scene_data')])

    # Prepare rows to process
    rows_to_process = [row for _, row in sps_df.iterrows() if row['scene_name'] not in processed_scenes]

    print(f"Total scenes to process: {len(rows_to_process)}")

    # Process scenes in parallel
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        future_to_row = {executor.submit(process_row, row): row['scene_name'] for row in rows_to_process}

        for future in as_completed(future_to_row):
            scene_name = future_to_row[future]
            try:
                result = future.result()
                if result:
                    print(f"Completed processing {scene_name}")
                else:
                    print(f"Failed to process {scene_name}")
            except Exception as exc:
                print(f"Exception occurred while processing {scene_name}: {exc}")

    # After processing, collect data from all scenes
    scene_data_files = [os.path.join('processed_scene_data', f) for f in os.listdir('processed_scene_data')]
    all_static_scene_maps = {}
    all_scene_poses = {}
    all_scene_octomaps = {}
    all_scene_voxel_maps = {}

    for data_file in scene_data_files:
        with open(data_file, 'rb') as f:
            processed_data = pickle.load(f)
            scene_name = os.path.basename(data_file).split('_data.pkl')[0]
            all_static_scene_maps.update(processed_data['static_scene_maps'])
            all_scene_poses.update(processed_data['poses'])
            # Load processed voxel maps
            checkpoint_file = f'checkpoints/scene_scans_voxel_maps/{scene_name}_voxel_map.pkl'
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as vf:
                    data = pickle.load(vf)
                    all_scene_voxel_maps[scene_name] = data['voxel_map']
            else:
                print(f"Voxel map for {scene_name} not found.")
                all_scene_voxel_maps[scene_name] = None
            # Load processed octomaps
            octomap_file = os.path.join('octomap_tmp', f"{scene_name}_octomap.bt")
            if os.path.exists(octomap_file):
                tree = octomap.OcTree(OCTOMAP_RESOLUTION)
                tree.readBinary(octomap_file)
                all_scene_octomaps[scene_name] = tree
            else:
                print(f"Octomap for {scene_name} not found.")
                all_scene_octomaps[scene_name] = None

    # Create the sps_labeler instance with all collected data
    sps_labeler = AutoLabeler(
        scene_maps=all_static_scene_maps,
        ref_map_id=None,  # Specify if needed
        scene_poses=all_scene_poses,
        scene_octomaps=all_scene_octomaps,
        dynamic_priors=all_scene_voxel_maps,
        use_octomaps=True,
        use_combined_map=USE_COMBINED_MAP,
        search_in_radius=SEARCH_IN_RADIUS,
        radius=RADIUS,
        downsample=True,
        voxel_size=VOXEL_SIZE,
        filter_out_of_bounds=FILTER_OUT_OF_BOUNDS
    )

    # Call label_maps once with all data
    sps_labeler.label_maps()

    # After labeling, collect the labeled maps
    labeled_maps = sps_labeler.labeled_environment_map

    # Save and plot the final map
    points = labeled_maps[:, :3]
    stable_probs = labeled_maps[:, -1]

    plt.figure(figsize=(10, 10))
    plt.scatter(points[:, 0], points[:, 1], c=stable_probs, cmap='RdYlGn', s=0.05)
    plt.colorbar(label='Stability')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.savefig(f'{MAP_NAME}.jpg', dpi=300)

    filename = f"{MAP_NAME}.asc"
    np.savetxt(
        filename,
        np.hstack([points, stable_probs.reshape(-1, 1)]),
        fmt='%.6f', delimiter=' ',
        header='x y z stable_prob',
        comments=''
    )

if __name__ == "__main__":
    main()