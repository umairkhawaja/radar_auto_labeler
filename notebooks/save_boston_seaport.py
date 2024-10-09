import repackage
repackage.up()

import multiprocessing
from datasets.nuscenes import NuScenesMultipleRadarMultiSweeps
from utils.occupancy import *
from utils.labelling import *
from utils.transforms import *
from utils.postprocessing import *
from utils.ransac_solver import RANSACSolver
from utils.motion_estimation import *
from autolabeler import AutoLabeler
import pandas as pd
import os
import os.path as osp
import numpy as np
import open3d as o3d
from pathlib import Path
from nuscenes import NuScenes
from multiprocessing import cpu_count, set_start_method
import matplotlib
import matplotlib.pyplot as plt
from utils.visualization import map_pointcloud_to_image, render_pointcloud_in_image, plot_maps
from utils.motion_estimation import remove_dynamic_points
matplotlib.use('Agg')
plt.ioff()
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle

# Set the multiprocessing start method
set_start_method("spawn", force=True)

# Adjust this number based on your machine's capabilities
NUM_WORKERS = min(cpu_count(), 4)

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
os.makedirs('tmp_labelled_maps', exist_ok=True)  # Ensure the directory exists
os.makedirs('checkpoints/scene_scans_voxel_maps', exist_ok=True)
os.makedirs('octomap_tmp', exist_ok=True)

# Define a function for processing each row with checkpointing
def process_row(row):
    ref_scene_name = row['scene_name']
    ref_split = row['split']
    # Updated check: A scene is marked fully processed if its map can be found in tmp_labelled_maps dir
    labelled_map_file = f'tmp_labelled_maps/{ref_scene_name}_labelled_map.npy'

    # Check if this scene has already been processed
    if os.path.exists(labelled_map_file):
        print(f"Skipping {ref_scene_name}, already processed.")
        return ref_scene_name  # Return the scene name to indicate completion

    try:
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

        dynamic_scene_maps = filter_maps_icp(dynamic_scene_maps, alignment_thresh=0.5, overlapping_thresh=0.25)

        # Proceed to generate scene scans, voxel maps, and octomaps in parallel
        # Save necessary data for parallel processing
        # We'll use per-scene checkpointing to avoid reprocessing

        # Process scene scans and voxel maps
        dataloader_lengths = {name: len(dl) for name, dl in row_dls.items()}

        args_list = []
        for name in row_dls.keys():
            args = (name, dataloader_lengths[name], global_scene_pointclouds[name], VOXEL_SIZE)
            args_list.append(args)

        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            pool.map(process_scene_scan_and_voxel_map, args_list)

        # Process scene octomaps
        args_list = []
        for name, dl in row_dls.items():
            # Extract poses and point clouds from the dataloader
            poses_scene = dl.global_poses
            num_readings = dl.num_readings
            point_clouds = []
            calibs_scene = []
            for batch in dl:
                point_clouds.append(batch[0])
                calibs_scene.append(batch[1])
            args = (name, poses_scene, point_clouds, calibs_scene, num_readings, OCTOMAP_RESOLUTION, 'octomap_tmp')
            args_list.append(args)

        with multiprocessing.Pool(processes=NUM_WORKERS) as pool:
            pool.map(process_scene_octomap, args_list)

        # Proceed to label the maps for this scene
        # Prepare data for labeling
        scene_maps = static_scene_maps
        scene_pointclouds = global_scene_pointclouds
        scene_poses = poses
        dataloaders = row_dls

        # Load processed voxel maps
        scene_scans = {}
        scene_voxel_maps = {}
        for name in row_dls.keys():
            checkpoint_file = f'checkpoints/scene_scans_voxel_maps/{name}_voxel_map.pkl'
            if os.path.exists(checkpoint_file):
                with open(checkpoint_file, 'rb') as f:
                    data = pickle.load(f)
                    scene_scans[name] = data['scans']
                    scene_voxel_maps[name] = data['voxel_map']
            else:
                print(f"Voxel map for {name} not found. Skipping occupancy priors.")
                scene_voxel_maps[name] = None

        # Load processed octomaps
        scene_octomaps = {}
        for name in row_dls.keys():
            octomap_file = os.path.join('octomap_tmp', f"{name}_octomap.bt")
            if os.path.exists(octomap_file):
                with open(octomap_file, 'rb') as f:
                    scene_octomaps[name] = octomap.OcTree(OCTOMAP_RESOLUTION)
                    scene_octomaps[name].readBinary(f.read())
            else:
                print(f"Octomap file for scene {name} not found.")
                scene_octomaps[name] = None

        # Create the sps_labeler instance
        sps_labeler = AutoLabeler(
            scene_maps=scene_maps,
            ref_map_id=ref_scene_name,
            scene_poses=scene_poses,
            scene_octomaps=scene_octomaps,
            dynamic_priors=scene_voxel_maps,
            use_octomaps=True,
            use_combined_map=USE_COMBINED_MAP,
            search_in_radius=SEARCH_IN_RADIUS,
            radius=RADIUS,
            downsample=True,
            voxel_size=VOXEL_SIZE,
            filter_out_of_bounds=FILTER_OUT_OF_BOUNDS
        )

        sps_labeler.label_maps()

        labelled_map = sps_labeler.labeled_environment_map
        # Save the labelled map to tmp_labelled_maps directory
        np.save(labelled_map_file, labelled_map)

        print(f"Processed and saved labelled map for {ref_scene_name}")
        return ref_scene_name

    except Exception as e:
        print(f"Error processing {ref_scene_name}: {e}")
        return None  # Indicate failure

# Function to process scene scans and voxel maps with checkpointing
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

# Worker function for processing scene octomaps with checkpointing
def process_scene_octomap(args):
    name, poses, point_clouds, calibs, num_readings, OCTOMAP_RESOLUTION, tmp_dir = args
    octomap_file = os.path.join(tmp_dir, f"{name}_octomap.bt")

    # Check if octomap already exists
    if os.path.exists(octomap_file):
        print(f"Skipping octomap for {name}, already processed.")
        return name

    try:
        dl = {
            'name': name,
            'poses': poses,
            'point_clouds': point_clouds,
            'num_readings': num_readings,
            'calibs': calibs
        }
        octomap_result = build_octomap(dl, resolution=OCTOMAP_RESOLUTION)

        # Save the octomap_result to a file
        with open(octomap_file, 'wb') as f:
            f.write(octomap_result.writeBinary())
        print(f"Processed octomap for {name}")
        return name
    except Exception as e:
        print(f"Failed to process octomap for {name}: {e}")
        return None

# Main processing function
def main():
    # Collect already processed scenes
    processed_scenes = set([fname.split('_labelled_map.npy')[0] for fname in os.listdir('tmp_labelled_maps')])

    # Prepare rows to process
    rows_to_process = [row for _, row in sps_df.iterrows() if row['scene_name'] not in processed_scenes]

    print(f"Total scenes to process: {len(rows_to_process)}")

    # Use ProcessPoolExecutor for parallel processing
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

    # After processing, load all labelled maps
    labelled_map_files = [os.path.join('tmp_labelled_maps', f) for f in os.listdir('tmp_labelled_maps')]
    complete_sps_labelled_map = np.vstack([np.load(f) for f in labelled_map_files])

    # Proceed with plotting and saving the final map
    labeled_map = complete_sps_labelled_map
    points = labeled_map[:, :3]
    stable_probs = labeled_map[:, -1]

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