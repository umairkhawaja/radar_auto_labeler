import os
import sys
import datetime
import importlib
import os.path as osp
from pathlib import Path
from functools import reduce

import numpy as np
import matplotlib.pyplot as plt
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import RadarPointCloud
from nuscenes.utils.geometry_utils import transform_matrix

from utils.ransac_solver import RANSACSolver
from utils.transforms import *
from utils.labelling import  get_sps_labels
RadarPointCloud.default_filters()

class NuScenesDataloader:
    def __init__(self, nusc, data_dir: Path, sequence: int,  nusc_version: str = "v1.0-mini", *_, **__):
        try:
            importlib.import_module("nuscenes")
        except ModuleNotFoundError:
            print("nuscenes-devkit is not installed on your system")
            print('run "pip install nuscenes-devkit"')
            sys.exit(1)

        self.data_dir = data_dir
        self.sequence_id = str(int(sequence)).zfill(4)
        self.sequence = self.sequence_id

        if nusc is None:
            raise "No NuScenes dataset provided"
        
        self.nusc = nusc
        self.scene_name = f"scene-{self.sequence_id}"
        
        # Get the scene
        self.scene = None
        for s in self.nusc.scene:
           if self.scene_name == s["name"]:
               self.scene = s
        
        if  self.scene is None:
            print(f'[ERROR] Sequence "{self.sequence_id}" not available scenes')
            print("\nAvailable scenes:")
            self.nusc.list_scenes()
            sys.exit(1)

    @staticmethod
    def doppler_v(pts, vx_idx, vy_idx):
        # pts: [x y z dyn_prop id rcs vx vy vx_comp vy_comp ...]
        # Compute the Doppler shift for each point
        x, y, vx, vy = pts[:, 0], pts[:, 1], pts[:, vx_idx], pts[:, vy_idx]
        v_doppler = (vx*x + vy*y) / np.sqrt(x**2 + y**2)

        return v_doppler



class NuScenesMultipleRadarMultiSweeps(NuScenesDataloader):
    def __init__(self, 
                 nusc,
                 data_dir: Path,
                 sequence: int,
                 seq_crop_indices: list = [],
                 nsweeps = 5,
                 sensors: list= [],
                 filter_dynamic_pts=False,
                 sps_labels_dir=None,
                 sps_thresh=0.5,
                 return_sps_scores=False,
                 apply_dpr = False,
                 ransac_threshold = 0.1,
                 ref_sensor='LIDAR_TOP',
                 ref_frame=None,
                 measure_range = 100,
                 nusc_version: str = "v1.0-mini",
                 annotated_keyframes: bool = False, *_, **__):
        """
        Initializer for the NuScenes Radar and LIDAR synchronized dataset

        Args:
            data_dir (Path): Location of the NuScenes dataset
            sequence (int): Sequence ID
            mode (str, optional): Do we want radar or LIDAR data?.  Defaults to "lidar".
            sensor_name (str, optional): Sensor name from the dataset: 
            front_only (bool, optional): Do we want only data from the front radar?
        """
        if len(sensors) == 0:
            raise ValueError("No sensors selected")
        super().__init__(nusc, data_dir, sequence, "radar", nusc_version)

        self.sensors = sensors
        self.ref_sensor = ref_sensor
        self.channels = ["RADAR_FRONT", "RADAR_FRONT_LEFT", "RADAR_FRONT_RIGHT", "RADAR_BACK_LEFT", "RADAR_BACK_RIGHT"]

        self.ref_frame = ref_frame

        self.seq_crop_indices = seq_crop_indices # Keep between of these frames only
        assert(nsweeps > 0), "Number of sweeps should be >= 1"
        self.n_sweeps = nsweeps
        self.measure_range = measure_range

        self.ransac_solver = RANSACSolver(ransac_threshold)
        self.apply_dpr = apply_dpr
        self.filter_dynamic_pts = filter_dynamic_pts
        self.sps_labels_dir = sps_labels_dir
        self.sps_thresh = sps_thresh
        self.return_sps_scores = return_sps_scores
        if self.return_sps_scores:
            # Filtering will take outside the dataloader then
            self.sps_thresh = -1

        self.min_distance = 1.0 # For Multi-sweep reading

        
        if annotated_keyframes:
            self.sensor_readings, self.num_readings = self._get_annotated_sensor_readings(self.sensors)
            self.cam_readings = self._get_annotated_sensor_readings(["CAM_FRONT"])[0]["CAM_FRONT"]
        else:
            self.sensor_readings, self.num_readings = self._get_sensor_readings()
        
        self.gt_poses = self._load_poses()
        self.local_poses = clear_z(self._load_poses(global_poses=False))
        self.global_poses = clear_z(self._load_poses(global_poses=True))
        self.timestamps = self._get_timestamps()[:len(self.gt_poses)]
        
        self.frame_data, self.valid_sample_tokens, self.valid_frame_indices  = self._read_frames_multisweeps()
        
        self.gt_poses = [p for i,p in enumerate(self.gt_poses) if i in self.valid_frame_indices]
        self.local_poses = [p for i,p in enumerate(self.local_poses) if i in self.valid_frame_indices]
        self.global_poses = [p for i,p in enumerate(self.global_poses) if i in self.valid_frame_indices]
        self.timestamps = [p for i,p in enumerate(self.timestamps) if i in self.valid_frame_indices]
        self.gt_poses_global = np.array(self.global_poses)
        self.num_readings = len(self.valid_frame_indices)


        self.image_width = 1600
        self.image_height = 9000
        self.h_fov_deg = 70
        self.v_fov_deg = 44
        
        
        
    def __len__(self):
        return self.num_readings


    def __getitem__(self, idx):
        if self.ref_frame == self.ref_sensor and self.ref_frame is not None:
            sensors = [self.ref_sensor]
        else:
            sensors = self.sensors
        return self.read_point_clouds(idx), self.read_calibs(idx), sensors, self.read_timestamps(idx)
    
    
    def read_point_clouds(self, idx):
        pointclouds = []
        
        frame_dict = self.frame_data[idx]
        pcd = frame_dict['points']
        sensor_ids = frame_dict['sensor_ids']
        calibs = self.read_calibs(idx)
        global_from_car = self.global_poses[idx]

        parsed_points = None
        parsed_sps_scores = []

        if self.ref_frame == self.ref_sensor and self.ref_frame is not None:
            # Just return the reference sensor pcd
            points = pcd[self.ref_sensor]
            pointcloud_reformatted = np.zeros((len(points), 6))
            pointcloud_reformatted[:, :3] = points[:, :3] # x y z
            pointcloud_reformatted[:, 3] = points[:, 3] # rcs
            pointcloud_reformatted[:, 4]   = self.doppler_v(points, 5, 6) # doppler_shift
            pointcloud_reformatted[:, 5]   = self.doppler_v(points, 7, 8) # compensated velocities doppler shift
            parsed_points = [pointcloud_reformatted.astype(np.float64)]
        elif self.ref_frame is None:
            # Output each pointcloud in its own sensor frame
            for sensor in self.sensors:
                points = pcd[sensor]
                pointcloud_reformatted = np.zeros((len(points), 6))
                pointcloud_reformatted[:, :3] = points[:, :3] # x y z
                pointcloud_reformatted[:, 3] = points[:, 3] # rcs
                pointcloud_reformatted[:, 4]   = self.doppler_v(points, 5, 6) # doppler_shift
                pointcloud_reformatted[:, 5]   = self.doppler_v(points, 7, 8) # compensated velocities doppler shift
                pointclouds.append(pointcloud_reformatted.astype(np.float64))
            parsed_points = pointclouds
        
        elif self.ref_frame == 'ego':
            # Transform pointcloud to the ego vehicle frame for each sensor and stack them
            for sensor, car_from_sensor in zip(self.sensors, calibs):
                points = pcd[sensor]
                pointcloud_reformatted = np.zeros((len(points), 6))
                pointcloud_reformatted[:, :3] = points[:, :3] # x y z
                pointcloud_reformatted[:, 3] = points[:, 3] # rcs
                pointcloud_reformatted[:, 4]   = self.doppler_v(points, 5, 6) # doppler_shift
                pointcloud_reformatted[:, 5]   = self.doppler_v(points, 7, 8) # compensated velocities doppler shift
                ego_points = transform_doppler_points(car_from_sensor, pointcloud_reformatted.astype(np.float64))
                pointclouds.append(ego_points)
            parsed_points = np.vstack(pointclouds)
        elif self.ref_frame == 'global':
            for sensor, car_from_sensor in zip(self.sensors, calibs):
                points = pcd[sensor]
                pointcloud_reformatted = np.zeros((len(points), 6))
                pointcloud_reformatted[:, :3] = points[:, :3] # x y z
                pointcloud_reformatted[:, 3] = points[:, 3] # rcs
                pointcloud_reformatted[:, 4]   = self.doppler_v(points, 5, 6) # doppler_shift
                pointcloud_reformatted[:, 5]   = self.doppler_v(points, 7, 8) # compensated velocities doppler shift
                ego_points = transform_doppler_points(car_from_sensor, pointcloud_reformatted.astype(np.float64))
                global_points = transform_doppler_points(global_from_car, ego_points)
                pointclouds.append(global_points)
            parsed_points = np.vstack(pointclouds)

        if self.sps_labels_dir is not None:
            # Label and filter points
            labelled_map_path = os.path.join(self.sps_labels_dir, f'scene-{self.sequence_id}.asc')
            lmap = np.loadtxt(labelled_map_path, delimiter=' ', skiprows=1)


            if self.ref_frame is None:
                # Need to extract labels for each sensor's points 
                filtered_pcds = []
                for sensor in self.sensors:
                    s_idx = self.sensors.index(sensor)
                    sensor_pcd = parsed_points[s_idx]
                    ego_sensor_pcd = transform_doppler_points(calibs[s_idx], sensor_pcd)
                    global_sensor_pcd = transform_doppler_points(global_from_car, ego_sensor_pcd)
                    sensor_pcd_labels = get_sps_labels(lmap, global_sensor_pcd)
                    parsed_sps_scores.append(sensor_pcd_labels)
                    sps_filtered_pcd = sensor_pcd[sensor_pcd_labels >= self.sps_thresh]
                    # print(f"Before: {len(sensor_pcd)} | After {len(sps_filtered_pcd)}")
                    if len(sps_filtered_pcd) == 0:
                        filtered_pcds.append(sensor_pcd.astype(np.float64))
                    else:
                        filtered_pcds.append(sps_filtered_pcd.astype(np.float64))
                parsed_points = filtered_pcds

            elif self.ref_frame in self.sensors:
                s_idx = self.sensors.index(self.ref_frame)
                sps_filtered_pcd = parsed_points[s_idx]
                ego_sensor_pcd = transform_doppler_points(calibs[s_idx], sps_filtered_pcd)
                global_sensor_pcd = transform_doppler_points(global_from_car, ego_sensor_pcd)
                sps_labels = get_sps_labels(lmap, global_sensor_pcd)
                sps_filtered_pcd = sps_filtered_pcd[sps_labels >= self.sps_thresh]
                parsed_points = [sps_filtered_pcd.astype(np.float64)]
                parsed_sps_scores = sps_labels

            elif self.ref_frame == 'ego':
                sps_filtered_pcd = parsed_points
                global_sensor_pcd = transform_doppler_points(global_from_car, sps_filtered_pcd)
                sps_labels = get_sps_labels(lmap, global_sensor_pcd)
                sps_filtered_pcd = sps_filtered_pcd[sps_labels >= self.sps_thresh]
                parsed_points = sps_filtered_pcd.astype(np.float64)
                parsed_sps_scores = sps_labels
            
            elif self.ref_frame == 'global':
                sps_labels = get_sps_labels(lmap, parsed_points)
                sps_filtered_pcd = sps_filtered_pcd[sps_labels >= self.sps_thresh]
                parsed_points = sps_filtered_pcd.astype(np.float64)
                parsed_sps_scores = sps_labels

        if self.return_sps_scores:
            return parsed_points, parsed_sps_scores
        else:
            return parsed_points

    @staticmethod
    def filter_static_reliable_points(pointcloud):
        """
        Filters radar pointcloud data to keep the most reliable static points.
        
        :param pointcloud: <np.float: d, n>. Point cloud matrix with d dimensions and n points.
        :return: Filtered point cloud matrix.
        """
        
        # Define the states to keep
        invalid_states_to_keep = [0,4,8,9,10,11,12,15,16,17]
        dynprop_states_to_keep = [1, 3, 5, 7]
        ambig_states_to_keep = [3]
        
        # Extract relevant fields from the point cloud data
        dyn_prop = pointcloud[3, :]
        invalid_state = pointcloud[15, :]
        ambig_state = pointcloud[11, :]

        # Create a boolean mask for each condition
        mask_dynprop = np.isin(dyn_prop, dynprop_states_to_keep)
        mask_invalid = np.isin(invalid_state, invalid_states_to_keep)
        mask_ambig = np.isin(ambig_state, ambig_states_to_keep)
        
        # Combine masks to filter points that satisfy all conditions
        combined_mask = mask_dynprop & mask_invalid & mask_ambig
        # Filter the point cloud
        filtered_pointcloud = pointcloud[:, combined_mask]
        
        return filtered_pointcloud
    
    def _read_frames_multisweeps(self):
        """
        Reads radar data and processes it into frames.

        Args:
            merge_into_ref_sensor (bool): If True, merge data from all sensors into ref_sensor coordinate frame.
                                        If False, keep point clouds from each sensor separate.

        Returns:
            List of frames.
        """
        def process_sweeps(start_index, n_sweeps):
            current_frame = {}
            sensor_ids = []
            pcd_dict = {sensor: np.zeros((0, 10)) for sensor in self.channels}  # Dictionary to store point clouds for each sensor

            for sensor in self.channels:
                points = np.zeros((RadarPointCloud.nbr_dims(), 0))
                all_pc = RadarPointCloud(points)
                all_times = np.zeros((1, 0))

                if self.ref_sensor is None:
                    # Use the sensor itself
                    ref_chan = sensor
                else:
                    ref_chan = self.ref_sensor
                    
                ref_sample_data = self.sensor_readings[ref_chan][start_index]
                ref_time = 1e-6 * ref_sample_data['timestamp']
                ref_pose_rec = self.nusc.get('ego_pose', ref_sample_data['ego_pose_token'])
                ref_cs_rec = self.nusc.get('calibrated_sensor', ref_sample_data['calibrated_sensor_token'])
                ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)
                car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']), inverse=True)


                current_sample_data = self.sensor_readings[sensor][start_index]
                ext_nbr_points = []

                # print(f"Processing index: {start_index} | Sweep indices: ", end="")
                for j in range(n_sweeps):
                    current_index = start_index - j
                    if current_index < 0:
                        break

                    # print(current_index, end=", ")

                    # invalid_states_to_keep = [0x00, 0x04, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0f, 0x10, 0x11]
                    invalid_states_to_keep = [0,4,8,9,10,11,12,15,16,17]
                    dynprop_states_to_keep = [1, 3, 5, 7]
                    ambig_states_to_keep = [3]
                    if self.filter_dynamic_pts:
                        current_pc = RadarPointCloud.from_file(osp.join(self.nusc.dataroot,
                                                                        current_sample_data['filename']),
                                                                        invalid_states=invalid_states_to_keep,
                                                                        dynprop_states=dynprop_states_to_keep, ambig_states=ambig_states_to_keep)
                    else:
                        current_pc = RadarPointCloud.from_file(osp.join(self.nusc.dataroot,
                                                                        current_sample_data['filename']),
                                                                        invalid_states=range(18),
                                                                        dynprop_states=range(8), ambig_states=range(5))
                    current_pc.remove_close(self.min_distance)
                    

                    # Transform to reference channel
                    if self.ref_sensor is not None and self.ref_frame in self.sensors:
                        current_pose_rec = self.nusc.get('ego_pose', current_sample_data['ego_pose_token'])
                        global_from_car = transform_matrix(current_pose_rec['translation'], Quaternion(current_pose_rec['rotation']), inverse=False)
                        current_cs_rec = self.nusc.get('calibrated_sensor', current_sample_data['calibrated_sensor_token'])
                        car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']), inverse=False)
                        trans_matrix = reduce(np.dot, [ref_from_car, car_from_global, global_from_car, car_from_current])
                        current_pc.transform(trans_matrix)

                    time_lag = ref_time - 1e-6 * current_sample_data['timestamp']
                    times = time_lag * np.ones((1, current_pc.nbr_points()))
                    all_times = np.hstack((all_times, times))
                    all_pc.points = np.hstack((all_pc.points, current_pc.points))
                    ext_nbr_points.append(current_pc.points.shape[1])

                    if current_index > 0:
                        current_sample_data = self.sensor_readings[sensor][current_index]
                # print()
                radar_pc = all_pc
                times = all_times
                nbr_points = np.array(ext_nbr_points)

                ## Filter points
                # radar_pc = self.filter_static_reliable_points(radar_pc.points)
                radar_pc = radar_pc.points
                radar_points = np.zeros((9, radar_pc.shape[1]))
                
                radar_points[:3, :] = radar_pc[:3, :]
                radar_points[3, :] = radar_pc[5, :]
                radar_points[4, :] = radar_pc[4, :]
                radar_points[5:7, :] = radar_pc[6:8, :]
                radar_points[7:9, :] = radar_pc[8:10, :]
                radar_points = radar_points.T
                radar_points = np.hstack((radar_points, times.transpose()))

                if self.apply_dpr and radar_points.shape[0] > 1 and (sensor not in ['RADAR_FRONT_LEFT', 'RADAR_FRONT_RIGHT']):
                    nbr_flag = np.cumsum(nbr_points)
                    pcl_list = np.split(radar_points, nbr_flag, axis=0)
                    pcls_new = np.zeros((0, 10))
                    for index, pcl in enumerate(pcl_list[:-1]):
                        if pcl.shape[0] > 1:
                            info = [
                                [self.sequence_id, current_index],
                                sensor,
                                nbr_points[0]
                            ]
                            best_mask, _, _ = self.ransac_solver.ransac_nusc(pcl, vis=False, info=info)
                            if best_mask is not None:
                                pcl = pcl[best_mask]
                        pcls_new = np.vstack((pcls_new, pcl))
                    radar_points = pcls_new

                pcd_dict[sensor] = np.concatenate((pcd_dict[sensor], radar_points), axis=0)
                sensor_ids.extend([sensor] * len(radar_points))

            pose = np.empty((4, 4), dtype=np.float32)
            pose_record = self.nusc.get("ego_pose", ref_sample_data["ego_pose_token"])
            pose[:, :] = transform_matrix(
                pose_record["translation"],
                Quaternion(pose_record["rotation"]),
            )

            current_frame = {
                'points': pcd_dict,
                'gt_pose': pose,
                'position': pose[:2, 3].reshape(1, -1),
                'timestamp': ref_sample_data['timestamp'],
                'sensor_ids': np.array(sensor_ids),
            }

            frames.append(current_frame)
            valid_indices.append(start_index)

        frames = []
        valid_indices = []
        valid_tokens = []

        # Process full sweeps
        for i in range(self.n_sweeps - 1, self.num_readings, self.n_sweeps):
            process_sweeps(i, self.n_sweeps)

        # Process remaining sweeps
        remaining = self.num_readings % self.n_sweeps
        if remaining > 0:
            process_sweeps(self.num_readings - 1, remaining)

        sorted_frames = sorted(frames, key=lambda t: t['timestamp'])
        frames = list(sorted_frames)
        for frame in frames:
            frame['day'] = (frame['timestamp'] - sorted_frames[0]['timestamp']) / (1e6 * 3600 * 24)
        return frames, valid_tokens, valid_indices


    def read_calibs(self, idx):
        if self.ref_frame == self.ref_sensor and self.ref_frame is not None:
            calib = self.nusc.get('calibrated_sensor', self.sensor_readings[self.ref_sensor][idx]['calibrated_sensor_token'])
            sensor_to_car = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=False)
            return [sensor_to_car]
        else:
            calibs = []
            for sensor in self.sensors:
                calib = self.nusc.get('calibrated_sensor', self.sensor_readings[sensor][idx]['calibrated_sensor_token'])
                sensor_to_car = transform_matrix(calib['translation'], Quaternion(calib['rotation']), inverse=False)
                calibs.append(sensor_to_car)
            
            return calibs
    

    def read_timestamps(self, idx):
        timestamp = np.array(self.timestamps[idx])
        return timestamp
    
    
    def _load_poses(self, global_poses=False) -> np.ndarray:
        poses = []
        sensors = self.sensors
       
        for idx in range(self.num_readings):
            current_pose_dict = {}
            for sensor in sensors:
                current_pose_dict[sensor] = {}
                current_sensor_reading = self.sensor_readings[sensor][idx]

                current_pose_reading = self.nusc.get("ego_pose", current_sensor_reading["ego_pose_token"])
                pose_matrix = transform_matrix(
                    current_pose_reading["translation"],
                    Quaternion(current_pose_reading["rotation"]),
                )
                current_pose_dict[sensor]['pose'] = pose_matrix
                current_pose_dict[sensor]['timestamp'] = current_pose_reading['timestamp']
                # print(current_pose_reading['timestamp'])
                # print(pose_matrix)
            
            sorted_result = sorted(list(current_pose_dict.items()), key=lambda t:t[1]['timestamp'])
            current_pose_dict = {}
            current_pose_dict.update(sorted_result)
            latest_pose = current_pose_dict[list(current_pose_dict.keys())[-1]]['pose']
            poses.append(np.expand_dims(latest_pose, axis=0))
        
        poses = np.concatenate(poses, axis=0)
        # Convert from global coordinate poses to local poses
        
        first_pose = poses[0, :, :]
        
        if not global_poses:
            poses = np.linalg.inv(first_pose) @ poses

        return poses
    
    def _to_seconds(self, t):
        return datetime.datetime.fromtimestamp(t * 1e-6).timestamp()
    
    def _get_timestamps(self):
        timestamps = {}
        for sensor in self.sensors:
            timestamps[sensor] = [d['timestamp'] for d in self.sensor_readings[sensor]]    

        timestamps_combined = []
        for i in range(self.num_readings):
            timestamps_combined.append([self._to_seconds(timestamps[sensor][i]) for sensor in self.sensors])

        return timestamps_combined
    
    
    def _get_sensor_readings(self) -> dict:
        
        # Get first annotated sample, then iterate starting from it
        current_sample_token = self.scene['first_sample_token']
        
        current_sample = self.nusc.get('sample', current_sample_token)
        if self.ref_sensor not in self.channels and self.ref_sensor is not None:
            channels = self.channels + [self.ref_sensor]
        else:
            channels = self.channels

        sensor_readings = {}
        for sensor in channels:
            sensor_readings[sensor] = []
            
        # Combine all sensor readings into a single list and sort using timestamps
        all_combined = []

        for sensor in channels:
            sensor_data_token = current_sample['data'][sensor]
            sensor_data = self.nusc.get('sample_data', sensor_data_token)

            while sensor_data['next'] != '':
                all_combined.append(sensor_data)
                sensor_data_token = sensor_data['next']
                sensor_data = self.nusc.get('sample_data', sensor_data_token)

        all_combined_sorted = sorted(all_combined, key=lambda d: d['timestamp'])

        # Extract blocks of sensor readings from the sorted list
        current_sensor_readings = {}
        
        for sensor in channels:
            current_sensor_readings[sensor] = []
            
        sample_tokens = []
        for sensor_reading in all_combined_sorted:
            # If it doesn't exist. Store sensor readings to current block
            if not current_sensor_readings[sensor_reading['channel']]:
                current_sensor_readings[sensor_reading['channel']] = sensor_reading
                sample_tokens.append(sensor_reading['sample_token'])

            # If we have a reading from all sensors
            if all(list(current_sensor_readings.values())):
                for key in current_sensor_readings.keys():
                    # Double check the readings are associated to the same sample token
                    if sample_tokens.count(sample_tokens[0]) == len(sample_tokens):
                        # Store all the readings for this block
                        sensor_readings[key].append(current_sensor_readings[key])
                    # Reset to read a new block
                    current_sensor_readings[key] = {}

                sample_tokens = []
             
        num_readings = len(sensor_readings[list(sensor_readings.keys())[-1]])
        
        return sensor_readings, num_readings, 

    def _get_annotated_sensor_readings(self, sensors) -> dict:
        sensor_readings = {}
        for sensor in sensors:
            first_sample_token = self.scene['first_sample_token']
            sample = self.nusc.get('sample', first_sample_token)
            sensor_readings[sensor] = []
            
            while sample["next"] != "":
                sensor_data_token = sample["data"][sensor]

                sensor_data = self.nusc.get('sample_data', sensor_data_token)
                sensor_readings[sensor].append(sensor_data)
                sample = self.nusc.get('sample', sample['next'])
                

        num_readings = len(sensor_readings[list(sensor_readings.keys())[-1]])

        return sensor_readings, num_readings
            

    def plot_ego_trajectory_with_radar(self, idx):
        """
        Plots the ego trajectory with radar point clouds for a given index.
        
        Args:
        idx (int): The index for which to plot the data.
        """
        # Extract ego positions
        ego_positions = np.array([pose[:3, 3] for pose in self.global_poses])
        # Calculate time elapsed in seconds from the first timestamp
        first_timestamp = self.timestamps[0][0]
        time_elapsed = [(t[0] - first_timestamp) for t in self.timestamps]
        # timestamps = [t[0] for t in self.timestamps]

        # Create the plot
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))

        # Plot the ego trajectory
        axes[0].plot(ego_positions[:, 0], ego_positions[:, 1], marker='o', linestyle='-', color='b')
        axes[0].plot(ego_positions[idx, 0], ego_positions[idx, 1], marker='o', color='r', markersize=10)
        axes[0].set_xlabel('X Position')
        axes[0].set_ylabel('Y Position')
        axes[0].set_title('Ego Positions (X-Y Plane)')
        axes[0].grid(True)

        # Plot the radar point clouds
        pointclouds, _, _, timestamp = self[idx]
        all_points = np.concatenate(pointclouds, axis=0)
        scatter = axes[1].scatter(all_points[:, 0], all_points[:, 1], s=1)
        axes[1].set_xlabel('X Position')
        axes[1].set_ylabel('Y Position')
        axes[1].set_title('Radar Point Clouds')
        axes[1].grid(True)

        # Add timestamp text
        elapsed_time = time_elapsed[idx]
        timestamp_text = axes[1].text(0.05, 0.95, f'Time Elapsed: {elapsed_time:.2f} seconds', transform=axes[1].transAxes, verticalalignment='top')

        plt.show()
