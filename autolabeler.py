from tqdm import tqdm
import octomap
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.transforms import *

class AutoLabeler:
    def __init__(self,
                 scene_maps, 
                 ref_map_id, 
                 scene_octomaps,
                 scene_poses, 
                 lidar_labels=None, 
                 dynamic_priors=None, 
                 use_octomaps=True, 
                 search_in_radius=False,
                 downsample=False, 
                 radius=1.5,
                 fallback_map_radius=100,
                 filter_out_of_bounds=False,
                 use_combined_map=False,
                 voxel_size=0.5):
        
        self.use_octomaps = use_octomaps
        self.maps = scene_maps
        self.octomaps = scene_octomaps
        self.lidar_labels = lidar_labels
        self.dynamic_priors = dynamic_priors
        self.search_in_radius = search_in_radius
        self.radius = radius
        self.fallback_map_radius = fallback_map_radius
        self.scene_poses = scene_poses
        self.occluded_point = -1
        self.filter_out_of_bounds = filter_out_of_bounds
        self.use_combined_map = use_combined_map

        if downsample:
            self.maps = {}
            for name,map in scene_maps.items():
                # Create an Open3D point cloud object
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(map[:, :3])
                
                # Downsample the point cloud
                ref_map_sampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
                map_points = np.asarray(ref_map_sampled.points)
                self.maps[name] = map_points

        self.ref_map_id = ref_map_id
        self.maps_ids = list(self.maps.keys())
        self.labelled_maps = {}

        if self.lidar_labels is not None:
            self.lidar_labeled_scene_maps = {}
            for name in self.maps:
                labels = self.lidar_labels[name]
                radar_map = self.maps[name]
                new_map = transfer_labels(labels, radar_map[:, :3])
                self.lidar_labeled_scene_maps[name] = new_map
        
        self.map_centers = {name: np.mean(map[:,:3], axis=0) for name,map in self.maps.items()}
        self.map_bounds_radius = self.calculate_map_bounds_radius(buffer=25)

    def adjust_stability_scores(self, labeled_map, map_id):
        center = self.map_centers[map_id]
        radius = self.map_bounds_radius
        for i, point in enumerate(labeled_map):
            distance = np.linalg.norm(point[:3] - center)
            if distance > radius:
                labeled_map[i, -1] = 0.5 + 0.5 * (labeled_map[i, -1] - 0.5)  # Move score closer to 0.5
        return labeled_map

    def calculate_map_bounds_radius(self, buffer=0):
        *poses, = self.scene_poses.values()
        all_poses = []
        [all_poses.extend(p) for p in poses]
        all_poses = np.stack(all_poses)[:,:3,3]
        
        center = np.mean(all_poses, axis=0)
        max_distance = np.max(np.linalg.norm(all_poses - center, axis=1)) + buffer
        if max_distance < self.fallback_map_radius / 2:  # Example threshold, adjust as needed
            max_distance = self.fallback_map_radius
        return max_distance

    def label_maps(self):
        for map_id, ref_map in self.maps.items():
            max_features = []
            print(f"Extracting features for {map_id}...")

            for point in tqdm(ref_map):
                # Check if the point is within the map bounds radius
                # if np.linalg.norm(point[:3] - self.map_centers[map_id]) > self.map_bounds_radius[map_id]:
                #     max_features.append(0.5)  # Assign stability score of 0.5 if outside bounds
                #     continue

                dis = []
                for query_map in self.maps_ids:
                    if query_map == map_id or len(self.maps[query_map]) < 1:
                        continue
                    occluded = self.is_point_occluded(point, query_map) if self.use_octomaps else False
                    if not occluded:
                        if self.search_in_radius:
                            d_radius = self.get_points_within_radius(point, self.maps[query_map], radius=self.radius)
                            dis.extend(d_radius)
                        else:
                            d = self.get_distance_to_closest_point(point, self.maps[query_map])
                            dis.append(d)
                    else:
                        dis.append(self.occluded_point)
                
                if len(dis):
                    max_dis = max(dis)
                else:
                    max_dis = np.inf # No correspondence found

                if max_dis != self.occluded_point:
                    max_dis = 1 - np.exp(-max_dis * (max_dis / 100))
            
                max_features.append(max_dis)

            labeled_map = np.hstack((ref_map, np.array(max_features).reshape(-1, 1)))
            labeled_map[:, -1] = 1 - labeled_map[:, -1]  # Using 1 for stability

            if self.lidar_labels is not None:
                labeled_map[:, -1] = labeled_map[:, -1] * self.lidar_labeled_scene_maps[map_id][:, -1]  # combine lidar proxy labels
            
            if self.dynamic_priors:
                voxel_hash_map = self.dynamic_priors[map_id]
                scan_dynamic_scores = voxel_hash_map.assign_scores_to_pointcloud(labeled_map[:, :3])
                labeled_map[:, -1] *= scan_dynamic_scores[:, -1]

            labeled_map[:, -1] = np.clip(labeled_map[:, -1], 0, 1)

            if self.filter_out_of_bounds:
                labeled_map = self.adjust_stability_scores(labeled_map, map_id)

            self.labelled_maps[map_id] = labeled_map

        self.labeled_environment_map = np.vstack([m for m in self.labelled_maps.values()])


    def get_distance_to_closest_point(self, point, points, euclidean_weight=1, rcs_weight=0):
        # Extract the RCS value of the point
        if rcs_weight > 0:
            point_rcs = point[3]  # Assuming the 4th column (index 3) is RCS
            # Calculate the differences in RCS values
            rcs_differences = np.abs(points[:, 3] - point_rcs)
        else:
            rcs_differences = 0

        # Calculate Euclidean distances
        euclidean_distances = np.linalg.norm(points[:, :3] - point[:3], axis=1)

        # Combine Euclidean distance and RCS difference to form a composite distance
        # Here we use adjustable weights for both distance and RCS difference
        combined_distances = np.sqrt((euclidean_weight * euclidean_distances)**2 + (rcs_weight * rcs_differences)**2)
        return np.min(combined_distances)
    

    def get_points_within_radius(self, point, points, radius=1, euclidean_weight=1, rcs_weight=0):
        """
        Get all points within a specified radius from the given point.

        Parameters:
        - point: The reference point.
        - points: An array of points to check against.
        - radius: The radius within which to find points (default is 0.5 meters).
        - euclidean_weight: The weight for the Euclidean distance.
        - rcs_weight: The weight for the RCS difference.

        Returns:
        - A list of points within the specified radius.
        """
        # Extract the RCS value of the point
        if rcs_weight > 0:
            point_rcs = point[3]  # Assuming the 4th column (index 3) is RCS
            # Calculate the differences in RCS values
            rcs_differences = np.abs(points[:, 3] - point_rcs)
        else:
            rcs_differences = np.zeros(len(points))  # Ensure the array shape matches

        # Calculate Euclidean distances
        euclidean_distances = np.linalg.norm(points[:, :3] - point[:3], axis=1)

        # Combine Euclidean distance and RCS difference to form a composite distance
        combined_distances = np.sqrt((euclidean_weight * euclidean_distances)**2 + (rcs_weight * rcs_differences)**2)

        # Find indices of points within the specified radius
        within_radius_indices = np.where(combined_distances <= radius)[0]

        # Get the points within the specified radius
        points_within_radius = points[within_radius_indices]

        return combined_distances[within_radius_indices].tolist()


    def is_point_occluded(self, point, map_id):
        # Placeholder function to check if a point is occluded using octomap
        # Replace with actual occlusion check logic
        if self.use_octomaps and map_id in self.octomaps:
            octree = self.octomaps[map_id]
            node = octree.search(point)
            try:
                if node and octree.isNodeOccupied(node):
                    return True
            except octomap.NullPointerException:
                return False # NOTE: If a search fails, then return occluded
        return False

    def label_scan(self, scan, map_id=None):
        if map_id is None or self.use_combined_map:
            print(f"Registering scan to combined environment map...")
            target_map = self.labeled_environment_map
        else:
            print(f"Registering scan to {map_id} map...")
            target_map = self.labelled_maps[map_id]

        transformation = self.icp_registration(scan[:, :3], target_map[:, :3])
        scan[:, :3] = self.apply_transformation(scan[:, :3], transformation)

        scan_labels = []
        labeled_map_points = target_map[:, :3]
        labeled_map_labels = target_map[:, -1]

        for point in tqdm(scan[:, :3]):
            # Check if the point is within the map bounds radius
            # if map_id is not None and np.linalg.norm(point[:3] - self.map_centers[map_id]) > self.map_bounds_radius[map_id]:
            #     scan_labels.append(0.5)  # Assign stability score of 0.5 if outside bounds
            #     continue

            distances = np.linalg.norm(labeled_map_points - point, axis=1)
            closest_point_idx = np.argmin(distances)
            closest_distance = distances[closest_point_idx]

            if closest_distance <= self.radius:
                scan_labels.append(labeled_map_labels[closest_point_idx])
            else:
                scan_labels.append(0)  # Assign stability score of 0 if no correspondence within radius
                # scan_labels.append(0.5)  # Assign stability score of 0.5 if no correspondence within radius

        scan_labels = np.array(scan_labels)
        labeled_scan = np.hstack((scan, scan_labels.reshape(-1, 1)))
        if self.filter_out_of_bounds:
            labeled_scan = self.adjust_stability_scores(labeled_scan, map_id)
            
        return labeled_scan
        

    def icp_registration(self, source_points, target_points):
        source = o3d.geometry.PointCloud()
        target = o3d.geometry.PointCloud()
        source.points = o3d.utility.Vector3dVector(source_points)
        target.points = o3d.utility.Vector3dVector(target_points)

        threshold = 0.02  # Distance threshold for ICP
        trans_init = np.identity(4)
        reg_p2p = o3d.pipelines.registration.registration_icp(
            source, target, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint())
        return reg_p2p.transformation

    def apply_transformation(self, points, transformation):
        points_hom = np.hstack((points, np.ones((points.shape[0], 1))))
        transformed_points_hom = points_hom.dot(transformation.T)
        return transformed_points_hom[:, :3]

    def plot_bev(self, points, labels, title="Bird's Eye View", size=1):
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.show()

    def plot_labeled_map_bev(self, map_id, size=1):
        if type(map_id) == str:
            labeled_map = self.labelled_maps[map_id]
            center = self.map_centers[map_id]
            radius = self.map_bounds_radius
        else:
            labeled_map = map_id
        
        points = labeled_map[:, :3]
        labels = labeled_map[:, -1]

        mean_point = np.mean(points, axis=0) # Color rescaling fix
        points = np.vstack([points, mean_point, mean_point + 1])
        labels = np.hstack([labels, [0, 1]])

        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)

        # circle = plt.Circle((center[0], center[1]), radius, color='blue', fill=False, linewidth=2)
        # plt.gca().add_artist(circle)
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(f"BEV of Labeled Map {map_id}")
        plt.show()

    def plot_labeled_scan_bev(self, labeled_scan, size=1):
        points = labeled_scan[:, :3]
        labels = labeled_scan[:, -1]
        
        mean_point = np.mean(points, axis=0) # Color rescaling fix
        points = np.vstack([points, mean_point, mean_point + 1])
        labels = np.hstack([labels, [0, 1]])
        
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)
        
        
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("Labeled scan and map")

    def save_bev_plot(self, map_id, save_path, title="Labelled Map Bird's Eye View", size=1):
        if type(map_id) == str:
            labeled_map = self.labelled_maps[map_id]
            center = self.map_centers[map_id]
            radius = self.map_bounds_radius
        else:
            labeled_map = map_id

        points = labeled_map[:, :2]
        labels = labeled_map[:, -1]

        mean_point = np.mean(points, axis=0) # color rescaling fix
        points = np.vstack([points, mean_point, mean_point + 1])
        labels = np.hstack([labels, [0, 1]])
        
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)

        # circle = plt.Circle((center[0], center[1]), radius, color='blue', fill=False, linewidth=2)
        # plt.gca().add_artist(circle)
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.savefig(save_path)
        plt.close()

