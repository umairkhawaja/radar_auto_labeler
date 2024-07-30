from tqdm import tqdm
import octomap
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from utils.transforms import *

OCCLUDED_POINT = -1  # Define a constant for occluded points

class AutoLabeler:
    def __init__(self,
                 scene_maps, 
                 ref_map_id, 
                 scene_octomaps, 
                 lidar_labels=None, 
                 dynamic_priors=None, 
                 use_octomaps=True, 
                 search_in_radius=False,
                 downsample=False, 
                 radius=1,
                 voxel_size=0.5):
        self.use_octomaps = use_octomaps
        self.maps = scene_maps
        self.octomaps = scene_octomaps
        self.lidar_labels = lidar_labels
        self.dynamic_priors = dynamic_priors
        self.search_in_radius = search_in_radius
        self.radius = radius

        if downsample:
            self.maps = {}
            for name,map in scene_maps.items():
                # Create an Open3D point cloud object
                point_cloud = o3d.geometry.PointCloud()
                point_cloud.points = o3d.utility.Vector3dVector(map[:, :3])
                ## TODO: Try adding dyn_prop and compare those points, if dyn_prop does not match then return np.inf
                # rcs_as_normals = np.zeros((map.shape[0], 3))
                # rcs_as_normals[:,0] = map[:,3]
                # rcs_as_normals[:,1] = map[:,-1] # compensated velocities
                
                # point_cloud.normals = o3d.utility.Vector3dVector(rcs_as_normals) # RCS 
                # Downsample the point cloud
                ref_map_sampled = point_cloud.voxel_down_sample(voxel_size=voxel_size)
                # ref_map_sampled_rcs = np.asarray(ref_map_sampled.normals)[:,0].reshape(-1, 1)
                # ref_map_sampled_cv = np.asarray(ref_map_sampled.normals)[:,1].reshape(-1, 1)
                # ref_map_sampled = np.asarray(ref_map_sampled.points)
                map_points = np.asarray(ref_map_sampled.points)

                # map_points = np.hstack([ref_map_sampled, ref_map_sampled_rcs, ref_map_sampled_cv])
                # map_points = map_points[map_points[:, -1] <= 0.25]
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


    def label_maps(self):
        for map_id, ref_map in self.maps.items():
            max_features = []
            print(f"Extracting features for {map_id}...")

            for point in tqdm(ref_map):
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


                        # if d < 0.05: # If its close enough
                        #     dis.append(d)
                        # else:
                        #     # not a good correspondence
                        #     dis.append(OCCLUDED_POINT)
                    else:
                        dis.append(OCCLUDED_POINT)
                
                if len(dis):
                    max_dis = max(dis)
                else:
                    max_dis = np.inf

                if max_dis != OCCLUDED_POINT:
                    max_dis = 1 - np.exp(-max_dis * (max_dis / 100))
            

                max_features.append(max_dis)

            labeled_map = np.hstack((ref_map, np.array(max_features).reshape(-1, 1)))
            labeled_map[:, -1] = 1 - labeled_map[:, -1] # Using 1 for stability
            # labeled_map[:, -1] = labeled_map[:, -1] # Using 1 for instability
            # labeled_map = labeled_map[labeled_map[:, -1] <= 1.0]

            if self.lidar_labels is not None:
                labeled_map[:, -1] = labeled_map[:, -1] * self.lidar_labeled_scene_maps[map_id][:, -1] # combine lidar proxy labels
            
            if self.dynamic_priors:
                voxel_hash_map = self.dynamic_priors[map_id]
                scan_dynamic_scores = voxel_hash_map.assign_scores_to_pointcloud(labeled_map[:, :3])
                labeled_map[:, -1] *= scan_dynamic_scores[:,-1]

            labeled_map[:, -1] = np.clip(0,1, labeled_map[:, -1])
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
        if map_id is None:
            # Use the combined "environment" map
            print(f"Registering scan to combined environment map...")
            target_map = self.labeled_environment_map
        else:
            print(f"Registering scan to {map_id} map...")
            target_map = self.labelled_maps[map_id]

        # Register scan to labeled map
        transformation = self.icp_registration(scan[:, :3], target_map[:, :3])
        scan[:, :3] = self.apply_transformation(scan[:, :3], transformation)

        # Transfer labels from labeled map to scan
        scan_labels = []
        labeled_map_points = target_map[:, :3]
        labeled_map_labels = target_map[:, -1]

        
            
        for point in tqdm(scan[:, :3]):
            distances = np.linalg.norm(labeled_map_points - point, axis=1)
            closest_point_idx = np.argmin(distances)
            scan_labels.append(labeled_map_labels[closest_point_idx])

        scan_labels = np.array(scan_labels)

        labeled_scan = np.hstack((scan, scan_labels.reshape(-1, 1)))
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

    def plot_bev(self, points, labels, title="Bird's Eye View",size=1):
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.show()

    def plot_labeled_map_bev(self, labeled_map, size=1):
        points = labeled_map[:, :3]
        labels = labeled_map[:, -1]
        self.plot_bev(points, labels, title=f"BEV of Labeled Map", size=size)

    def save_bev_plot(self, labeled_map, save_path, title="Bird's Eye View", size=1):
        # Extract points and labels from the labeled map
        points = labeled_map[:, :2]  # Assuming 2D points are in the first two columns
        labels = labeled_map[:, -1]  # Assuming labels are in the last column

        # Create the plot
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)

        # Save the plot to the specified path
        plt.savefig(save_path)
        plt.close()

    def plot_labeled_scan_bev(self, labeled_scan, size=1):
        points = labeled_scan[:, :3]
        labels = labeled_scan[:, -1]
        plt.figure(figsize=(10, 10))
        plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='RdYlGn', s=size)
        
        # self.labels[map_id] = np.load(os.path.join(self.main_dir, 'labelled', f"{map_id}_labeled.npy"))
        # map_points = self.labels[map_id][:, :3]
        # map_labels = self.labels[map_id][:, 3]
        # plt.scatter(map_points[:, 0], map_points[:, 1], c=map_labels, cmap='Accent', s=5)
        
        plt.colorbar(label='Stability')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title("Labeled scan and map")


    def load_octomap(self, filepath):
        # Placeholder function to load octomap
        # Replace with actual implementation
        with open(filepath, 'rb') as f:
            return octomap.OcTree(f.read())