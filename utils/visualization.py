import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt


def plot_maps(scene_maps, poses, size=0.5, zoom_level=3):
    plt.figure(figsize=(10, 10))

    colors = ['cyan', 'magenta', 'yellow', 'black', 'purple', 'brown']
    markers = ['o', 'v', 's', 'P', 'X', 'D']
    positions = {name: np.array(pose)[:, :3,3] for name,pose in poses.items()}

    for idx, (scene_name, map) in enumerate(scene_maps.items()):
        plt.scatter(map[:, 0], map[:, 1], s=size, label=scene_name, alpha=0.5)


        if scene_name in positions:
            pose = np.array(positions[scene_name])
            plt.plot(pose[:, 0], pose[:, 1], label=f'{scene_name} Trajectory', color=colors[idx % len(colors)])
            # plt.scatter(pose[:, 0], pose[:, 1], c=colors[idx % len(colors)], marker=markers[idx % len(markers)])
            
            for i in range(1, len(pose)):
                plt.arrow(pose[i-1, 0], pose[i-1, 1], pose[i, 0] - pose[i-1, 0], pose[i, 1] - pose[i-1, 1], 
                          head_width=0.5, head_length=0.5, fc=colors[idx % len(colors)], ec=colors[idx % len(colors)])

    plt.xlabel('X')
    plt.ylabel('Y')

    if zoom_level != -1:
        all_pos = np.concatenate(list(positions.values()))
        x_mean = np.mean(all_pos[:,0])
        y_mean = np.mean(all_pos[:,1])
        x_std = np.std(all_pos[:,0])
        y_std = np.std(all_pos[:,1])
        
        std_dev_range = zoom_level

        x_limits = [x_mean - std_dev_range*x_std, x_mean + std_dev_range*x_std]
        y_limits = [y_mean - std_dev_range*y_std, y_mean + std_dev_range*y_std]

        plt.xlim(x_limits)
        plt.ylim(y_limits)

    plt.title("Overlapped Maps and Trajectories")
    plt.legend()
    plt.grid(True)
    plt.show()



def plot_voxel_hash_map_open3d(normalized_scores, voxel_hash_map, voxel_size):
    points = []
    colors = []
    cmap = plt.cm.get_cmap('RdYlGn')

    for (ix, iy, iz), score in normalized_scores.items():
        x = (ix + 0.5) * voxel_size  # Center of the voxel
        y = (iy + 0.5) * voxel_size
        z = (iz + 0.5) * voxel_size
        if voxel_hash_map.voxel_map[(ix, iy, iz)] < 1:
            color = [1, 1, 1]  # White for voxels with no points
        else:
            color = cmap(score)[:3]  # Get RGB color from colormap
        points.append([x, y, z])
        colors.append(color)

    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(points)
    point_cloud.colors = o3d.utility.Vector3dVector(colors)

    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(point_cloud, voxel_size, point_cloud.get_min_bound(), point_cloud.get_max_bound())
    
    o3d.visualization.draw_geometries([voxel_grid])

    # Create colorbar
    fig, ax = plt.subplots(figsize=(6, 1))
    fig.subplots_adjust(bottom=0.5)

    norm = plt.Normalize(vmin=0, vmax=1)
    cb1 = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
    cb1.set_label('Static Score')

    plt.show()


def plot_voxel_map(voxel_hash_map, voxel_size):
    # Retrieve the normalized scores
    normalized_scores = voxel_hash_map.get_normalized_scores()

    plot_voxel_hash_map_open3d(normalized_scores, voxel_hash_map, voxel_size)