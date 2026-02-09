import laspy
import open3d as o3d
import numpy as np
from collections import defaultdict

from .utils import read_to_3d, export_pcd_to_laz

# Check if file exists
file_path = "data/LiDaR/871e1d880ffffff_cegl_m4_3.laz"
file_path = "data/LiDaR/871e1d886ffffff_cegl_m4_2.laz"
las = laspy.read(file_path)
pcd = read_to_3d(file_path)

print(list(las.point_format.dimension_names))

grid = defaultdict(list)
points = pcd.point.positions.numpy()
grid_size = 5
print("Points to grids")
for i, cell in enumerate(points):
    gx, gy = int(cell[0]/grid_size), int(cell[1]/grid_size)
    grid[(gx,gy)].append(i)


keep_mask = np.zeros(len(points), dtype=bool)


print("Ground segment in grids")
i=0
prevPercentage = 0
for cell, indices in grid.items():
    i+=1
    percentage = int(i/len(grid)*100)
    if percentage%10 == 0 and percentage != prevPercentage:
        print(f"Progress: {percentage}%")
        prevPercentage = percentage
    indices=np.asarray(indices)
    cell_points = points[indices]
    z_vals = cell_points[:,2]

    k = int(np.ceil(0.1 * len(indices)))

    order = np.argsort(z_vals)
    mid_idx = order[k:5*k]

    if len(mid_idx) < 3:
        continue

    pcd_mid = o3d.geometry.PointCloud()

    pcd_mid.points = o3d.utility.Vector3dVector(cell_points[mid_idx])
    plane_model, inliers = pcd_mid.segment_plane(
        distance_threshold=0.05,  # tighter threshold for fitting
        ransac_n=3,
        num_iterations=100
    )

    a, b, c, d = plane_model
    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    dist = np.abs(
        cell_points @ normal + d
    ) / normal_norm


    ground_mask = dist < 0.2  # 20 cm
    keep_mask[indices[ground_mask]] = True

ground_points = pcd.select_by_mask(keep_mask)
non_ground_points = pcd.select_by_mask(~keep_mask)

export_pcd_to_laz(ground_points, "data/LiDaR/871e1d886ffffff_cegl_m4_2_ground.laz")
export_pcd_to_laz(non_ground_points, "data/LiDaR/871e1d886ffffff_cegl_m4_2_non_ground.laz")

