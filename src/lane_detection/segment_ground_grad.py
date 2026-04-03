import laspy
import open3d as o3d
import numpy as np
from collections import defaultdict

from lane_detection.utils_old import read_to_3d, export_pcd_to_laz

# Check if file exists
file_path = "data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in.laz"
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
    mid_idx = order[:5*k]

    if len(mid_idx) < 3:
        continue

    pcd_mid = o3d.geometry.PointCloud()
    pcd_mid.points = o3d.utility.Vector3dVector(cell_points[mid_idx])
    
    # Try to find a valid plane with gradient < 40%
    max_iterations = 5
    valid_plane_found = False
    
    for attempt in range(max_iterations):
        plane_model, inliers = pcd_mid.segment_plane(
            distance_threshold=0.05,
            ransac_n=3,
            num_iterations=100
        )

        a, b, c, d = plane_model
        
        # Calculate gradient (slope) from plane normal
        # Gradient = sqrt(a^2 + b^2) / |c|
        # 40% gradient = 0.4 = tan(angle) ≈ 21.8 degrees
        horizontal_component = np.sqrt(a**2 + b**2)
        vertical_component = np.abs(c)
        
        if vertical_component < 1e-6:  # nearly vertical plane
            gradient = float('inf')
        else:
            gradient = horizontal_component / vertical_component
        
        # Check if gradient is less than 40% (0.4)
        if gradient < 0.4:
            valid_plane_found = True
            break
        
        # Remove inliers and try again with remaining points
        if len(inliers) > 0:
            remaining_mask = np.ones(len(pcd_mid.points), dtype=bool)
            remaining_mask[inliers] = False
            remaining_points = np.asarray(pcd_mid.points)[remaining_mask]
            
            if len(remaining_points) < 3:
                break
                
            pcd_mid.points = o3d.utility.Vector3dVector(remaining_points)
        else:
            break
    
    if not valid_plane_found:
        # Skip this cell if no valid plane found
        continue

    normal = np.array([a, b, c])
    normal_norm = np.linalg.norm(normal)
    dist = np.abs(
        cell_points @ normal + d
    ) / normal_norm

    ground_mask = dist < 0.2  # 20 cm
    keep_mask[indices[ground_mask]] = True



print("writing to files")
close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[keep_mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in_ground.laz")

close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[~keep_mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in_non_ground.laz")