import laspy 
import open3d as o3d
import numpy as np
import os
from collections import defaultdict

def read_to_3d(file_path: str) -> o3d.t.geometry.PointCloud:
    print(f"File exists: {os.path.exists(file_path)}")
    las = laspy.read(file_path)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = las.xyz

    intensities = las["intensity"]
    colors = np.column_stack((intensities, intensities, intensities)) / 255
    pcd_t.point.intensity = las["intensity"]/255
    pcd_t.point.colors = colors
    
    # if rgb present, convert to o3d colors
    all_dims = list(las.point_format.dimension_names)[3:]
    all_dims.remove("intensity")

    for attr in all_dims:
        # If only has single value, this is much faster
        if np.all(las[attr] == las[attr][0]):
            # Fill with the single value
            las_attr = np.full((len(las[attr]), 1), las[attr][0])
        else:
            # assumes only 1d.  Otherwise you need vstack which is slower
            las_attr = np.array(las[attr])[:, None]

        pcd_t.point[attr] = las_attr

    return pcd_t

def export_pcd_to_laz(
    pcd: o3d.t.geometry.PointCloud,
    out_path: str,
    scale=(0.001, 0.001, 0.001),
    offset=None,
):
    # ---- extract positions ----
    xyz = pcd.point.positions.numpy()  # (N, 3)

    # ---- create LAS header ----
    header = laspy.LasHeader(point_format=3, version="1.2")

    header.scales = scale
    if offset is None:
        header.offsets = xyz.min(axis=0)
    else:
        header.offsets = offset

    las = laspy.LasData(header)

    # ---- mandatory coordinates ----
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]

    # ---- optional attributes ----
    for attr in pcd.point:
        if attr in ("positions", "colors"):
            continue

        values = pcd.point[attr].numpy()

        # flatten (N,1) → (N,)
        if values.ndim == 2 and values.shape[1] == 1:
            values = values[:, 0]

        if attr in las.point_format.dimension_names:
            las[attr] = values
        else:
            # create extra dimension
            las.add_extra_dim(
                laspy.ExtraBytesParams(
                    name=attr,
                    type=values.dtype,
                )
            )
            las[attr] = values

    # ---- colors (if present) ----
    if "colors" in pcd.point:
        colors = pcd.point.colors.numpy()
        colors = np.clip(colors * 65535, 0, 65535).astype(np.uint16)

        las.red = colors[:, 0]
        las.green = colors[:, 1]
        las.blue = colors[:, 2]

    # ---- write ----
    las.write(out_path)

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

# Convert boolean mask to indices for tensor pointcloud
ground_indices = np.where(keep_mask)[0]
non_ground_indices = np.where(~keep_mask)[0]

# Create new point clouds using indices
ground_points = o3d.t.geometry.PointCloud()
ground_points.point.positions = pcd.point.positions[ground_indices]
ground_points.point.intensity = pcd.point.intensity[ground_indices]
ground_points.point.colors = pcd.point.colors[ground_indices]

# Copy other attributes
for attr in pcd.point:
    if attr not in ("positions", "colors", "intensity"):
        ground_points.point[attr] = pcd.point[attr][ground_indices]

non_ground_points = o3d.t.geometry.PointCloud()
non_ground_points.point.positions = pcd.point.positions[non_ground_indices]
non_ground_points.point.intensity = pcd.point.intensity[non_ground_indices]
non_ground_points.point.colors = pcd.point.colors[non_ground_indices]

# Copy other attributes
for attr in pcd.point:
    if attr not in ("positions", "colors", "intensity"):
        non_ground_points.point[attr] = pcd.point[attr][non_ground_indices]

export_pcd_to_laz(ground_points, "data/LiDaR/871e1d886ffffff_cegl_m4_2_ground.laz")
export_pcd_to_laz(non_ground_points, "data/LiDaR/871e1d886ffffff_cegl_m4_2_non_ground.laz")

