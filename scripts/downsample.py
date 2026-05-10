import laspy
import numpy as np

# --- 1. Read LAS ---
in_folder = "data/output/cegl_2/"
out_folder = "data/output/cegl_2/downsample/"
file_names = [
    "4_road_surface_in.laz",
    "3_intensity_in.laz",
    "2_ground_in.laz",
]

for f in file_names:
    las = laspy.read(in_folder+f)

    points = np.vstack((las.x, las.y, las.z)).T
    intensity = las.intensity

    voxel_size = 0.15  # adjust

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)

    dtype = [('x', int), ('y', int), ('z', int)]
    structured_voxels = np.array(list(map(tuple, voxel_indices)), dtype=dtype)

    unique_voxels, inverse_indices = np.unique(structured_voxels, return_inverse=True)

    # --- 4. Aggregate per voxel ---
    num_voxels = len(unique_voxels)

    sum_points = np.zeros((num_voxels, 3))
    sum_intensity = np.zeros(num_voxels)
    counts = np.zeros(num_voxels)

    # Accumulate
    np.add.at(sum_points, inverse_indices, points)
    np.add.at(sum_intensity, inverse_indices, intensity)
    np.add.at(counts, inverse_indices, 1)

    # Compute averages
    centroids = sum_points / counts[:, None]
    avg_intensity = sum_intensity / counts

    # --- 5. Write output LAS ---
    header = laspy.LasHeader(point_format=las.point_format, version=las.header.version)
    header.offsets = las.header.offsets
    header.scales = las.header.scales
    out_las = laspy.LasData(header=header)

    out_las.x = centroids[:, 0]
    out_las.y = centroids[:, 1]
    out_las.z = centroids[:, 2]

    # Intensity must be integer type (usually uint16)
    out_las.intensity = avg_intensity.astype(np.uint16)

    out_las.write(out_folder+f)
    print(f"written: {f}, {len(las.x)} points reduced to {len(out_las.x)}")