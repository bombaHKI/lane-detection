import laspy
import open3d as o3d
import open3d.core as o3c
import numpy as np

file_path = "data/LiDaR/871e1d886ffffff_cegl_m4_2_close_to_trace.laz"
las = laspy.read(file_path)

# 1. Define the device (Use CUDA:0 if you have a GPU, otherwise CPU:0)
device = o3c.Device("CPU:0")

pcd = o3d.t.geometry.PointCloud(device)

# 2. Convert las.xyz to a float32 tensor directly on the target device
# float32 is generally faster for GPU operations than float64
positions_tensor = o3c.Tensor(las.xyz, o3c.float32, device)
pcd.point.positions = positions_tensor

print("starting sor (running on GPU...)")
cl, ind = pcd.remove_statistical_outliers(nb_neighbors=10, std_ratio=1.1)

# 3. Bring the indices back to the CPU as a numpy array for boolean masking
ind_np = ind.cpu().numpy().flatten()

# Note: np.bool is deprecated in newer numpy versions; use bool
mask = np.zeros(len(las.points), dtype=bool) 
mask[ind_np] = True

print("writing to files")
close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in.laz")

close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[~mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_out.laz")