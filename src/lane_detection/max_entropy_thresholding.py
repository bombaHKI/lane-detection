import laspy
import open3d as o3d
import numpy as np

file_path = "data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in_ground.laz"
las = laspy.read(file_path)

points = las.xyz
intensities = las.intensity
val, H = np.unique(intensities, return_counts=True)

# 2. Probability Distribution
# We normalize the histogram to get probabilities (p_i) right away.
counts = H.astype(float)
probs = counts / counts.sum()

# 3. Vectorized Cumulative Calculation (Kapur's Method)
# Cumulative probability (omega)
P_cumulative = np.cumsum(probs)
log_probs = np.log(probs, out=np.zeros_like(probs), where=probs!=0) # Safe log
H_cumulative = np.cumsum(probs * -log_probs)

H_total = H_cumulative[-1]

# 4. Calculate Total Entropy (Phi) for every possible threshold T
epsilon = 1e-10 
P_omega = P_cumulative  # Probability of foreground
P_omega_bar = 1.0 - P_cumulative # Probability of background

term1 = np.log(P_omega + epsilon) + (H_cumulative / (P_omega + epsilon))
term2 = np.log(P_omega_bar + epsilon) + ((H_total - H_cumulative) / (P_omega_bar + epsilon))

Phi = term1 + term2

# Edge cases (first and last indices usually invalid for splitting)
Phi[0] = -np.inf
Phi[-1] = -np.inf

# 5. Find Max
max_index = np.argmax(Phi)
theta_max = val[max_index]
treshold = max(theta_max-20, 20)
mask = intensities > treshold

print(f"Optimal Threshold (Theta max): {theta_max}. Using: {treshold}")
print("writing to files")
close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in_ground_marking.laz")

close_points = laspy.create(point_format=las.header.point_format, file_version=las.header.version)
close_points.points = las.points[~mask]
close_points.write("data/LiDaR/871e1d886ffffff_cegl_m4_2_close_sor_in_ground_non_marking.laz")