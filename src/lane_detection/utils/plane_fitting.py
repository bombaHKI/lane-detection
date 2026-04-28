import open3d as o3d
import numpy as np

def fit_plane_ransac(points: np.ndarray, residual_threshold: float = 0.05):
    """Fit z = a*x + b*y + c with RANSAC via open3d. Returns (a, b, c, inlier_mask) or None."""
    if len(points) < 3:
        return None
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    try:
        plane_model, inlier_list = pcd.segment_plane(
            distance_threshold=residual_threshold,
            ransac_n=3,
            num_iterations=60,
        )
    except Exception:
        return None
    # open3d plane: px*x + py*y + pz*z + pd = 0  =>  z = -(px*x + py*y + pd) / pz
    px, py, pz, pd = plane_model
    if abs(pz) < 1e-9:   # near-vertical plane, can't express as z = f(x,y)
        return None
    a = -px / pz
    b = -py / pz
    c = -pd / pz
    inlier_mask = np.zeros(len(points), dtype=bool)
    inlier_mask[inlier_list] = True
    return float(a), float(b), float(c), inlier_mask
