"""Utility functions for LiDAR data processing."""

import laspy
import numpy as np
import open3d as o3d
import os


def read_to_3d(file_path: str) -> o3d.t.geometry.PointCloud:
    """
    Read a LAS/LAZ file and convert it to an Open3D tensor PointCloud.

    Args:
        file_path: Path to the LAS/LAZ file

    Returns:
        Open3D tensor PointCloud with positions, intensity, colors, and all other attributes
    """
    print(f"File exists: {os.path.exists(file_path)}")
    las = laspy.read(file_path)
    pcd_t = o3d.t.geometry.PointCloud()
    pcd_t.point.positions = las.xyz

    intensities = las["intensity"]
    colors = np.column_stack((intensities, intensities, intensities)) / 255
    pcd_t.point.intensity = las["intensity"] / 255
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
            # assumes only 1d. Otherwise you need vstack which is slower
            las_attr = np.array(las[attr])[:, None]

        pcd_t.point[attr] = las_attr

    return pcd_t


def export_pcd_to_laz(
    pcd: o3d.t.geometry.PointCloud,
    out_path: str,
    scale=(0.001, 0.001, 0.001),
    offset=None,
):
    """
    Export an Open3D tensor PointCloud to a LAZ file.

    Args:
        pcd: Open3D tensor PointCloud to export
        out_path: Output path for the LAZ file
        scale: Scale factors for X, Y, Z coordinates (default: (0.001, 0.001, 0.001))
        offset: Offset for coordinates (default: minimum XYZ values)
    """
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
