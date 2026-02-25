"""Utility functions for LiDAR data processing."""

import laspy
import numpy as np
import open3d as o3d
import os
import json
from pathlib import Path
from typing import List, Tuple
import geopandas as gpd

def geojson_to_geopackage(path_from):
    """
    Reads geojson from geojson, and exports it as geopackage (with the same name)
    """
    gdf = gpd.read_file(path_from)
    # Define output GeoPackage path
    output_dir = "../../data/geopackage"
    output_name = path_from.split("/")[-1].split(".")[-2]

    os.makedirs(output_dir, exist_ok=True)
    gpkg_path = os.path.join(output_dir, output_name+".gpkg")

    # Write to GeoPackage
    gdf.to_file(gpkg_path, driver="GPKG")
    print(f"GeoPackage written to: {gpkg_path}")
    print(f"Features: {len(gdf)}, CRS: {gdf.crs}")

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

def save_trajectory_geojson(
    trajectory: List[Tuple[np.ndarray, float]],
    output_path: str,
    source_file: str
):
    """
    Save trajectory as GeoJSON file.
    
    Note: This saves coordinates in the original projection (likely Web Mercator EPSG:3857).
    
    Args:
        trajectory: List of (location, timestamp) tuples
        output_path: Output file path
        source_file: Source LiDAR file path (for metadata)
    """
    if len(trajectory) == 0:
        print(f"No trajectory to save to {output_path}")
        return
    
    # Extract coordinates (X, Y, Z) using numpy for efficiency
    locations = np.array([loc for loc, _ in trajectory])
    coordinates = locations.astype(float).tolist()
    
    # Get start and end timestamps
    start_time = float(trajectory[0][1])
    end_time = float(trajectory[-1][1])
    
    # Create GeoJSON structure with CRS information
    geojson = {
        "type": "FeatureCollection",
        "crs": {
            "type": "name",
            "properties": {
                "name": "urn:ogc:def:crs:EPSG::3857"
            }
        },
        "features": [
            {
                "type": "Feature",
                "properties": {
                    "name": "Car Trajectory",
                    "source_file": str(source_file),
                    "num_points": len(trajectory),
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration": end_time - start_time
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coordinates
                }
            }
        ]
    }
    
    # Save to file
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(geojson, f, indent=2)
    
    print(f"Trajectory saved to {output_path}")
    print(f"  Points: {len(trajectory)}")
    print(f"  Time range: {start_time:.3f} to {end_time:.3f}")
    print(f"  Duration: {end_time - start_time:.3f} time units")

def build_pulse_map(points: np.ndarray, gps_times: list):
    """
    Builds a timestamp -> ndarray of points map
    
    Args:
        :param points: The points in the lidar data
        :type points: np.ndarray
        :param gps_times: The timestamps corresponding to each point 
        :type gps_times: list
    Returns:
        the timestamp->points map and the `unique_timestamps` in the gps_times
    """
    # 1. Sort by time (returns indices that would sort the array)
    # This is the most expensive step: O(N log N)
    sort_idx = np.argsort(gps_times)
    
    # 2. Apply sorting to both arrays
    sorted_points = points[sort_idx]
    sorted_times = gps_times[sort_idx]
    
    # 3. Find unique times and the split indices
    # return_index=True gives the first index where each unique value appears
    unique_times, start_indices = np.unique(sorted_times, return_index=True)
    
    # 4. Split the points array into chunks based on start indices
    # We skip start_indices[0] because it's always 0 (the beginning)
    grouped_points = np.split(sorted_points, start_indices[1:])
    
    # 5. Combine into a dictionary
    # zip is fast here because we are zipping 412k items, not 104M
    return dict(zip(unique_times, grouped_points)), unique_times
