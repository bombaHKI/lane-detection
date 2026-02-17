"""Recreate car trajectory from LiDAR GPS timestamps."""

import numpy as np
from typing import Tuple, List, Optional, DefaultDict
import laspy
from collections import defaultdict
from timeit import default_timer as timer
from lane_detection.utils import save_trajectory_geojson, build_pulse_map


def find_points_in_time_window(
    time_to_points: DefaultDict[float, List[np.ndarray]],
    timestamps: np.ndarray,
    unique_times: np.ndarray,
    target_time: float,
    time_window: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points within time window and optionally within spatial epsilon of previous location.
    
    Args:
        time_to_points: Map from time -> Nx3 array of XYZ coordinates
        timestamps: N array of GPS timestamps
        unique_times: Sorted array of all the unique timestamp values
        target_time: Target timestamp to search around
        time_window: Time window in milliseconds (±window)
        
    Returns:
        Tuple of (filtered_points, filtered_timestamps)
    """
    # Time filtering
    lo, hi = np.searchsorted(unique_times, [target_time - time_window, target_time + time_window])
    candidate_times = unique_times[lo:hi]

    if candidate_times.size == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=timestamps.dtype)

    filtered_points: List[np.ndarray] = []
    filtered_timestamps: List[float] = []

    for t in candidate_times:
        pts = time_to_points[float(t)]
        if pts.size:
            filtered_points.extend(pts)
            filtered_timestamps.extend([float(t)] * len(pts))

    if len(filtered_points) == 0:
        return np.empty((0, 3)), np.empty((0,), dtype=timestamps.dtype)

    return np.asarray(filtered_points), np.asarray(filtered_timestamps, dtype=timestamps.dtype)


def estimated_pos(points: np.ndarray) -> Optional[np.ndarray]:
    """
    Calculate estimated position from given points.
    Current implementation: median of xyz coordinates.
    
    Args:
        points: Nx3 array of XYZ coordinates
        
    Returns:
        Median location [x, y, z] or None if no points
    """
    if len(points) == 0:
        return None
    return np.median(points, axis=0)


def recreate_trajectory(
    file_path: str,
    initial_offset: float = 4.0,
    time_window: float = .1,
    time_step: float = 0.5,
    output_path: str = "data/geojson/car_trajectory.geojson"
) -> List[Tuple[np.ndarray, float]]:
    """
    Recreate car trajectory from LiDAR GPS timestamps.
    
    Args:
        file_path: Path to LiDAR file (.laz or .las)
        initial_offset: Seconds after minimum timestamp to start (default: 4)
        time_window: Time window in milliseconds for point selection (±window, default: .1)
        time_step: Seconds to increment time each iteration (default: .5)
        epsilon: Maximum distance in meters for spatial filtering (default: 10)
        output_path: Path to save GeoJSON output
        
    Returns:
        List of (location, timestamp) tuples representing the trajectory
    """
    print(f"Loading point cloud from {file_path}...")
    
    # Read point cloud using laspy for direct access to gps_time
    las = laspy.read(file_path)
    points = las.xyz  # Nx3 array
    
    if 'gps_time' not in las.point_format.dimension_names:
        raise ValueError(f"No 'gps_time' attribute found in {file_path}")
    
    timestamps = las.gps_time  # GPS timestamps

    
    print(f"Loaded {len(points)} points")
    print(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    print(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
    # Initialize
    min_timestamp = timestamps.min()
    current_time = min_timestamp + initial_offset
    max_timestamp = timestamps.max()
    
    trajectory = []
    
    start = timer()
    print(f"\nStarting trajectory reconstruction.")
    print(f"Initial time: {current_time:.3f}")
    print(f"Time window: ±{time_window} ms")
    print(f"Time step: {time_step} ms")

    print(f"Mapping points to time values")
    time_to_points, unique_times = build_pulse_map(las.xyz, timestamps)
    print(f"Mapping done in {timer()-start} seconds")
    
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            print(f"Progress: {curr_progress:.2f}%")

        # Find points within time window and spatial constraints
        filtered_points, filtered_times = find_points_in_time_window(
            time_to_points, timestamps, unique_times, current_time, time_window
        )
        


        if len(filtered_points) > 0:
            # Calculate average location
            avg_location = estimated_pos(filtered_points)
            trajectory.append((avg_location, current_time))
        
        # Increment time
        current_time += time_step
    
    print(f"\nTrajectory reconstruction complete! Took: {(timer()-start):.2f} seconds")
    print(f"Total trajectory points: {len(trajectory)}")
    
    if len(trajectory) == 0:
        print("Warning: No trajectory points generated!")
        return trajectory
    
    # Save as GeoJSON
    save_trajectory_geojson(trajectory, output_path, file_path)
    
    return trajectory

def main():
    """Main entry point."""
    recreate_trajectory(
        file_path="data/LiDaR/871e1d886ffffff_cegl_m4_2.laz",
        initial_offset=4.0,
        time_window=.05,
        time_step=.2,
        output_path="data/geojson/car_trajectory_v1_window_median.geojson"
    )


if __name__ == "__main__":
    main()
