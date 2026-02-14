"""Recreate car trajectory from LiDAR GPS timestamps."""

import json
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import laspy


def find_points_in_time_window(
    points: np.ndarray,
    timestamps: np.ndarray,
    target_time: float,
    time_window: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points within time window and optionally within spatial epsilon of previous location.
    
    Args:
        points: Nx3 array of XYZ coordinates
        timestamps: N array of GPS timestamps
        target_time: Target timestamp to search around
        time_window: Time window in milliseconds (±window)
        
    Returns:
        Tuple of (filtered_points, filtered_timestamps)
    """
    # Time filtering
    mask = np.abs(timestamps - target_time) <= time_window
    return points[mask], timestamps[mask]


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
    prev_location = None
    
    print(f"\nStarting trajectory reconstruction...")
    print(f"Initial time: {current_time:.3f}")
    print(f"Time window: ±{time_window} ms")
    print(f"Time step: {time_step} ms")
    
    no_points_count = 0
    max_no_points = 20  # Stop if no points found for 100 consecutive iterations
    
    while current_time <= max_timestamp:
        print(f"Timestamp: {current_time}")
        # Find points within time window and spatial constraints
        filtered_points, filtered_times = find_points_in_time_window(
            points, timestamps, current_time, time_window
        )
        
        if len(filtered_points) > 0:
            # Calculate average location
            avg_location = estimated_pos(filtered_points)
            trajectory.append((avg_location, current_time))
            prev_location = avg_location
            no_points_count = 0
            
            print(f"Time={current_time:.3f}, Points={len(filtered_points)}, "
                      f"Location=[{avg_location[0]:.2f}, {avg_location[1]:.2f}, {avg_location[2]:.2f}]")
        else:
            no_points_count += 1
            if no_points_count >= max_no_points:
                print(f"\nNo points found for {max_no_points} consecutive iterations. Stopping.")
                break
        
        # Increment time
        current_time += time_step
    
    print(f"\nTrajectory reconstruction complete!")
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
        output_path="data/geojson/car_trajectory.geojson"
    )


if __name__ == "__main__":
    main()
