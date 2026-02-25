"""Recreate car trajectory from LiDAR GPS timestamps."""

import numpy as np
from numpy.linalg import LinAlgError
from typing import Tuple, List, Optional, DefaultDict
import laspy
from collections import defaultdict
from timeit import default_timer as timer
from lane_detection.utils import save_trajectory_geojson, build_pulse_map

def recreate_trajectory(
    file_path: str,
    pulse_step: int = 5
) -> List[Tuple[np.ndarray, float]]:
    """
    Recreate car trajectory from LiDAR GPS timestamps.

    Adds the next pulse's closest point (to the current pos) to the trajectory.
    The next pulse is pulse_step pulses.
    
    Args:
        file_path: Path to LiDAR file (.laz or .las)
        pulse_step: Distance between two consecutive pulses
        
    Returns:
        List of (location, timestamp) tuples representing the trajectory
    """
    print(f"Loading point cloud from {file_path}...")
    
    # Read point cloud using laspy for direct access to gps_time
    las = laspy.read(file_path)
    points = las.xyz  # Nx3 array
    
    if 'gps_time' not in las.point_format.dimension_names:
        raise ValueError(f"No 'gps_time' attribute found in {file_path}")
    
    timestamps = las.gps_time
    
    print(f"Loaded {len(points)} points")
    print(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    print(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
    # Initialize
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    start = timer()
    print(f"Mapping points to time values")
    time_to_points, unique_times = build_pulse_map(las.xyz, timestamps)
    print(f"Mapping done in {timer()-start} seconds")
    print(f"\nStarting trajectory reconstruction.")
    
    trajectory = []
    pos_estimation = time_to_points[min_timestamp][0]
    progress = 0
    for curr_time in unique_times[::pulse_step]:
        curr_progress = (curr_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            print(f"Progress: {progress:.2f}%")

        curr_pulse = time_to_points[curr_time]
        # Find the closest point in curr_pulse to the current pos_estimation
        distances = np.linalg.norm(curr_pulse[:,:2] - pos_estimation[:2], axis=1)
        closest_idx = np.argmin(distances)
        pos_estimation = curr_pulse[closest_idx]
        trajectory.append((pos_estimation.copy(), curr_time))
    
    print(f"\nTrajectory reconstruction complete! Took: {(timer()-start):.2f} seconds")
    print(f"Total trajectory points: {len(trajectory)}")
    
    if len(trajectory) == 0:
        print("Warning: No trajectory points generated!")    
    return trajectory

def moving_avg_smoothing(trajectory: List[Tuple[np.ndarray, float]], window_size: int = 5) -> List[Tuple[np.ndarray, float]]:
    """
    Apply moving average smoothing to trajectory positions.
    
    Args:
        trajectory: List of (location, timestamp) tuples
        window_size: Size of the sliding window for averaging
        
    Returns:
        Smoothed trajectory with same format as input
    """
    if len(trajectory) < window_size:
        return trajectory
    
    points = np.array([pos for pos, _ in trajectory])
    timestamps = np.array([ts for _, ts in trajectory])
    
    smoothed_points = np.zeros_like(points)
    half_window = window_size // 2
    
    for i in range(len(points)):
        start_idx = max(0, i - half_window)
        end_idx = min(len(points), i + half_window + 1)
        smoothed_points[i] = np.mean(points[start_idx:end_idx], axis=0)
    
    smoothed_trajectory = [(smoothed_points[i], timestamps[i]) for i in range(len(trajectory))]
    
    return smoothed_trajectory

def window_avging(trajectory: List[Tuple[np.ndarray, float]], window_size: int = 5) -> List[Tuple[np.ndarray, float]]:
    if len(trajectory) < window_size:
        return trajectory
    
    points = np.array([pos for pos, _ in trajectory])
    timestamps = np.array([ts for _, ts in trajectory])
    
    num_windows = len(points) // window_size
    smoothed_points = []
    smoothed_timestamps = []
    
    for i in range(num_windows):
        start_idx = i * window_size
        end_idx = start_idx + window_size
        smoothed_points.append(np.mean(points[start_idx:end_idx], axis=0))
        smoothed_timestamps.append(timestamps[start_idx + window_size // 2])
    
    # Handle remaining points
    if len(points) % window_size != 0:
        start_idx = num_windows * window_size
        smoothed_points.append(np.mean(points[start_idx:], axis=0))
        smoothed_timestamps.append(timestamps[start_idx + (len(points) - start_idx) // 2])
    
    smoothed_trajectory = [(smoothed_points[i], smoothed_timestamps[i]) for i in range(len(smoothed_points))]
    
    return smoothed_trajectory


def main():
    """Main entry point."""

    file_path="data/LiDaR/871e1d886ffffff_cegl_m4_2.laz"
    output_path="data/geojson/car_trajectory_v3_pulse_closest_points.geojson"
    trajectory = recreate_trajectory(
        file_path=file_path,
        pulse_step=10
    )
    trajectory = moving_avg_smoothing(trajectory=trajectory,window_size=80)
    trajectory = window_avging(trajectory=trajectory,window_size=15)
    save_trajectory_geojson(trajectory, output_path, file_path)


if __name__ == "__main__":
    main()
