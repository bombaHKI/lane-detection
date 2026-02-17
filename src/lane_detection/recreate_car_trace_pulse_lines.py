"""Recreate car trajectory from LiDAR GPS timestamps."""

import numpy as np
from numpy.linalg import LinAlgError
from typing import Tuple, List, Optional, DefaultDict
import laspy
from collections import defaultdict
from timeit import default_timer as timer
from lane_detection.utils import save_trajectory_geojson, build_pulse_map

def are_perpendicular(m1, m2, tolerance_degrees=20.0):
    # 1. Calculate the angle of each line relative to the x-axis
    theta1 = np.arctan(m1)
    theta2 = np.arctan(m2)
    
    # 2. Calculate the angle difference in degrees
    # This gives us a value between 0 and 180
    angle_diff_deg = np.degrees(np.abs(theta1 - theta2))
    
    # 3. Check deviation from 90 degrees
    # We use abs(angle - 90) to handle both 85° and 95° cases equally
    deviation = np.abs(angle_diff_deg - 90)
    
    return deviation <= tolerance_degrees


def recreate_trajectory(
    file_path: str,
    initial_offset: float = 4.0,
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
    
    timestamps = las.gps_time
    
    print(f"Loaded {len(points)} points")
    print(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    print(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
    # Initialize
    min_timestamp = timestamps.min()
    current_time = min_timestamp + initial_offset
    max_timestamp = timestamps.max()

    start = timer()
    print(f"Mapping points to time values")
    time_to_points, unique_times = build_pulse_map(las.xyz, timestamps)
    print(f"Mapping done in {timer()-start} seconds")
    print(f"\nStarting trajectory reconstruction.")
    print(f"Initial time: {current_time:.3f}")
    print(f"Time step: {time_step} ms")
    
    trajectory = []
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            print(f"Progress: {progress:.2f}%")

        line1 = None
        line2 = None
        idx1 = np.searchsorted(unique_times, current_time)
        while idx1<len(unique_times):
            if len(time_to_points[unique_times[idx1]]) >= 120:
                try:
                    pulse_points = time_to_points[unique_times[idx1]]
                    x1 = pulse_points[:, 0]
                    y1 = pulse_points[:, 1]
                    z1 = pulse_points[:, 2]
                    line1 = np.polyfit(x1,y1,1)
                    break
                except LinAlgError as e:
                    pass
            idx1 += 1

        idx=idx1+1
        while idx<len(unique_times):
            if len(time_to_points[unique_times[idx]]) >= 120:
                if idx-idx>=20:
                    break
                try:
                    pulse_points = time_to_points[unique_times[idx]]
                    x2 = pulse_points[:, 0]
                    y2 = pulse_points[:, 1]
                    z2 = pulse_points[:, 2]
                    line2 = np.polyfit(x2,y2,1)
                    if are_perpendicular(line1[0],line2[0],10):
                        break
                except LinAlgError as e:
                    pass
            idx += 1
        
        if line1 is not None and line2 is not None:
            # Calculate x
            m1,b1=line1
            m2,b2=line2
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            trajectory.append(([x,y,np.mean(z1)],current_time))

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
        initial_offset=.5,
        time_step=.1,
        output_path="data/geojson/car_trajectory_v3_pulse_line_intersection.geojson"
    )


if __name__ == "__main__":
    main()
