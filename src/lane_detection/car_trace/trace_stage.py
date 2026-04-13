from typing import Tuple, List, Optional, DefaultDict
import numpy as np
from timeit import default_timer as timer

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Car Trace Stage')

class CarTraceStage(Stage):
    def __init__(self, algorithm):
        self.algorithm = algorithm

    def run(self, context):
        context.trace = self.algorithm(context.las)

def window_median(
    point_cloud,
    initial_offset: float = 4.0,
    time_window: float = .1,
    time_step: float = 0.5,) -> List[Tuple[np.ndarray, float]]:
    """
    Recreate car trajectory from LiDAR GPS timestamps.
    
    Args:
        point_cloud: laspy.lasdata.LasData 
        initial_offset: Seconds after minimum timestamp to start (default: 4)
        time_window: Time window in milliseconds for point selection (±window, default: .1)
        time_step: Seconds to increment time each iteration (default: .5)
        epsilon: Maximum distance in meters for spatial filtering (default: 10)
        
    Returns:
        List of (location, timestamp) tuples representing the trajectory
    """

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
    
    xyz = point_cloud.xyz
    timestamps = point_cloud.gps_time
    
    # Initialize
    min_timestamp = timestamps.min()
    current_time = min_timestamp + initial_offset
    max_timestamp = timestamps.max()
    
    trajectory = []
    
    start = timer()
    logger.info(f"\nStarting trajectory reconstruction.")
    logger.info(f"Initial time: {current_time:.3f}")
    logger.info(f"Time window: ±{time_window} ms")
    logger.info(f"Time step: {time_step} ms")

    logger.info(f"Mapping points to time values")
    time_to_points, unique_times = build_pulse_map(xyz, timestamps)
    logger.info(f"Mapping done in {timer()-start} seconds")
    
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            logger.info(f"Progress: {curr_progress:.2f}%")

        # Find points within time window and spatial constraints
        filtered_points, filtered_times = find_points_in_time_window(
            time_to_points, timestamps, unique_times, current_time, time_window
        )
        


        if len(filtered_points) > 0:
            # Calculate average location
            pos = estimated_pos(filtered_points)
            trajectory.append((pos, current_time))
        
        # Increment time
        current_time += time_step
    
    logger.info(f"\nTrajectory reconstruction complete! Took: {(timer()-start):.2f} seconds")
    logger.info(f"Total trajectory points: {len(trajectory)}")
    
    if len(trajectory) == 0:
        print("Warning: No trajectory points generated!")
    
    return trajectory

def build_pulse_map(points: np.ndarray, gps_times: list):
    """
    Builds a timestamp -> ndarray of points map
    
    Args:
        :param points: The xyz coordinates of the lidar data
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

def find_points_in_time_window(
    time_to_points: DefaultDict[float, List[np.ndarray]],
    timestamps: np.ndarray,
    unique_times: np.ndarray,
    target_time: float,
    time_window: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Find points within time window.
    
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
