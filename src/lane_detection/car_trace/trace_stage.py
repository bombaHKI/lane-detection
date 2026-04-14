from typing import Tuple, List, Optional, DefaultDict
from timeit import default_timer as timer
import numpy as np
from numpy.linalg import LinAlgError
import open3d as o3d

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

    time_to_points, unique_times = build_pulse_map(xyz, timestamps)
    
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
        logger.info("Warning: No trajectory points generated!")
    
    return trajectory

def windows_median_v2(
    point_cloud,
    initial_offset: float = 4.0,
    time_window: float = .1,
    time_step: float = 0.5,
    offset_from_ground: float = 2,
    radius_around_1st_guess: float = 5
) -> List[Tuple[np.ndarray, float]]:
    """
    Improvement to `wondow_median`: detect ground at 1st esimation, 
    then consider points below a treshold
    """

    def estimated_pos(points: np.ndarray, offset_from_ground: int = 2, segment_radius: int = 5) -> Optional[np.ndarray]:
        """
        Calculate estimated position from given points.
        Implementation: 
        1. 1st estimation: median of xyz coordinates.
        2. Ground detection around point.
        3. Median of all points that are below ground+offset.
        
        Args:
            points: Nx3 array of XYZ coordinates
            offset_from_ground: Points above ground level + offset will not be considered in the estimation
            segment_radius: segment ground in the radius of the first estimation
            
        Returns:
            Median location [x, y, z] or None if no points
        """
        if len(points) == 0:
            return None
        estimation1 =  np.median(points, axis=0)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[np.linalg.norm(points[:,:2]-estimation1[:2],axis=1)<segment_radius])
        max_iterations = 5
        valid_plane_found = False    
        for attempt in range(max_iterations):
            if (len(pcd.points) < 50):
                logger.info(f"Too few points! x: {estimation1[0]}, y: {estimation1[1]}")
                return None
            plane_model, inliers = pcd.segment_plane(
                distance_threshold=0.05,
                ransac_n=3,
                num_iterations=100
            )

            a, b, c, d = plane_model
            
            # Calculate gradient (slope) from plane normal
            # Gradient = sqrt(a^2 + b^2) / |c|
            # 40% gradient = 0.4 = tan(angle) ≈ 21.8 degrees
            horizontal_component = np.sqrt(a**2 + b**2)
            vertical_component = np.abs(c)
            
            if vertical_component < 1e-6:  # nearly vertical plane
                gradient = float('inf')
            else:
                gradient = horizontal_component / vertical_component
            
            # Check if gradient is less than 40% (0.4)
            if gradient < 0.4:
                valid_plane_found = True
                break
            
            # Remove inliers and try again with remaining points
            if len(inliers) > 0:
                remaining_mask = np.ones(len(pcd.points), dtype=bool)
                remaining_mask[inliers] = False
                remaining_points = np.asarray(pcd.points)[remaining_mask]
                
                if len(remaining_points) < 3:
                    break
                    
                pcd.points = o3d.utility.Vector3dVector(remaining_points)
            else:
                break
        
        if not valid_plane_found:
            logger.info(f"No ground segmented")
            return estimation1

        x,y = estimation1[:2]
        ground_height = -(a*x+b*y+d)/c
        estimation2 =  np.median(points[points[:,2]<=ground_height+offset_from_ground], axis=0)
        estimation2[2] = ground_height
        return estimation2

    points = point_cloud.xyz  # Nx3 array
    timestamps = point_cloud.gps_time  # GPS timestamps

    
    logger.info(f"Loaded {len(points)} points")
    logger.info(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    logger.info(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
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

    time_to_points, unique_times = build_pulse_map(points, timestamps)
    
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
            avg_location = estimated_pos(
                filtered_points,
                offset_from_ground=offset_from_ground,
                segment_radius=radius_around_1st_guess
            )
            if avg_location is not None:
                trajectory.append((avg_location, current_time))
        
        # Increment time
        current_time += time_step
    
    logger.info(f"\nTrajectory reconstruction complete! Took: {(timer()-start):.2f} seconds")
    logger.info(f"Total trajectory points: {len(trajectory)}")
    
    if len(trajectory) == 0:
        logger.info("Warning: No trajectory points generated!")
        return trajectory
    
    return trajectory

def pulse_lines(
    point_cloud,
    initial_offset: float = 4.0,
    time_step: float = 0.5
) -> List[Tuple[np.ndarray, float]]:
    """
    Recreate car trajectory from LiDAR GPS timestamps.
    
    Args:
        initial_offset: Seconds after minimum timestamp to start (default: 4)
        time_step: Seconds to increment time each iteration (default: .5)
        
    Returns:
        List of (location, timestamp) tuples representing the trajectory
    """

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
    points = point_cloud.xyz
    timestamps = point_cloud.gps_time
    
    logger.info(f"Loaded {len(points)} points")
    logger.info(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    logger.info(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
    # Initialize
    min_timestamp = timestamps.min()
    current_time = min_timestamp + initial_offset
    max_timestamp = timestamps.max()

    start = timer()
    time_to_points, unique_times = build_pulse_map(points, timestamps)
    logger.info(f"\nStarting trajectory reconstruction.")
    logger.info(f"Initial time: {current_time:.3f}")
    logger.info(f"Time step: {time_step} ms")
    
    trajectory = []
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            logger.info(f"Progress: {progress:.2f}%")

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
    
    logger.info(f"\nTrajectory reconstruction complete! Took: {(timer()-start):.2f} seconds")
    logger.info(f"Total trajectory points: {len(trajectory)}")
    
    if len(trajectory) == 0:
        logger.info("Warning: No trajectory points generated!")
        return trajectory
    
    return trajectory

def closest_point(
    point_cloud,
    pulse_step: int = 5,
    window_size: int = 1,

) -> List[Tuple[np.ndarray, float]]:
    """
    Recreate car trajectory from LiDAR GPS timestamps.

    Adds the next pulse's closest point (to the current pos) to the trajectory.
    The next pulse is pulse_step pulses.
    
    Args:
        pulse_step: Distance between two consecutive pulses
        
    Returns:
        List of (location, timestamp) tuples representing the trajectory
    """
    points = point_cloud.xyz
    timestamps = point_cloud.gps_time
    
    print(f"Loaded {len(points)} points")
    print(f"Timestamp range: {timestamps.min():.3f} to {timestamps.max():.3f}")
    print(f"Duration: {(timestamps.max() - timestamps.min()):.3f} time units")
    
    # Initialize
    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    start = timer()

    time_to_points, unique_times = build_pulse_map(points, timestamps)

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

    if window_size > 1:
        trajectory = moving_avg_smoothing(trajectory=trajectory, window_size=window_size)

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
    start = timer()
    logger.info(f"Mapping points to time values")

    sort_idx = np.argsort(gps_times)
    
    sorted_points = points[sort_idx]
    sorted_times = gps_times[sort_idx]
    
    unique_times, start_indices = np.unique(sorted_times, return_index=True)
    
    grouped_points = np.split(sorted_points, start_indices[1:])
    
    logger.info(f"Mapping done in {timer()-start} seconds")
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
