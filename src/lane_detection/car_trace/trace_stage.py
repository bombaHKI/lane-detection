from __future__ import annotations

from typing import Tuple, List, Optional, DefaultDict, Callable, Dict
from timeit import default_timer as timer
from pathlib import Path

import numpy as np
from numpy.linalg import LinAlgError
from sklearn.linear_model import RANSACRegressor

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Car Trace Stage')


__all__ = ['CarTraceStage']


# --------------------------------------------------------------------------- #
# Plane helpers
# --------------------------------------------------------------------------- #

def _fit_plane_ransac(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_trials: int = 100,
) -> Optional[Tuple[float, float, float, np.ndarray]]:
    """Fit ``z = a*x + b*y + c`` with RANSAC. Returns ``(a, b, c, inlier_mask)`` or None."""
    if len(points) < 3:
        return None
    try:
        model = RANSACRegressor(
            residual_threshold=distance_threshold,
            min_samples=3,
            max_trials=max_trials,
        ).fit(points[:, :2], points[:, 2])
    except ValueError:
        return None
    a, b = model.estimator_.coef_
    c = model.estimator_.intercept_
    return float(a), float(b), float(c), model.inlier_mask_


def _fit_ground_plane(
    points: np.ndarray,
    distance_threshold: float = 0.05,
    max_gradient: float = 0.4,
    max_iterations: int = 5,
) -> Optional[Tuple[float, float, float]]:
    """Fit a near-horizontal plane; retry on inliers if gradient exceeds ``max_gradient``."""
    remaining = points
    for _ in range(max_iterations):
        res = _fit_plane_ransac(remaining, distance_threshold)
        if res is None:
            return None
        a, b, c, inliers = res
        if np.hypot(a, b) < max_gradient:
            return a, b, c
        remaining = remaining[~inliers]
        if len(remaining) < 3:
            return None
    return None


# --------------------------------------------------------------------------- #
# Algorithm: window_median
# --------------------------------------------------------------------------- #

def _window_median(
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

    time_to_points, unique_times = _build_pulse_map(xyz, timestamps)
    
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            logger.info(f"Progress: {curr_progress:.2f}%")

        # Find points within time window and spatial constraints
        filtered_points, filtered_times = _find_points_in_time_window(
            time_to_points, timestamps, unique_times, current_time, time_window
        )

        if len(filtered_points) > 0:
            # Calculate average location
            pos = estimated_pos(filtered_points)
            trajectory.append((pos, current_time))
        
        # Increment time
        current_time += time_step
    
    return trajectory

# --------------------------------------------------------------------------- #
# Algorithm: window_median_v2
# --------------------------------------------------------------------------- #

def _window_median_v2(
    point_cloud,
    initial_offset: float = 4.0,
    time_window: float = .1,
    time_step: float = 0.6,
    offset_from_ground: float = 2,
    radius_around_1st_guess: float = 5
) -> List[Tuple[np.ndarray, float]]:
    """
    Improvement to `wondow_median`: detect ground at 1st esimation, 
    then consider points below a treshold
    """

    def estimated_pos(
        points: np.ndarray,
        offset_from_ground: float = 2.0,
        segment_radius: float = 5.0,
    ) -> Optional[np.ndarray]:
        """Estimate position via:

        1. Initial guess = median XYZ.
        2. RANSAC ground plane fit on points within ``segment_radius`` of the guess.
        3. Final position = median of points within ``ground_height + offset_from_ground``.
        """
        if len(points) == 0:
            return None

        guess = np.median(points, axis=0)
        nearby = points[np.linalg.norm(points[:, :2] - guess[:2], axis=1) < segment_radius]
        if len(nearby) < 50:
            logger.info(f"Too few points! x: {guess[0]}, y: {guess[1]}")
            return None

        plane = _fit_ground_plane(nearby, distance_threshold=0.05, max_gradient=0.4)
        if plane is None:
            logger.info("No ground segmented")
            return guess

        a, b, c = plane
        ground_height = a * guess[0] + b * guess[1] + c
        below = points[points[:, 2] <= ground_height + offset_from_ground]
        result = np.median(below, axis=0)
        result[2] = ground_height
        return result

    points = point_cloud.xyz  # Nx3 array
    timestamps = point_cloud.gps_time  # GPS timestamps
    
    # Initialize
    min_timestamp = timestamps.min()
    current_time = min_timestamp + initial_offset
    max_timestamp = timestamps.max()
    
    trajectory = []

    time_to_points, unique_times = _build_pulse_map(points, timestamps)
    
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time-min_timestamp)/(max_timestamp-min_timestamp)*100
        if curr_progress >= progress+5:
            progress+=5
            logger.info(f"Progress: {curr_progress:.2f}%")

        # Find points within time window and spatial constraints
        filtered_points, filtered_times = _find_points_in_time_window(
            time_to_points, timestamps, unique_times, current_time, time_window
        )

        if len(filtered_points) > 0:
            # Calculate average location
            pos = estimated_pos(
                filtered_points,
                offset_from_ground=offset_from_ground,
                segment_radius=radius_around_1st_guess
            )
            if pos is not None:
                trajectory.append((pos, current_time))
        
        # Increment time
        current_time += time_step
    
    return trajectory

# --------------------------------------------------------------------------- #
# Algorithm: pulse_lines
# --------------------------------------------------------------------------- #

def _pulse_lines(
    point_cloud,
    initial_offset: float = 4.0,
    time_step: float = 0.5,
    min_pulse_size: int = 70,
    max_pulse_search: int = 70,
    perpendicular_tolerance_deg: float = 10.0,
) -> List[Tuple[np.ndarray, float]]:
    """Estimate the car position by intersecting two near-perpendicular pulse lines.

    For each sample time:
      1. Walk forward pulse-by-pulse from ``current_time`` until a pulse with
         enough points to fit a line is found.
      2. Continue walking (up to ``max_pulse_search`` pulses) until another
         pulse line is found that is roughly perpendicular to the first.
      3. Record the intersection of the two lines as the estimated position.

    After each estimate the sample time advances by ``time_step`` seconds,
    independent of how many pulses were consumed.
    """

    def are_perpendicular(m1: float, m2: float, tolerance_deg: float) -> bool:
        angle_diff_deg = np.degrees(np.abs(np.arctan(m1) - np.arctan(m2)))
        return np.abs(angle_diff_deg - 90) <= tolerance_deg

    def fit_pulse_line(
        start_idx: int, end_idx: int
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], int]:
        """Find the next pulse in ``[start_idx, end_idx)`` with enough points to fit a line.

        Returns ``(line, pulse_xyz, idx)`` where ``line = [m, b]`` in ``y = m*x + b``.
        The fit axis is chosen by dominant spread to stay stable for near-vertical pulses.
        """
        idx = start_idx
        while idx < end_idx:
            pulse = time_to_points[unique_times[idx]]
            if len(pulse) >= min_pulse_size:
                try:
                    dx = np.ptp(pulse[:, 0])
                    dy = np.ptp(pulse[:, 1])
                    if dx >= dy:
                        line = np.polyfit(pulse[:, 0], pulse[:, 1], 1)
                    else:
                        m_inv, b_inv = np.polyfit(pulse[:, 1], pulse[:, 0], 1)
                        if m_inv == 0:
                            raise LinAlgError("vertical pulse")
                        line = np.array([1.0 / m_inv, -b_inv / m_inv])
                    return line, pulse, idx
                except (LinAlgError, ValueError):
                    pass
            idx += 1
        return None, None, idx

    points = point_cloud.xyz
    timestamps = point_cloud.gps_time

    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    time_to_points, unique_times = _build_pulse_map(points, timestamps)
    n = len(unique_times)

    trajectory: List[Tuple[np.ndarray, float]] = []
    current_time = min_timestamp + initial_offset
    progress = 0
    while current_time <= max_timestamp:
        curr_progress = (current_time - min_timestamp) / (max_timestamp - min_timestamp) * 100
        if curr_progress >= progress + 5:
            progress += 5
            logger.info(f"Progress: {progress:.2f}%")

        start_idx = int(np.searchsorted(unique_times, current_time))

        line1, pulse1, idx1 = fit_pulse_line(start_idx, n)
        if line1 is None:
            break

        line2 = None
        pulse2 = None
        idx2 = idx1 + 1
        search_end = min(idx1 + 1 + max_pulse_search, n)
        while idx2 < search_end:
            candidate, cand_pulse, idx2 = fit_pulse_line(idx2, search_end)
            if candidate is None or not are_perpendicular(line1[0], candidate[0], perpendicular_tolerance_deg):
                idx2 += 1
                continue
            line2 = candidate
            pulse2 = cand_pulse
            break

        if line2 is not None and pulse2 is not None:
            m1, b1 = line1
            m2, b2 = line2
            x = (b2 - b1) / (m1 - m2)
            y = m1 * x + b1
            z = float(0.5 * (np.mean(pulse1[:, 2]) + np.mean(pulse2[:, 2])))
            trajectory.append((np.array([x, y, z]), current_time))

        current_time += time_step

    return trajectory

# --------------------------------------------------------------------------- #
# Algorithm: closest_point
# --------------------------------------------------------------------------- #

def _moving_avg_smoothing(
    trajectory: List[Tuple[np.ndarray, float]],
    window_size: int = 5,
) -> List[Tuple[np.ndarray, float]]:
    """Centered moving-average smoothing over the XYZ positions of a trajectory."""
    if len(trajectory) < window_size:
        return trajectory

    points = np.array([pos for pos, _ in trajectory])
    timestamps = np.array([ts for _, ts in trajectory])

    smoothed = np.zeros_like(points)
    half = window_size // 2
    for i in range(len(points)):
        smoothed[i] = np.mean(points[max(0, i - half): i + half + 1], axis=0)

    return [(smoothed[i], timestamps[i]) for i in range(len(trajectory))]


def _closest_point(
    point_cloud,
    pulse_step: int = 5,
    window_size: int = 50,
) -> List[Tuple[np.ndarray, float]]:
    """Walk pulse-by-pulse, appending the point closest to the previous estimate.

    Args:
        pulse_step: Step (in pulses) between consecutive trajectory samples.
        window_size: Optional moving-average window applied to the result.
    """
    points = point_cloud.xyz
    timestamps = point_cloud.gps_time

    min_timestamp = timestamps.min()
    max_timestamp = timestamps.max()

    time_to_points, unique_times = _build_pulse_map(points, timestamps)

    trajectory: List[Tuple[np.ndarray, float]] = []
    pos_estimation = time_to_points[min_timestamp][0]
    progress = 0
    for curr_time in unique_times[::pulse_step]:
        curr_progress = (curr_time - min_timestamp) / (max_timestamp - min_timestamp) * 100
        if curr_progress >= progress + 5:
            progress += 5
            logger.info(f"Progress: {progress:.2f}%")

        curr_pulse = time_to_points[curr_time]
        distances = np.linalg.norm(curr_pulse[:, :2] - pos_estimation[:2], axis=1)
        pos_estimation = curr_pulse[int(np.argmin(distances))]
        trajectory.append((pos_estimation.copy(), curr_time))

    if window_size > 1:
        trajectory = _moving_avg_smoothing(trajectory, window_size=window_size)

    return trajectory

# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #

def _build_pulse_map(points: np.ndarray, gps_times: list):
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

def _find_points_in_time_window(
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


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

_METHODS: Dict[str, Callable] = {
    'window_median': _window_median,
    'window_median_v2': _window_median_v2,
    'pulse_lines': _pulse_lines,
    'closest_point': _closest_point,
}


class CarTraceStage(Stage):
    """Car-trajectory reconstruction as a pipeline stage.

    Four algorithms are supported via the ``method`` parameter:

    * ``'window_median'``    — slide a time window over the GPS timestamps and
                            take the median XYZ of points inside it.
    * ``'window_median_v2'`` — same as ``'window_median'`` but discards points
                            above the locally segmented ground plane + threshold before
                            taking the median.
    * ``'pulse_lines'``      — fit lines to successive LiDAR pulses and use the
                            intersection of two near-perpendicular pulses.
    * ``'closest_point'``    — walk pulse-by-pulse, appending the point closest
                            to the previous position estimate. Optional
                            moving-average smoothing.

    The stage loads a previously saved trace from ``load_path`` if given,
    otherwise runs the selected algorithm on ``context.las`` and (if
    ``context.output_dir`` is set) caches the result to ``trace.npz``.
    """
    def __init__(
        self,
        method: str = 'window_median_v2',
        load_path: Optional[str] = None,
        **kwargs,
    ):
        if method not in _METHODS:
            raise ValueError(
                f"Unknown car-trace method: {method!r}. "
                f"Expected one of {sorted(_METHODS)}."
            )
        self.method = method
        self.load_path = Path(load_path) if load_path else None
        self.kwargs = kwargs

    def run(self, context):
        if self.load_path is not None:
            try:
                logger.info(f"Loading trace from: {self.load_path}")
                data = np.load(self.load_path)
                context.trace = [(xyz, t) for xyz, t in zip(data['positions'], data['timestamps'])]
                logger.info(f"Loaded {len(context.trace)} trajectory points.")
                return
            except Exception as e:
                logger.info(f"Failed to load trace: {e}. Computing instead.")

        algorithm = _METHODS[self.method]
        logger.info(f"Starting trajectory reconstruction: method={self.method}")
        start = timer()
        context.trace = algorithm(context.las, **self.kwargs)

        logger.info(f"Trajectory reconstruction complete! Took: {(timer() - start):.2f} seconds")
        logger.info(f"Total trajectory points: {len(context.trace)}")

        if context.output_dir is not None and len(context.trace) > 0:
            path = context.output_dir / f"trace_{self.method}.npz"
            positions = np.array([p[0] for p in context.trace])
            timestamps = np.array([p[1] for p in context.trace])
            np.savez(path, positions=positions, timestamps=timestamps)
            logger.info(f"Trace saved to: {path}")

        if len(context.trace) == 0:
            logger.info("Warning: No trajectory points generated!")
