import numpy as np

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Trace Distance Filter Stage')

_SEGMENT_MARGIN = 5  # number of extra trace segments to check on each side of a bin


class TraceDistanceFilterStage(Stage):
    """Filter each bin's points to those within *distance* metres of the car
    trace (2-D XY).

    Each bin is compared only against the nearby trace segments (determined by
    the bin's position along the trace), keeping memory usage small.

    Reads  ``context.bins``, ``context.las``, ``context.trace``.
    Writes ``context.bins`` with filtered index arrays.
    """

    def __init__(self, distance: float = 21.0):
        self.distance = float(distance)

    def _segment_distances(self, xs: np.ndarray, ys: np.ndarray, seg_start: np.ndarray, seg_vec: np.ndarray, seg_len2: np.ndarray) -> np.ndarray:
        """Minimum distance from each point to the nearest of the given segments.
        xs/ys: (N,), seg_*: (K, 2) / (K,)
        """
        pts = np.column_stack([xs, ys])                              # (N, 2)
        diff    = pts[:, None, :] - seg_start[None, :, :]            # (N, K, 2)
        t       = (diff * seg_vec[None, :, :]).sum(axis=2) / seg_len2  # (N, K)
        t       = np.clip(t, 0.0, 1.0)
        closest = seg_start[None, :, :] + t[:, :, None] * seg_vec[None, :, :]  # (N, K, 2)
        dist2   = ((pts[:, None, :] - closest) ** 2).sum(axis=2)     # (N, K)
        return np.sqrt(dist2.min(axis=1))                            # (N,)

    def run(self, context):
        logger.info(f"Starting distance clipping with: {self.distance} meters.")
        if not context.bins:
            logger.info("No bins to filter.")
            return

        trace = context.trace
        if trace is None or len(trace) < 2:
            logger.info("Trace unavailable; skipping distance filter.")
            return

        polyline  = np.asarray([p[0][:2] for p in trace], dtype=np.float64)
        trace_t   = np.asarray([p[1]       for p in trace], dtype=np.float64)

        seg_start = polyline[:-1]
        seg_vec   = polyline[1:] - seg_start
        seg_len2  = (seg_vec ** 2).sum(axis=1)
        seg_len2  = np.where(seg_len2 == 0, 1.0, seg_len2)
        n_segs    = len(seg_start)

        las = context.las
        xs = np.asarray(las.x,        dtype=np.float64)
        ys = np.asarray(las.y,        dtype=np.float64)
        gps = np.asarray(las.gps_time, dtype=np.float64)

        total_before = total_after = 0
        filtered: list[np.ndarray] = []
        n_bins = len(context.bins)
        log_interval = max(1, n_bins // 10)

        for bin_idx, indices in enumerate(context.bins):
            if bin_idx % log_interval == 0 or bin_idx == n_bins - 1:
                logger.info(f"Filtering bins: {bin_idx + 1}/{n_bins} ({(bin_idx + 1) / n_bins * 100:.0f}%)")
            total_before += indices.size
            if indices.size == 0:
                filtered.append(indices)
                continue

            # Find the trace segment range that covers this bin's gps_time span
            t_lo = gps[indices].min()
            t_hi = gps[indices].max()
            seg_lo = max(0,      np.searchsorted(trace_t, t_lo, side='left')  - 1 - _SEGMENT_MARGIN)
            seg_hi = min(n_segs, np.searchsorted(trace_t, t_hi, side='right') + 1 + _SEGMENT_MARGIN)

            dists = self._segment_distances(
                xs[indices], ys[indices],
                seg_start[seg_lo:seg_hi],
                seg_vec[seg_lo:seg_hi],
                seg_len2[seg_lo:seg_hi],
            )
            keep = indices[dists <= self.distance]
            filtered.append(keep)
            total_after += keep.size

        logger.info(
            f"Trace distance filter (d<={self.distance} m): "
            f"{total_before} -> {total_after} points across {len(filtered)} bins."
        )
        context.bins = filtered

        global_mask = np.zeros(len(xs), dtype=bool)
        for indices in filtered:
            global_mask[indices] = True
        context.global_mask = global_mask

