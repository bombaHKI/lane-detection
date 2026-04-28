import numpy as np

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.logger import create_logger

logger = create_logger('Binning Stage')


class BinningStage(Stage):
    """Partition LAS points into spatially + temporally consistent bins along the car trace.

    The trace is resampled into virtual points every ``bin_length`` metres (arc length in XY).
    At each virtual point a 2D perpendicular line is implicitly defined via the local tangent.
    Bin ``i`` is the strip between perpendicular ``i`` and ``i + 1``. A LAS point belongs to
    bin ``i`` iff:
        * its gps_time is within ``[t_i - time_treshold, t_{i+1} + time_treshold]``
        * its (x, y) lies between the two perpendiculars
          (``(p - P_i) . t_i >= 0`` and ``(p - P_{i+1}) . t_{i+1} < 0``).

    ``context.bins`` is set to ``list[np.ndarray]`` of int64 indices into the LAS arrays.
    """

    def __init__(self, time_treshold: float = 10.0):
        self.time_treshold = float(time_treshold)

    def run(self, context):
        trace = context.trace
        las = context.las

        if trace is None or len(trace) < 2:
            logger.info("Trace has fewer than 2 points; no bins produced.")
            context.bins = []
            return

        # --- Resample trace by arc length (XY) -------------------------------------------
        trace_xy = np.asarray([p[0][:2] for p in trace], dtype=np.float64)
        trace_t = np.asarray([p[1] for p in trace], dtype=np.float64)

        seg = np.diff(trace_xy, axis=0)
        seg_len = np.linalg.norm(seg, axis=1)
        cum = np.concatenate(([0.0], np.cumsum(seg_len)))
        total_length = cum[-1]

        if total_length < context.bin_length:
            logger.info(
                f"Trace length {total_length:.2f} < bin_length {context.bin_length}; no bins.")
            context.bins = []
            return

        # Virtual (boundary) points every bin_length, including the final boundary.
        n_boundaries = int(np.floor(total_length / context.bin_length)) + 1
        s_vals = np.arange(n_boundaries, dtype=np.float64) * context.bin_length
        vp_x = np.interp(s_vals, cum, trace_xy[:, 0])
        vp_y = np.interp(s_vals, cum, trace_xy[:, 1])
        vp_t = np.interp(s_vals, cum, trace_t)
        vp = np.column_stack([vp_x, vp_y])  # (K, 2)

        # Tangent at each virtual point (central diff; endpoints one-sided).
        tang = np.empty_like(vp)
        if len(vp) >= 3:
            tang[1:-1] = vp[2:] - vp[:-2]
        tang[0] = vp[1] - vp[0]
        tang[-1] = vp[-1] - vp[-2]
        norms = np.linalg.norm(tang, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tang /= norms

        n_bins = len(vp) - 1
        logger.info(
            f"Resampled trace: length={total_length:.2f} m, "
            f"bin_length={context.bin_length} m, n_bins={n_bins}")

        # --- Pull LAS arrays once (avoid re-materialising ScaledArrayView per access) ----
        gps = np.asarray(las.gps_time, dtype=np.float64)
        xs = np.asarray(las.x, dtype=np.float64)
        ys = np.asarray(las.y, dtype=np.float64)

        # Sort by gps_time once, reuse for every bin's time-range query.
        order = np.argsort(gps, kind='stable')
        gps_sorted = gps[order]

        # --- Assign points to bins -------------------------------------------------------
        th = self.time_treshold
        bins: list[np.ndarray] = []
        log_interval = max(1, n_bins // 10)
        for i in range(n_bins):
            if i % log_interval == 0 or i == n_bins - 1:
                logger.info(f"Assigning points to bins: {i + 1}/{n_bins} ({(i + 1) / n_bins * 100:.0f}%)")
            t_lo = vp_t[i] - th
            t_hi = vp_t[i + 1] + th
            lo = np.searchsorted(gps_sorted, t_lo, side='left')
            hi = np.searchsorted(gps_sorted, t_hi, side='right')
            if hi <= lo:
                bins.append(np.empty(0, dtype=np.int64))
                continue

            cand = order[lo:hi]
            # Signed distances to the two perpendicular boundaries.
            dx0 = xs[cand] - vp[i, 0]
            dy0 = ys[cand] - vp[i, 1]
            d0 = dx0 * tang[i, 0] + dy0 * tang[i, 1]

            dx1 = xs[cand] - vp[i + 1, 0]
            dy1 = ys[cand] - vp[i + 1, 1]
            d1 = dx1 * tang[i + 1, 0] + dy1 * tang[i + 1, 1]

            mask = (d0 >= 0.0) & (d1 < 0.0)
            bins.append(cand[mask])

        total_assigned = sum(b.size for b in bins)
        logger.info(
            f"Assigned {total_assigned} point-bin memberships across {n_bins} bins "
            f"(avg {total_assigned / max(n_bins, 1):.0f} per bin).")

        context.bins = bins
