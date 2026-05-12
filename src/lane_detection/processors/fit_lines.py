from __future__ import annotations

import numpy as np
import shapely
from shapely.ops import linemerge
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from lane_detection.pipeline.pipeline import Stage, Context


class FitLinesStage(Stage):
    """
    Fits quadratic lane-lines per window.

    For each window the points are rotated so that the car trajectory is aligned
    with the x-axis.  The stage then iteratively:
      1. Fits a quadratic curve (y = ax² + bx + c) with RANSAC to the remaining
         transformed (x, y) points.
      2. Accepts the fit when ALL conditions hold:
           - inlier count  >= min_inliers
           - x-span of inliers >= min_span
           - |a|  <= max_curvature   (curvature limit)
      3. Removes inliers from the pool and adds the sampled curve to the window
         line list.
      4. Stops when n_lines have been collected, or when too few points remain,
         or when the latest fit is rejected.

    Fitted curves are transformed back to original (x, y) coordinates and
    appended to context.lines as shapely.LineString objects.
    """

    def __init__(
        self,
        n_lines: int = 4,
        max_curvature: float = 0.01,
        min_span: float = 10.0,
        residual_threshold: float = 0.3,
        min_inliers: int = 10,
        max_trials: int = 130,
        sample_points: int = 100,
    ):
        """
        Parameters
        ----------
        n_lines : int
            Maximum number of lines to fit per window.
        max_curvature : float
            Maximum allowed absolute value of the quadratic coefficient (|a|).
        min_span : float
            Minimum required x-span (in trace-aligned units) of the inliers.
        residual_threshold : float
            RANSAC residual threshold.
        min_inliers : int
            Minimum inlier count to accept a fit and to continue iterating.
        max_trials : int
            Maximum RANSAC trials per fit.
        sample_points : int
            Number of points sampled along the fitted curve for the LineString.
        """
        super().__init__()
        self.n_lines = n_lines
        self.max_curvature = max_curvature
        self.min_span = min_span
        self.residual_threshold = residual_threshold
        self.min_inliers = min_inliers
        self.max_trials = max_trials
        self.sample_points = sample_points

    # ------------------------------------------------------------------
    # Coordinate transform helpers
    # ------------------------------------------------------------------

    def _get_transform(self, window_indices: np.ndarray, context: Context):
        """Build a 2-D rotation that aligns the local car trace with the x-axis.

        Returns
        -------
        (R, origin) or None
            R      : (2, 2) orthogonal matrix whose rows are the new basis vectors
                     [u_along_trace, u_perpendicular].
            origin : (2,) translation (first trace point clipped to window hull).

        Transform conventions
        ---------------------
        Forward  : pts_t   = (pts_xy - origin) @ R.T
        Inverse  : pts_xy  = pts_t  @ R  + origin
        """
        las = context.las
        xy = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[window_indices],
            np.asarray(las.y, dtype=np.float64)[window_indices],
        ])
        if len(xy) < 2:
            return None

        # Build convex hull for trace clipping; buffer handles collinear cases
        hull = shapely.MultiPoint(xy).convex_hull.buffer(1.0)

        gps_time = np.asarray(las.gps_time, dtype=np.float64)
        min_t = float(gps_time[window_indices].min())
        max_t = float(gps_time[window_indices].max())

        trace_pts = [loc[:2] for loc, t in context.trace if min_t <= t <= max_t]
        if len(trace_pts) < 2:
            return None

        trace_raw = shapely.LineString(trace_pts).intersection(hull)
        if trace_raw.is_empty:
            return None
        if trace_raw.geom_type == "MultiLineString":
            trace = linemerge(trace_raw)
            if trace.geom_type == "MultiLineString":
                trace = max(trace.geoms, key=lambda g: g.length)
        else:
            trace = trace_raw

        if trace.is_empty or len(trace.coords) < 2:
            return None

        S = np.array(trace.coords[0], dtype=np.float64)
        E = np.array(trace.coords[-1], dtype=np.float64)
        v = E - S
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return None

        u = v / norm
        u_perp = np.array([-u[1], u[0]])
        R = np.stack([u, u_perp], axis=0)  # rows are basis vectors → orthogonal
        return R, S

    # ------------------------------------------------------------------
    # RANSAC quadratic fit
    # ------------------------------------------------------------------

    def _fit_one(self, x: np.ndarray, y: np.ndarray):
        """Fit y = ax² + bx + c with RANSAC.

        Returns
        -------
        (fitted_pipeline, inlier_mask) or None
        """
        if len(x) < self.min_inliers:
            return None

        estimator = make_pipeline(
            PolynomialFeatures(degree=2, include_bias=False),
            LinearRegression(),
        )
        ransac = RANSACRegressor(
            estimator=estimator,
            min_samples=3,
            residual_threshold=self.residual_threshold,
            max_trials=self.max_trials,
        )
        try:
            ransac.fit(x.reshape(-1, 1), y)
        except Exception:
            return None

        return ransac.estimator_, ransac.inlier_mask_

    # ------------------------------------------------------------------
    # Per-window processing
    # ------------------------------------------------------------------

    def _process_window(
        self, window_indices: np.ndarray, context: Context
    ) -> list[shapely.LineString]:
        result = self._get_transform(window_indices, context)
        if result is None:
            return []
        R, origin = result

        las = context.las
        xy = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[window_indices],
            np.asarray(las.y, dtype=np.float64)[window_indices],
        ])
        # Forward transform: rotate so trace = x-axis
        xy_t = (xy - origin) @ R.T  # (N, 2)  col0=along-trace, col1=perp

        remaining = np.ones(len(window_indices), dtype=bool)
        window_lines: list[shapely.LineString] = []

        while len(window_lines) < self.n_lines:
            rem_idx = np.nonzero(remaining)[0]
            if len(rem_idx) < self.min_inliers:
                break

            x_rem = xy_t[rem_idx, 0]
            y_rem = xy_t[rem_idx, 1]

            fit = self._fit_one(x_rem, y_rem)
            if fit is None:
                break

            model, inlier_local = fit

            # --- acceptance checks ---
            n_inliers = int(inlier_local.sum())
            if n_inliers < self.min_inliers:
                break

            x_inliers = x_rem[inlier_local]
            span = float(x_inliers.max() - x_inliers.min())
            if span < self.min_span:
                break

            # coef_ layout: [x-coef, x²-coef]  (include_bias=False, degree=2)
            a = float(model.named_steps["linearregression"].coef_[1])
            if abs(a) > self.max_curvature:
                break

            # --- build LineString in original coordinates ---
            x_sample = np.linspace(x_inliers.min(), x_inliers.max(), self.sample_points)
            y_sample = model.predict(x_sample.reshape(-1, 1))
            pts_t = np.column_stack([x_sample, y_sample])
            pts_orig = pts_t @ R + origin  # inverse transform

            window_lines.append(shapely.LineString(pts_orig))

            # Remove inliers before next iteration
            remaining[rem_idx[inlier_local]] = False

        return window_lines

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def run(self, context: Context):
        bins = context.bins
        window_size = context.window_size
        window_shift = context.window_shift

        windows: list[np.ndarray] = []
        start = 0
        while start + window_size <= len(bins):
            combined = np.concatenate(bins[start : start + window_size])
            windows.append(np.unique(combined))
            start += window_shift

        n_windows = len(windows)
        self.logger.info(
            f"FitLines: {n_windows} windows "
            f"(size={window_size}, shift={window_shift}, n_lines={self.n_lines})."
        )

        if context.lines is None:
            context.lines = []

        total_lines = 0
        log_step = max(1, n_windows // 10)
        for i, window_indices in enumerate(windows):
            if i % log_step == 0 or i == n_windows - 1:
                self.logger.info(
                    f"Window {i + 1}/{n_windows} ({(i + 1) / n_windows * 100:.0f}%)"
                )
            if window_indices.size == 0:
                continue
            lines = self._process_window(window_indices, context)
            context.lines.append(lines)
            total_lines += len(lines)

        self.logger.info(f"FitLines done: {total_lines} lines across {n_windows} windows.")
