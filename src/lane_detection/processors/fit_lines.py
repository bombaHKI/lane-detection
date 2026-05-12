from __future__ import annotations

import numpy as np
import shapely
from shapely.ops import linemerge
from sklearn.cluster import DBSCAN
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

from lane_detection.pipeline.pipeline import Stage, Context


class FitLinesStage(Stage):
    """Incremental lane-line tracker using DBSCAN clustering and cubic RANSAC.

    context.lines is initialised to [] at the start of run() and holds one
    shapely.LineString per tracked lane divider in world coordinates.

    Per-window algorithm
    --------------------
    1.  Rotate points so the local car trajectory is the x-axis.
    2.  Compute the trajectory curvature in the rotated frame (quadratic fit →
        second derivative = 2·a₂).  Accepted fits must satisfy
        |κ_fit − κ_traj| ≤ curvature_limit.
    3.  Scale the along-trace (x) coordinate by *x_scale*, then run DBSCAN.
    4.  Loop:
        a.  Find the (cluster, existing_line) pair with the smallest mean
            distance (similarity).
        b.  If best_distance ≤ similarity_threshold:
              - Fit cubic RANSAC on the cluster.
              - Accept if span and curvature conditions hold.
              - Merge the fitted segment into the matching existing line.
              - Remove the cluster and repeat.
        c.  Else (no cluster is close enough to any existing line):
              - Bootstrap: fit and add remaining clusters as new lines.
              - Stop the loop.

    Parameters
    ----------
    eps : float
        DBSCAN neighbourhood radius in scaled space.
    min_samples : int
        DBSCAN minimum points per cluster.
    x_scale : float
        Compression applied to the along-trace axis before DBSCAN.
    min_cluster_points : int
        Minimum inliers for a cubic fit to be accepted.
    min_span : float
        Minimum along-trace span (metres) of inliers for a fit to be accepted.
    residual_threshold : float
        RANSAC inlier threshold for the cubic fit (metres, perpendicular).
    max_trials : int
        RANSAC maximum iterations.
    similarity_threshold : float
        Mean distance (metres, in rotated space) below which a cluster is
        considered to belong to an existing line.
    curvature_limit : float
        Maximum allowed deviation of the fit's curvature from the trajectory's.
    sample_points : int
        Number of points sampled per fitted segment when building LineStrings.
    """

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 5,
        x_scale: float = 0.2,
        min_cluster_points: int = 10,
        min_span: float = 5.0,
        residual_threshold: float = 0.15,
        max_trials: int = 200,
        similarity_threshold: float = 0.5,
        curvature_limit: float = 0.05,
        sample_points: int = 100,
    ):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.x_scale = x_scale
        self.min_cluster_points = min_cluster_points
        self.min_span = min_span
        self.residual_threshold = residual_threshold
        self.max_trials = max_trials
        self.similarity_threshold = similarity_threshold
        self.curvature_limit = curvature_limit
        self.sample_points = sample_points

    # ------------------------------------------------------------------
    # Transform helpers
    # Forward  : pts_rot = (pts_xy − origin) @ R.T
    # Inverse  : pts_xy  = pts_rot @ R + origin
    # ------------------------------------------------------------------

    def _get_transform(self, window_indices: np.ndarray, context: Context):
        """Return (R, origin, trace_geom) or None.

        R      : (2,2) rotation matrix whose rows are [u_along, u_perp].
        origin : (2,) world-space anchor (first point of the clipped trace).
        trace_geom : shapely.LineString of the local trace segment.
        """
        las = context.las
        xy = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[window_indices],
            np.asarray(las.y, dtype=np.float64)[window_indices],
        ])
        if len(xy) < 2:
            return None

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
            trace_geom = linemerge(trace_raw)
            if trace_geom.geom_type == "MultiLineString":
                trace_geom = max(trace_geom.geoms, key=lambda g: g.length)
        else:
            trace_geom = trace_raw

        if trace_geom.is_empty or len(trace_geom.coords) < 2:
            return None

        S = np.array(trace_geom.coords[0], dtype=np.float64)
        E = np.array(trace_geom.coords[-1], dtype=np.float64)
        v = E - S
        norm = np.linalg.norm(v)
        if norm < 1e-9:
            return None

        u = v / norm
        u_perp = np.array([-u[1], u[0]])
        R = np.stack([u, u_perp], axis=0)  # (2,2)
        return R, S, trace_geom

    # ------------------------------------------------------------------
    # Trajectory curvature
    # ------------------------------------------------------------------

    def _trajectory_curvature(
        self,
        trace_geom: shapely.LineString,
        R: np.ndarray,
        origin: np.ndarray,
    ) -> float | None:
        """Fit y = a·x² + b·x + c to the trace in rotated space; return 2·a."""
        coords = np.array(trace_geom.coords, dtype=np.float64)[:, :2]
        if len(coords) < 3:
            return None
        rot = (coords - origin) @ R.T  # (M,2)
        try:
            coeffs = np.polyfit(rot[:, 0], rot[:, 1], 2)  # [a, b, c]
            return float(2.0 * coeffs[0])  # second derivative = 2a
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Cubic RANSAC
    # ------------------------------------------------------------------

    def _fit_cubic(self, x: np.ndarray, y: np.ndarray):
        """Fit y = a₃x³ + a₂x² + a₁x + a₀ with RANSAC.

        Returns (fitted_pipeline, inlier_mask) or None.
        """
        if len(x) < self.min_cluster_points:
            return None
        estimator = make_pipeline(
            PolynomialFeatures(degree=3, include_bias=False),
            LinearRegression(),
        )
        ransac = RANSACRegressor(
            estimator=estimator,
            min_samples=max(4, self.min_cluster_points // 2),
            residual_threshold=self.residual_threshold,
            max_trials=self.max_trials,
        )
        try:
            ransac.fit(x.reshape(-1, 1), y)
        except Exception:
            return None
        return ransac.estimator_, ransac.inlier_mask_

    def _curvature_at_midpoint(self, model, x_min: float, x_max: float) -> float:
        """Second derivative of the cubic at the x midpoint: 6·a₃·x_mid + 2·a₂."""
        x_mid = (x_min + x_max) / 2.0
        coef = model.named_steps["linearregression"].coef_  # [a₁, a₂, a₃]
        a2 = float(coef[1])
        a3 = float(coef[2])
        return 6.0 * a3 * x_mid + 2.0 * a2

    # ------------------------------------------------------------------
    # Cluster ↔ line similarity
    # ------------------------------------------------------------------

    def _cluster_line_similarity(
        self,
        cluster_rot: np.ndarray,       # (N,2) in rotated space
        line: shapely.LineString,
        R: np.ndarray,
        origin: np.ndarray,
    ) -> float:
        """Mean distance from cluster points to *line* in the rotated frame.

        Returns inf when there is no overlap in the along-trace (x) direction.
        """
        line_coords = np.array(line.coords, dtype=np.float64)[:, :2]
        line_rot = (line_coords - origin) @ R.T  # (M,2)

        # Require x-range overlap
        overlap_lo = max(line_rot[:, 0].min(), cluster_rot[:, 0].min())
        overlap_hi = min(line_rot[:, 0].max(), cluster_rot[:, 0].max())
        if overlap_lo >= overlap_hi:
            return float("inf")

        # Consider only cluster points inside the overlap x-range
        in_overlap = (
            (cluster_rot[:, 0] >= overlap_lo) & (cluster_rot[:, 0] <= overlap_hi)
        )
        if not in_overlap.any():
            return float("inf")

        line_rot_geom = shapely.LineString(line_rot[:, :2])
        pts = shapely.points(cluster_rot[in_overlap, 0], cluster_rot[in_overlap, 1])
        dists = shapely.distance(pts, line_rot_geom)  # vectorised
        return float(dists.mean())

    # ------------------------------------------------------------------
    # Build / merge line segments
    # ------------------------------------------------------------------

    def _sample_line(
        self,
        model,
        x_min: float,
        x_max: float,
        R: np.ndarray,
        origin: np.ndarray,
    ) -> shapely.LineString:
        """Sample the cubic model over [x_min, x_max] and return in world coords."""
        x_s = np.linspace(x_min, x_max, self.sample_points)
        y_s = model.predict(x_s.reshape(-1, 1))
        pts_world = np.column_stack([x_s, y_s]) @ R + origin
        return shapely.LineString(pts_world)

    def _merge_line(
        self,
        existing: shapely.LineString,
        model,
        x_min: float,
        x_max: float,
        R: np.ndarray,
        origin: np.ndarray,
    ) -> shapely.LineString:
        """Merge existing line with a new fitted segment.

        The portion of the existing line within [x_min, x_max] (in rotated
        space) is replaced by the new fit; portions outside are kept, giving
        the union x-range.
        """
        old_coords = np.array(existing.coords, dtype=np.float64)[:, :2]
        old_rot = (old_coords - origin) @ R.T  # transform into current window frame

        # Keep only existing points that lie outside the new segment's x-range
        outside = (old_rot[:, 0] < x_min) | (old_rot[:, 0] > x_max)
        kept_rot = old_rot[outside]  # (K,2)

        # Sample the new fit
        x_s = np.linspace(x_min, x_max, self.sample_points)
        y_s = model.predict(x_s.reshape(-1, 1))
        new_rot = np.column_stack([x_s, y_s])  # (S,2)

        combined_rot = (
            np.vstack([kept_rot[:, :2], new_rot]) if len(kept_rot) else new_rot
        )
        order = np.argsort(combined_rot[:, 0], kind="stable")
        sorted_rot = combined_rot[order]

        pts_world = sorted_rot @ R + origin
        if len(pts_world) < 2:
            return existing
        return shapely.LineString(pts_world)

    # ------------------------------------------------------------------
    # Per-window logic
    # ------------------------------------------------------------------

    def _process_window(self, window_indices: np.ndarray, context: Context):
        result = self._get_transform(window_indices, context)
        if result is None:
            return
        R, origin, trace_geom = result

        las = context.las
        xy = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[window_indices],
            np.asarray(las.y, dtype=np.float64)[window_indices],
        ])
        xy_rot = (xy - origin) @ R.T  # (N,2) — col0 = along-trace, col1 = perp

        traj_curvature = self._trajectory_curvature(trace_geom, R, origin)

        # Scale along trace and cluster
        xy_scaled = xy_rot.copy()
        xy_scaled[:, 0] *= self.x_scale
        raw_labels = DBSCAN(
            eps=self.eps, min_samples=self.min_samples
        ).fit_predict(xy_scaled)

        # Build cluster pool (local indices into window)
        clusters: dict[int, np.ndarray] = {
            label: np.where(raw_labels == label)[0]
            for label in np.unique(raw_labels)
            if label != -1
            and (raw_labels == label).sum() >= self.min_cluster_points
        }
        if not clusters:
            return

        remaining: set[int] = set(clusters.keys())

        while remaining:
            # ----------------------------------------------------------
            # Find (cluster, existing_line) pair with best similarity
            # ----------------------------------------------------------
            best_dist = float("inf")
            best_ck = None
            best_li = None

            for ck in remaining:
                cluster_rot = xy_rot[clusters[ck]]
                for li, line in enumerate(context.lines):
                    d = self._cluster_line_similarity(cluster_rot, line, R, origin)
                    if d < best_dist:
                        best_dist = d
                        best_ck = ck
                        best_li = li

            # ----------------------------------------------------------
            # Merge branch: cluster is close to an existing line
            # ----------------------------------------------------------
            if context.lines and best_dist <= self.similarity_threshold:
                cluster_idx = clusters[best_ck]
                remaining.discard(best_ck)

                x_c = xy_rot[cluster_idx, 0]
                y_c = xy_rot[cluster_idx, 1]
                fit = self._fit_cubic(x_c, y_c)
                if fit is None:
                    continue

                model, inlier_mask = fit
                x_in = x_c[inlier_mask]

                if len(x_in) < self.min_cluster_points:
                    continue
                if float(x_in.max() - x_in.min()) < self.min_span:
                    continue
                if traj_curvature is not None:
                    fit_curv = self._curvature_at_midpoint(
                        model, float(x_in.min()), float(x_in.max())
                    )
                    if abs(fit_curv - traj_curvature) > self.curvature_limit:
                        continue

                context.lines[best_li] = self._merge_line(
                    context.lines[best_li],
                    model,
                    float(x_in.min()),
                    float(x_in.max()),
                    R,
                    origin,
                )

            # ----------------------------------------------------------
            # Bootstrap branch: no existing line matches → create new ones
            # ----------------------------------------------------------
            else:
                for ck in list(remaining):
                    cluster_idx = clusters[ck]
                    x_c = xy_rot[cluster_idx, 0]
                    y_c = xy_rot[cluster_idx, 1]

                    fit = self._fit_cubic(x_c, y_c)
                    if fit is None:
                        continue

                    model, inlier_mask = fit
                    x_in = x_c[inlier_mask]

                    if len(x_in) < self.min_cluster_points:
                        continue
                    if float(x_in.max() - x_in.min()) < self.min_span:
                        continue
                    if traj_curvature is not None:
                        fit_curv = self._curvature_at_midpoint(
                            model, float(x_in.min()), float(x_in.max())
                        )
                        if abs(fit_curv - traj_curvature) > self.curvature_limit:
                            continue

                    context.lines.append(
                        self._sample_line(
                            model, float(x_in.min()), float(x_in.max()), R, origin
                        )
                    )
                break  # stop after bootstrapping remaining clusters

    # ------------------------------------------------------------------
    # Stage entry point
    # ------------------------------------------------------------------

    def run(self, context: Context):
        context.lines = []  # fresh start; flat list of tracked lane LineStrings

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
        self.logger.info(f"FitLinesV2: {n_windows} windows.")
        log_step = max(1, n_windows // 10)

        for i, window_indices in enumerate(windows):
            if i % log_step == 0 or i == n_windows - 1:
                self.logger.info(
                    f"Window {i + 1}/{n_windows} — {len(context.lines)} tracked lines."
                )
            if window_indices.size == 0:
                continue
            self._process_window(window_indices, context)

        self.logger.info(
            f"FitLinesV2 done: {len(context.lines)} lane lines total."
        )
