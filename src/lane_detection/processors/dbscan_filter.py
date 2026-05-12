from __future__ import annotations

import numpy as np
import shapely
from shapely.ops import linemerge
from sklearn.cluster import DBSCAN

from lane_detection.pipeline.pipeline import Context
from lane_detection.processors.base import Processor


class DBSCANFilterStage(Processor):
    """Per-window DBSCAN cluster filter.

    For each window the points are:
      1. Rotated so that the car trajectory aligns with the x-axis.
      2. Scaled along x by *x_scale* (squishing the along-trace direction), so
         that clusters elongated along the road (lane markings) appear more
         compact and round to DBSCAN, while still being distinguished from truly
         round blobs in the perpendicular direction.
      3. DBSCAN is run in the scaled space.
      4. Each cluster is evaluated in the *original* (rotated-but-not-scaled)
         space. A cluster is kept when it is elongated enough along the trace
         axis, measured by the ratio ``x_span / y_span``.  Clusters that are
         too round (ratio below *min_elongation*) are discarded as non-markings.
      5. The boolean keep/discard decision per point is propagated back to the
         global mask via the standard Processor mechanism.

    Parameters
    ----------
    eps : float
        DBSCAN neighbourhood radius (in scaled space).
    min_samples : int
        DBSCAN minimum cluster size.
    x_scale : float
        Scale factor applied to the along-trace (x) coordinate before running
        DBSCAN.  Values < 1 compress x, making elongated clusters appear
        rounder so the single ``eps`` can still group them.
    min_elongation : float
        Minimum x_span / y_span ratio for a cluster to be kept.  Clusters below
        this threshold are considered too round to be lane markings.
    min_cluster_points : int
        Clusters with fewer points than this are always discarded (noise guard).
    """

    def __init__(
        self,
        eps: float = 0.3,
        min_samples: int = 5,
        x_scale: float = 0.2,
        min_elongation: float = 5.0,
        min_cluster_points: int = 5,
    ):
        super().__init__()
        self.eps = eps
        self.min_samples = min_samples
        self.x_scale = x_scale
        self.min_elongation = min_elongation
        self.min_cluster_points = min_cluster_points

    # ------------------------------------------------------------------
    # Transform helpers (same convention as FitLinesStage)
    # Forward  : pts_t  = (pts_xy - origin) @ R.T
    # Inverse  : pts_xy = pts_t  @ R  + origin
    # ------------------------------------------------------------------

    def _get_transform(self, window_indices: np.ndarray, context: Context):
        """Return (R, origin) aligning the local trace with the x-axis, or None."""
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
        R = np.stack([u, u_perp], axis=0)  # (2, 2), rows = basis vectors
        return R, S

    # ------------------------------------------------------------------
    # Processor interface
    # ------------------------------------------------------------------

    def process_window(self, indices: np.ndarray, context: Context) -> np.ndarray:
        result = self._get_transform(indices, context)
        if result is None:
            # No usable trace for this window — keep everything
            return np.ones(len(indices), dtype=bool)

        R, origin = result
        las = context.las
        xy = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[indices],
            np.asarray(las.y, dtype=np.float64)[indices],
        ])

        # Rotate: col 0 = along-trace (x), col 1 = perpendicular (y)
        xy_rot = (xy - origin) @ R.T  # (N, 2)

        # Scale along trace for DBSCAN
        xy_scaled = xy_rot.copy()
        xy_scaled[:, 0] *= self.x_scale

        labels = DBSCAN(eps=self.eps, min_samples=self.min_samples).fit_predict(xy_scaled)

        keep = np.zeros(len(indices), dtype=bool)

        for label in np.unique(labels):
            if label == -1:  # DBSCAN noise
                continue

            mask = labels == label
            if mask.sum() < self.min_cluster_points:
                continue

            cluster_rot = xy_rot[mask]
            x_span = float(cluster_rot[:, 0].max() - cluster_rot[:, 0].min())
            y_span = float(cluster_rot[:, 1].max() - cluster_rot[:, 1].min())

            # Avoid division by zero for degenerate (single-row) clusters
            if y_span < 1e-6:
                elongation = float("inf")
            else:
                elongation = x_span / y_span

            if elongation >= self.min_elongation:
                keep[mask] = True

        return keep
