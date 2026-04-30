from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import open3d as o3d

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.grid import build_grid
from lane_detection.utils.plane_fitting import fit_plane_ransac

# --------------------------------------------------------------------------- #
# Plane helpers
# --------------------------------------------------------------------------- #


def _plane_gradient(a: float, b: float) -> float:
    """Slope of the plane z = a*x + b*y + c (|grad z|)."""
    return float(np.hypot(a, b))


# --------------------------------------------------------------------------- #
# Per-square fitting strategies
# --------------------------------------------------------------------------- #

def _fit_basic(points, distance_threshold):
    res = fit_plane_ransac(points, distance_threshold)
    if res is None:
        return None
    a, b, c, _ = res
    return (a, b, c)


def _fit_grad(points, distance_threshold, max_gradient, max_iterations=5):
    remaining = points
    for _ in range(max_iterations):
        if len(remaining) < 3:
            return None
        res = fit_plane_ransac(remaining, distance_threshold)
        if res is None:
            return None
        a, b, c, inliers = res
        if _plane_gradient(a, b) < max_gradient:
            return (a, b, c)
        # Too steep — drop inliers and retry with the rest
        remaining = remaining[~inliers]
    return None


# --------------------------------------------------------------------------- #
# Stage
# --------------------------------------------------------------------------- #

class GroundSegmentStage(Stage):
    """Ground segmentation as a pipeline stage.

    Three algorithms are supported via the ``method`` parameter:

    * ``'basic'``    — fit a plane in each grid square with RANSAC once.
    * ``'grad'``     — same, but reject planes that are too steep and retry by
                        dropping the inliers and re-fitting until a plane with
                        gradient < ``max_gradient`` is found or RANSAC gives up.

    All three operate on whatever points are currently "in" the cloud
    (``context.global_mask`` if set, otherwise all points). They update
    ``context.global_mask`` / ``context.prev_mask`` and refresh ``context.bins``
    the same way :class:`Processor` does.
    """
    def __init__(
        self,
        square_size: float = 5.0,
        method: str = 'basic',
        distance_threshold: float = 0.2,
        fit_threshold: float = 0.05,
        max_gradient: float = 0.4,
    ):
        super().__init__()
        if method not in ('basic', 'grad'):
            raise ValueError(f"Unknown ground-segment method: {method}")
        self.square_size = float(square_size)
        self.method = method
        self.distance_threshold = float(distance_threshold)
        self.fit_threshold = float(fit_threshold)
        self.max_gradient = float(max_gradient)

    # --------------------------------------------------------------------- #
    def run(self, context):
        self.logger.info(f"Ground segmentation: method={self.method}, square_size={self.square_size}")
        xyz = context.las.xyz

        # Work on currently-included points
        if context.global_mask is not None:
            active_idx = np.nonzero(context.global_mask)[0]
        else:
            active_idx = np.arange(len(xyz), dtype=np.int64)
        active_pts = xyz[active_idx]

        if len(active_idx) == 0:
            self.logger.info("No active points; skipping.")
            return

        grid = build_grid(active_pts, self.square_size)
        self.logger.info(f"Grid built: {len(grid)} cells over {len(active_idx)} points.")

        ground_local = self._run_independent(active_pts, grid)

        global_mask = np.zeros(len(xyz), dtype=bool)
        global_mask[active_idx[ground_local]] = True

        kept = int(global_mask.sum())
        self.logger.info(f"Ground points: {kept}/{len(active_idx)}")

        context.prev_mask = context.global_mask
        context.global_mask = global_mask
        if context.bins is not None:
            context.bins = [indices[global_mask[indices]] for indices in context.bins]

    # --------------------------------------------------------------------- #
    def _run_independent(self, active_pts, grid):
        """Methods 'basic' and 'grad': each cell is processed independently."""
        fit = _fit_basic if self.method == 'basic' else _fit_grad
        ground = np.zeros(len(active_pts), dtype=bool)
        n = len(grid)
        log_step = max(1, n // 10)
        for i, (cell, indices) in enumerate(grid.items()):
            if i % log_step == 0 or i == n - 1:
                self.logger.info(f"Fitting cells: {i + 1}/{n} ({(i + 1) / n * 100:.0f}%)")
            pts = active_pts[indices]
            plane = fit(pts, self.fit_threshold) if self.method == 'basic' \
                else fit(pts, self.fit_threshold, self.max_gradient)
            if plane is None:
                continue
            a, b, c = plane
            pred = a * pts[:, 0] + b * pts[:, 1] + c
            ground[indices[np.abs(pts[:, 2] - pred) < self.distance_threshold]] = True
        return ground
