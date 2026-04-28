"""Ground segmentation as a pipeline stage.

Three algorithms are supported via the ``method`` parameter:

* ``'basic'``    — fit a plane in each grid square with RANSAC once.
* ``'grad'``     — same, but reject planes that are too steep and retry by
                    dropping the inliers and re-fitting until a plane with
                    gradient < ``max_gradient`` is found or RANSAC gives up.
* ``'propagate'`` — start from a square that lies on the car trace, fit the
                    plane with the ``'grad'`` logic, then flood-fill outward.
                    For each new square the already-processed neighbours
                    impose a height along the shared edge; points whose height
                    would make an edge slope steeper than ``max_gradient`` are
                    removed *before* RANSAC so they cannot bias the fit.

All three operate on whatever points are currently "in" the cloud
(``context.global_mask`` if set, otherwise all points). They update
``context.global_mask`` / ``context.prev_mask`` and refresh ``context.bins``
the same way :class:`Processor` does.
"""
from __future__ import annotations

from collections import deque
from typing import Optional

import numpy as np
import open3d as o3d

from lane_detection.pipeline.pipeline import Stage
from lane_detection.utils.grid import build_grid
from lane_detection.utils.logger import create_logger
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
    def __init__(
        self,
        square_size: float = 5.0,
        method: str = 'basic',
        distance_threshold: float = 0.2,
        fit_threshold: float = 0.05,
        max_gradient: float = 0.4,
    ):
        super().__init__()
        if method not in ('basic', 'grad', 'propagate'):
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

        if self.method == 'propagate':
            ground_local = self._run_propagate(active_pts, grid, context)
        else:
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

    # --------------------------------------------------------------------- #
    def _run_propagate(self, active_pts, grid, context):
        """Method 'propagate': flood-fill from cells that cover the car trace."""
        # Seed cells from trace
        trace = context.trace
        if not trace:
            self.logger.info("No trace available; falling back to 'grad' method.")
            self.method = 'grad'
            return self._run_independent(active_pts, grid)

        trace_xy = np.asarray([p[0][:2] for p in trace], dtype=np.float64)
        seed_cells: list[tuple[int, int]] = []
        seen_seeds: set[tuple[int, int]] = set()
        for x, y in trace_xy:
            cell = (int(np.floor(x / self.square_size)),
                    int(np.floor(y / self.square_size)))
            if cell in grid and cell not in seen_seeds:
                seen_seeds.add(cell)
                seed_cells.append(cell)

        if not seed_cells:
            self.logger.info("No trace cell overlaps grid; falling back to 'grad' method.")
            self.method = 'grad'
            return self._run_independent(active_pts, grid)

        # BFS over cells; for each new cell, use already-processed neighbours'
        # edge heights to filter steep points before fitting.
        planes: dict[tuple[int, int], tuple[float, float, float]] = {}
        ground = np.zeros(len(active_pts), dtype=bool)

        queue = deque(seed_cells)
        queued: set[tuple[int, int]] = set(seed_cells)
        processed = 0
        total = len(grid)
        log_step = max(1, total // 10)

        while queue:
            cell = queue.popleft()
            processed += 1
            if processed % log_step == 0:
                self.logger.info(f"Propagating: {processed}/{total} cells visited "
                            f"({processed / total * 100:.0f}%)")

            indices = grid[cell]
            pts = active_pts[indices]

            # Pre-filter points that would form a too-steep slope to any
            # processed neighbour's plane along the shared edge.
            keep = self._prefilter_by_neighbors(pts, cell, planes)
            fit_pts = pts[keep]

            plane = _fit_grad(fit_pts, self.fit_threshold, self.max_gradient)
            if plane is not None:
                planes[cell] = plane
                a, b, c = plane
                pred = a * pts[:, 0] + b * pts[:, 1] + c
                ground[indices[np.abs(pts[:, 2] - pred) < self.distance_threshold]] = True

            # Enqueue 4-neighbours that have points and aren't already queued
            cx, cy = cell
            for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                ncell = (nx, ny)
                if ncell in grid and ncell not in queued:
                    queued.add(ncell)
                    queue.append(ncell)

        unreached = total - len(queued)
        if unreached:
            self.logger.info(f"{unreached} cells unreachable from trace; left unprocessed.")
        return ground

    # --------------------------------------------------------------------- #
    def _prefilter_by_neighbors(
        self,
        pts: np.ndarray,
        cell: tuple[int, int],
        planes: dict[tuple[int, int], tuple[float, float, float]],
    ) -> np.ndarray:
        """Return boolean mask: True = keep for plane fitting.

        For each already-processed neighbour, evaluate its plane on this cell's
        shared edge (the neighbour's closest face). A point ``(x, y, z)`` is
        discarded if the vertical difference to that edge height, divided by
        the horizontal distance to the edge, exceeds ``max_gradient``.
        """
        cx, cy = cell
        s = self.square_size
        # Cell AABB
        x0, x1 = cx * s, (cx + 1) * s
        y0, y1 = cy * s, (cy + 1) * s

        mask = np.ones(len(pts), dtype=bool)

        # Neighbour direction -> (is_x_edge, edge_coord_of_this_cell)
        # For the +x neighbour, the shared edge lies on x = x1; distance from a
        # point (x, y) to that edge is (x1 - x). Evaluate neighbour plane at
        # (x1, y) to get the reference height on the edge.
        neighbours = [
            ((cx + 1, cy), 'x', x1),
            ((cx - 1, cy), 'x', x0),
            ((cx, cy + 1), 'y', y1),
            ((cx, cy - 1), 'y', y0),
        ]

        for ncell, axis, edge in neighbours:
            plane = planes.get(ncell)
            if plane is None:
                continue
            a, b, c = plane
            if axis == 'x':
                edge_z = a * edge + b * pts[:, 1] + c
                horiz = np.abs(pts[:, 0] - edge)
            else:
                edge_z = a * pts[:, 0] + b * edge + c
                horiz = np.abs(pts[:, 1] - edge)
            # Avoid division by zero for points exactly on the edge
            safe = horiz > 1e-9
            slope = np.zeros(len(pts))
            slope[safe] = np.abs(pts[safe, 2] - edge_z[safe]) / horiz[safe]
            mask &= slope <= self.max_gradient

        return mask
