from __future__ import annotations

import numpy as np
from scipy.spatial import KDTree

from lane_detection.pipeline.pipeline import Stage, Context
from lane_detection.utils.grid import build_grid


class LocalSORStage(Stage):
    """
    Local Statistical Outlier Removal (SOR).

    Points are assigned to a square grid. For each cell the SOR threshold is
    computed from the *k* nearest neighbours of every point **in the full active
    point cloud** (not limited to the same cell). A point is kept when its mean
    k-NN distance is within ``mean ± sigma * std`` of all mean distances in that
    cell. The result is written to ``context.global_mask``.

    Parameters
    ----------
    square_size : float
        Side length of the grid cells used to group points.
    k : int
        Number of nearest neighbours to query (the point itself is excluded).
    sigma : float
        Number of standard deviations away from the cell mean a point may be
        before it is considered an outlier.
    """

    def __init__(self, square_size: float, k: int = 16, sigma: float = 2.0):
        super().__init__()
        self.square_size = square_size
        self.k = k
        self.sigma = sigma

    def run(self, context: Context):
        las = context.las

        if context.global_mask is not None:
            active_idx = np.nonzero(context.global_mask)[0]
        else:
            active_idx = np.arange(len(las.x), dtype=np.int64)

        n_active = len(active_idx)
        if n_active == 0:
            self.logger.info("No active points; skipping LocalSOR.")
            return

        self.logger.info(
            f"LocalSOR: {n_active} active points, square_size={self.square_size}, "
            f"k={self.k}, sigma={self.sigma}."
        )

        xyz = np.column_stack([
            np.asarray(las.x, dtype=np.float64)[active_idx],
            np.asarray(las.y, dtype=np.float64)[active_idx],
            np.asarray(las.z, dtype=np.float64)[active_idx],
        ])

        # Build a KDTree over all active points for global neighbourhood queries
        k_query = min(self.k + 1, n_active)  # +1 because the point itself is included
        tree = KDTree(xyz)
        dists, _ = tree.query(xyz, k=k_query)
        # dists[:, 0] is 0.0 (self); take the rest
        mean_nn_dist = dists[:, 1:].mean(axis=1)  # (n_active,)

        # Assign active points to grid cells and apply per-cell SOR threshold
        grid = build_grid(xyz, self.square_size)
        self.logger.info(f"Grid has {len(grid)} cells.")

        local_keep = np.zeros(n_active, dtype=bool)

        for cell_indices in grid.values():
            cell_dists = mean_nn_dist[cell_indices]
            mu = cell_dists.mean()
            std = cell_dists.std()
            threshold = mu + self.sigma * std
            local_keep[cell_indices] = cell_dists <= threshold

        kept = int(local_keep.sum())
        self.logger.info(f"LocalSOR done: kept {kept}/{n_active} active points.")

        new_global = np.zeros(len(las.x), dtype=bool)
        new_global[active_idx[local_keep]] = True

        context.prev_mask = context.global_mask
        context.global_mask = new_global
