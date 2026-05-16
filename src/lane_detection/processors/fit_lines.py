from __future__ import annotations

from typing import Optional

import numpy as np
import shapely

from lane_detection.pipeline.pipeline import Stage, Context
from lane_detection.utils.transform import scale_along_trace_mtx

def _eval_quadratic(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return coeffs[0] * x ** 2 + coeffs[1] * x + coeffs[2]

class _TrackedLine:
    """Accumulated lane-line geometry in world (2-D) coordinates."""

    def __init__(self, pts: np.ndarray, last_inlier_idx: int | None = None):
        self.pts_world: np.ndarray = None
        self.latest_points: np.ndarray = pts
        # Index into latest_points of the last point with inlier support.
        # Points beyond this are extrapolation and can be trimmed.
        self.last_inlier_idx: int = last_inlier_idx if last_inlier_idx is not None else len(pts) - 1

    def local_pts_latest(self, R: np.ndarray, origin: np.ndarray) -> np.ndarray:
        return (self.latest_points - origin) @ R.T

    def extend(self, pts_new: np.ndarray, last_inlier_idx: int | None = None):
        if self.pts_world is None:
            self.pts_world = self.latest_points
        else:
            self.pts_world = np.vstack([self.pts_world, self.latest_points])
        self.latest_points = pts_new
        self.last_inlier_idx = last_inlier_idx if last_inlier_idx is not None else len(pts_new) - 1

    def trim_to_last_inlier(self):
        """Remove the extrapolated tail beyond the last inlier-supported point."""
        self.latest_points = self.latest_points[:self.last_inlier_idx + 1]

    def to_linestring(self):
        n = (len(self.pts_world) if self.pts_world is not None else 0) + len(self.latest_points)
        if n < 2:
            return None
        if self.pts_world is None:
            return shapely.LineString(self.latest_points)
        return shapely.LineString(np.vstack([self.pts_world, self.latest_points]))

class FitLinesStage(Stage):
    """
    Sliding-window quadratic lane fitting.

    Processing steps
    ----------------
    1. Rotate points into trace-aligned coordinates.
    2. Continue relevant lines.
    3. Merge best continuation repeatedly.
    4. Grow new lines from remaining pool.
    """

    def __init__(
        self,
        seed_distance: float = 10.0,
        fit_threshold: float = 0.15,
        line_width_margin: float = 4,
        curvature_limit: float = 0.05,
        min_inliers: int = 20,
        ransac_iterations: int = 200,
    ):
        super().__init__()

        self.seed_distance = float(seed_distance)
        self.fit_threshold = float(fit_threshold)
        self.line_width_margin = float(line_width_margin)
        self.curvature_limit = float(curvature_limit)
        self.min_inliers = int(min_inliers)
        self.ransac_iterations = int(ransac_iterations)

    def run(self, context: Context):
        bins = context.bins
        window_size = context.window_size
        window_shift = context.window_shift

        las_xy = np.column_stack([
            context.las.x,
            context.las.y,
        ])

        lines : list[_TrackedLine] = []

        relevant_lines = set()
        for win_start in range(0, len(bins) - window_size + 1, window_shift):
            win_end = win_start + window_size
            window_bins = bins[win_start:win_end]

            trace_S = np.array(window_bins[0].trace_S)
            trace_E = np.array(window_bins[-1].trace_E)

            M, M_inv = scale_along_trace_mtx(trace_S, trace_E)

            unprocessed_bins = window_bins[-window_shift:]
            if win_start == 0:
                unprocessed_bins = window_bins

            pts_world = las_xy[np.concatenate([bin.indices for bin in unprocessed_bins])]
            pts_rot = (M @ (pts_world - trace_S).T).T
            active_lines = relevant_lines.copy()

            # extend existing lines
            while len(active_lines) > 0:
                candidates = []
                failed = []

                for line_id in active_lines:
                    line_geom = lines[line_id].to_linestring()
                    # Last inlier-supported point on the line
                    last_inlier_dist = line_geom.project(
                        shapely.Point(lines[line_id].latest_points[lines[line_id].last_inlier_idx])
                    )
                    p2_world = np.array(line_geom.interpolate(last_inlier_dist).coords[0])
                    p1_world = np.array(line_geom.interpolate(
                        max(last_inlier_dist - self.seed_distance, 0)
                    ).coords[0])
                    line_end_world = np.array(line_geom.coords[-1])

                    # Transform to local (trace-aligned) coordinates
                    p1 = M @ (p1_world - trace_S)
                    p2 = M @ (p2_world - trace_S)
                    line_end = M @ (line_end_world- trace_S)
                    inliers, coeffs = self._ransac_fit(
                        pts_rot, seed_pts=np.vstack([p1, p2]),
                    )

                    if inliers is None or len(inliers) < self.min_inliers:
                        failed.append(line_id)
                    else:
                        candidates.append({
                            "line_id": line_id,
                            "inliers": inliers,
                            "coeffs": coeffs,
                            "start_x": line_end[0],
                            "seed_pts": np.vstack([p1, p2]),
                        })

                for line_id in failed:
                    lines[line_id].trim_to_last_inlier()
                    active_lines.discard(line_id)
                    relevant_lines.discard(line_id)

                if len(candidates) == 0:
                    break

                best = max(candidates, key=lambda c: len(c["inliers"]))

                line_id = best["line_id"]
                inlier_pts = best["inliers"]
                max_inlier_x = np.max(inlier_pts[:, 0])

                fit_pts = np.vstack([best["seed_pts"], inlier_pts])
                coeffs = np.polyfit(
                    fit_pts[:, 0],
                    fit_pts[:, 1],
                    deg=2,
                )

                # Sample fitted curve in local coordinates
                start_x = best["start_x"]
                end_x = (M @ (trace_E - trace_S))[0]
                end_x += .5*(end_x-start_x)
                xs = np.linspace(start_x, end_x, 60)
                ys = _eval_quadratic(coeffs, xs)
                local_new = np.column_stack([xs, ys])

                # Transform back to world coordinates
                world_new = (M_inv @ local_new.T).T + trace_S

                # Remove points past the last unprocessed bin perpendicular
                tangent = trace_E - trace_S
                tangent = tangent / np.linalg.norm(tangent)
                signed_dist = (world_new - trace_E) @ tangent
                world_new = world_new[signed_dist <= 0]

                # Index of last point with inlier support
                last_inlier_idx = int(np.searchsorted(xs, max_inlier_x, side='right')) - 1
                last_inlier_idx = min(last_inlier_idx, len(world_new) - 1)

                lines[line_id].extend(world_new, last_inlier_idx=last_inlier_idx)

                # remove nearby points
                mask = self._distance_to_curve_mask(
                    pts_rot,
                    coeffs,
                    self.line_width_margin,
                )

                pts_rot = pts_rot[~mask]

                active_lines.remove(line_id)

            # Grow new lines
            while len(pts_rot) >= self.min_inliers:

                inliers, coeffs = self._ransac_fit(pts_rot)

                if inliers is None:
                    break

                max_inlier_x = np.max(inliers[:, 0])

                # Refit on all inliers
                coeffs = np.polyfit(
                    inliers[:, 0],
                    inliers[:, 1],
                    deg=2,
                )

                # Sample fitted curve in local coordinates
                start_x = np.min(inliers[:, 0])
                end_x = (M @ (trace_E - trace_S))[0]
                end_x += .5*(end_x-start_x)
                xs = np.linspace(start_x, end_x, 60)
                ys = _eval_quadratic(coeffs, xs)
                local_new = np.column_stack([xs, ys])

                # Transform back to world coordinates
                world_new = (M_inv @ local_new.T).T + trace_S

                # Remove points past the last unprocessed bin perpendicular
                tangent = trace_E - trace_S
                tangent = tangent / np.linalg.norm(tangent)
                signed_dist = (world_new - trace_E) @ tangent
                world_new = world_new[signed_dist <= 0]

                # Index of last point with inlier support
                last_inlier_idx = int(np.searchsorted(xs, max_inlier_x, side='right')) - 1
                last_inlier_idx = min(last_inlier_idx, len(world_new) - 1)

                new_line = _TrackedLine(world_new, last_inlier_idx=last_inlier_idx)
                lines.append(new_line)
                relevant_lines.add(len(lines) - 1)

                mask = self._distance_to_curve_mask(
                    pts_rot,
                    coeffs,
                    self.line_width_margin,
                )

                pts_rot = pts_rot[~mask]

        context.lines = [line.to_linestring() for line in lines if line.to_linestring() is not None]

    def _ransac_fit(self, pool: np.ndarray, seed_pts: Optional[np.ndarray] = None):
        """RANSAC quadratic fit.

        If *seed_pts* (Nx2) are given, each iteration samples 1 random point
        from *pool* and fits through the seeds + that point.  Otherwise 3
        random points are sampled from *pool*.
        """
        n_seed = 0 if seed_pts is None else len(seed_pts)
        n_random = 3 - n_seed

        if len(pool) < max(n_random, 1):
            return None, None

        best_inliers = None
        best_coeffs = None

        for _ in range(self.ransac_iterations):
            idx = np.random.choice(len(pool), size=n_random, replace=False)
            if seed_pts is not None:
                fit_pts = np.vstack([seed_pts, pool[idx]])
            else:
                fit_pts = pool[idx]

            try:
                coeffs = np.polyfit(fit_pts[:, 0], fit_pts[:, 1], deg=2)
            except np.linalg.LinAlgError:
                continue

            if abs(coeffs[0]) > self.curvature_limit:
                continue

            residuals = np.abs(
                pool[:, 1] - np.polyval(coeffs, pool[:, 0])
            )
            mask = residuals < self.fit_threshold
            inliers = pool[mask]

            if len(inliers) < self.min_inliers:
                continue

            if best_inliers is None or len(inliers) > len(best_inliers):
                best_inliers = inliers
                best_coeffs = coeffs

        return best_inliers, best_coeffs

    def _distance_to_curve_mask(
        self,
        pts,
        coeffs,
        threshold,
    ):
        y_hat = np.polyval(coeffs, pts[:, 0])
        d = np.abs(pts[:, 1] - y_hat)
        return d < threshold
