from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from lane_detection.pipeline.pipeline import Stage, Context


class LineSmoothingStage(Stage):
    """Refine fitted lane lines using moving least squares on nearby points.
    
    Parameters
    ----------
    window_length : the window length of the Moving Least Squares
    output_dentity : number fo point in output string per 10 meters
    """

    def __init__(
        self,
        window_length: float = 5.0,
        output_density: float = 3.0,
    ):
        super().__init__()
        self.window_length = float(window_length)
        self.output_density = float(output_density)

    def run(self, context: Context):
        n_lines = len(context.lines)
        self.logger.info(f"Refining {n_lines} lines")
        refined_lines = []
        for i, (line, inliers) in enumerate(context.lines):
            if (i + 1) % max(1, n_lines // 10) == 0 or i == n_lines - 1:
                self.logger.info(f"  Line {i + 1}/{n_lines}")
            if line is None or inliers is None or len(inliers) < 2:
                if line is not None:
                    refined_lines.append((line, inliers))
                continue

            n_pts = max(2, int(np.ceil(line.length / 10 * self.output_density)))
            # Add samples from the fitted line itself to guide MLS
            line_samples = np.array([
                line.interpolate(d).coords[0]
                for d in np.linspace(0, line.length, n_pts)
            ])
            combined_pts = np.vstack([inliers, line_samples])

            refined = self._moving_least_squares(line, combined_pts, n_pts)
            if refined is not None:
                refined_lines.append((refined, inliers))
            else:
                refined_lines.append((line, inliers))

        context.lines = refined_lines

    def _moving_least_squares(
        self, line: LineString, pts: np.ndarray, n_pts: int
    ) -> LineString | None:
        """Fit a smooth line through pts using a moving least squares approach."""
        line_length = line.length
        if line_length < self.window_length:
            return None

        # Sample evaluation points along the line
        distances = np.linspace(0, line_length, n_pts)
        result_pts = []

        # Direction vectors at each evaluation point
        for d in distances:
            center = np.array(line.interpolate(d).coords[0])

            # Distances from center to all candidate points
            diffs = pts - center
            dists = np.linalg.norm(diffs, axis=1)

            # Gaussian weight based on window_length
            sigma = self.window_length / 2.0
            weights = np.exp(-0.5 * (dists / sigma) ** 2)

            # Only use points with meaningful weight
            mask = weights > 1e-4
            if mask.sum() < 3:
                result_pts.append(center)
                continue

            w = weights[mask]
            local_pts = pts[mask]

            # Local tangent direction from the original line
            eps = 3
            p_fwd = np.array(line.interpolate(min(d + eps, line_length)).coords[0])
            p_bwd = np.array(line.interpolate(max(d - eps, 0)).coords[0])
            tangent = p_fwd - p_bwd
            tangent = tangent / (np.linalg.norm(tangent) + 1e-12)
            normal = np.array([-tangent[1], tangent[0]])

            # Project onto local coordinate (along tangent, along normal)
            local_coords = (local_pts - center) @ np.column_stack([tangent, normal])
            t_local = local_coords[:, 0]  # along-track
            n_local = local_coords[:, 1]  # cross-track

            # Weighted least squares fit: n = a*t + b
            W = np.diag(w)
            A = np.column_stack([t_local, np.ones(len(t_local))])
            try:
                AtWA = A.T @ W @ A
                AtWn = A.T @ W @ n_local
                params = np.linalg.solve(AtWA, AtWn)
            except np.linalg.LinAlgError:
                result_pts.append(center)
                continue

            # Evaluate at t=0 (the center position) to get the refined cross-track offset
            n_refined = params[1]  # a*0 + b
            refined_pt = center + n_refined * normal
            result_pts.append(refined_pt)

        if len(result_pts) < 2:
            return None

        return LineString(result_pts)
