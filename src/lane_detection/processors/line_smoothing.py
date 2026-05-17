from __future__ import annotations

import numpy as np
from shapely.geometry import LineString

from lane_detection.pipeline.pipeline import Stage, Context


class LineSmoothingStage(Stage):
    """Refine fitted lane lines using local weighted regression.

    - uses along-track weighting instead of Euclidean weighting
    - uses quadratic local fitting
    - uses adaptive tangent estimation
    - uses numerically stable weighted least squares
    """

    def __init__(
        self,
        window_length: float = 5.0,
        output_density: float = 3.0,
        line_sample_weight: float = 0.5,
        min_inliers: int = 50
    ):
        super().__init__()

        self.window_length = float(window_length)
        self.output_density = float(output_density)
        self.line_sample_weight = float(line_sample_weight)
        self.min_inliers = min_inliers

    def run(self, context: Context):
        n_lines = len(context.lines)

        self.logger.info(f"Refining {n_lines} lines")

        refined_lines = []

        for i, (line, inliers) in enumerate(context.lines):

            if (i + 1) % max(1, n_lines // 10) == 0 or i == n_lines - 1:
                self.logger.info(f"  Line {i + 1}/{n_lines}")

            if line is None or inliers is None or len(inliers) < self.min_inliers:
                if line is not None:
                    refined_lines.append((line, inliers))
                continue

            n_pts = max(
                2,
                int(np.ceil(line.length / 10.0 * self.output_density)),
            )

            # Sample original line to stabilize MLS
            line_samples = np.array([
                line.interpolate(d).coords[0]
                for d in np.linspace(0, line.length, n_pts)
            ])

            combined_pts = np.vstack([inliers, line_samples])

            # Weights:
            # - real inliers get weight 1.0
            # - synthetic line samples get smaller weight
            point_weights = np.concatenate([
                np.ones(len(inliers)),
                np.full(len(line_samples), self.line_sample_weight),
            ])

            refined = self._moving_least_squares(
                line=line,
                pts=combined_pts,
                point_weights=point_weights,
                n_pts=n_pts,
            )

            refined_lines.append((
                refined if refined is not None else line,
                inliers,
            ))

        context.lines = refined_lines

    def _moving_least_squares(
        self,
        line: LineString,
        pts: np.ndarray,
        point_weights: np.ndarray,
        n_pts: int,
    ) -> LineString | None:
        """Smooth line using local quadratic weighted regression."""

        line_length = line.length

        if line_length < self.window_length:
            return None

        # Precompute approximate arc-length projections
        point_arc_lengths = np.array([
            line.project(LineString([p, p]).centroid)
            for p in pts
        ])

        sample_distances = np.linspace(0.0, line_length, n_pts)

        result_pts = []

        sigma = self.window_length / 2.0

        for d in sample_distances:

            center = np.array(line.interpolate(d).coords[0])

            # Adaptive tangent estimation
            eps = min(
                self.window_length * 0.25,
                max(line_length * 0.02, 0.5),
            )

            p_fwd = np.array(
                line.interpolate(min(d + eps, line_length)).coords[0]
            )

            p_bwd = np.array(
                line.interpolate(max(d - eps, 0.0)).coords[0]
            )

            tangent = p_fwd - p_bwd

            tangent_norm = np.linalg.norm(tangent)

            if tangent_norm < 1e-12:
                result_pts.append(center)
                continue

            tangent /= tangent_norm

            normal = np.array([
                -tangent[1],
                tangent[0],
            ])

            # Along-track distances
            along_track_dist = point_arc_lengths - d

            # Gaussian weighting using along-track distance
            weights = np.exp(
                -0.5 * (along_track_dist / sigma) ** 2
            )

            # Apply external point weights
            weights *= point_weights

            mask = weights > 1e-4

            if np.count_nonzero(mask) < 5:
                result_pts.append(center)
                continue

            local_pts = pts[mask]
            w = weights[mask]

            diffs = local_pts - center

            # Frenet coordinates
            t_local = diffs @ tangent
            n_local = diffs @ normal

            # Quadratic fit:
            # n = a*t² + b*t + c
            A = np.column_stack([
                t_local ** 2,
                t_local,
                np.ones_like(t_local),
            ])

            # Stable weighted least squares
            sqrt_w = np.sqrt(w)

            Aw = A * sqrt_w[:, None]
            bw = n_local * sqrt_w

            try:
                params, *_ = np.linalg.lstsq(
                    Aw,
                    bw,
                    rcond=None,
                )

            except np.linalg.LinAlgError:
                result_pts.append(center)
                continue

            # Evaluate at t=0
            n_refined = params[2]

            refined_pt = center + n_refined * normal

            result_pts.append(refined_pt)

        if len(result_pts) < 2:
            return None

        # Remove duplicate neighboring points
        filtered_pts = [result_pts[0]]

        for p in result_pts[1:]:
            if np.linalg.norm(p - filtered_pts[-1]) > 1e-6:
                filtered_pts.append(p)

        if len(filtered_pts) < 2:
            return None

        return LineString(filtered_pts)
