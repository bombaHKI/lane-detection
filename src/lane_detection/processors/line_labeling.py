from __future__ import annotations

import numpy as np
from shapely.geometry import LineString
from shapely.ops import substring
from shapely import Point

from lane_detection.pipeline.pipeline import Stage, Context


class LineLabelingStage(Stage):
    """Label lane lines as solid or dashed.

    Workflow:
    1. Project inliers onto the line
    2. Sort points along the line
    3. Split sections where gaps between consecutive points are large
    4. Classify:
       - long section  -> solid
       - short section -> dashed
    """

    def __init__(
        self,
        line_width: float = 0.25,
        max_point_gap: float = 2.0,
        min_line_len: float = 40.0,
        min_section_len: float = 3.0,
        empty_len_threshold: float = 30.0,
        solid_line_threshold: float = 30.0,
    ):
        super().__init__()
        self.line_width = float(line_width)
        self.max_point_gap = float(max_point_gap)
        self.min_line_len = float(min_line_len)
        self.min_section_len = float(min_section_len)
        self.empty_len_threshold = float(empty_len_threshold)
        self.solid_line_threshold = float(solid_line_threshold)

    def run(self, context: Context):
        n_lines = len(context.lines)
        self.logger.info(f"Labeling {n_lines} lines")

        # Get all masked points from context
        las = context.las
        all_pts = np.column_stack([las.x, las.y])[context.global_mask]

        labelled_lines: list[tuple[LineString, np.ndarray, str]] = []

        for i, (line, _inliers) in enumerate(context.lines):
            if (i + 1) % max(1, n_lines // 10) == 0 or i == n_lines - 1:
                self.logger.info(f"  Line {i + 1}/{n_lines}")

            if line is None:
                continue

            # Select points within line_width of the line
            dists = np.array([line.distance(Point(p)) for p in all_pts])
            nearby_mask = dists <= self.line_width
            nearby_pts = all_pts[nearby_mask]

            if len(nearby_pts) == 0:
                continue

            labelled_lines.extend(self._process_line(line, nearby_pts))

        self.logger.info(
            f"Produced {len(labelled_lines)} labelled segments "
            f"({sum(1 for _, _, l in labelled_lines if l == 'solid')} solid, "
            f"{sum(1 for _, _, l in labelled_lines if l == 'dashed')} dashed)"
        )

        context.lines = labelled_lines

    def _process_line(
        self,
        line: LineString,
        inliers: np.ndarray,
    ) -> list[tuple[LineString, np.ndarray, str]]:
        """Split line into continuous sections and classify them."""

        distances = np.array([line.project(Point(p)) for p in inliers])

        order = np.argsort(distances)
        distances = distances[order]
        inliers = inliers[order]

        if len(distances) < 2:
            return []

        intervals = self._extract_intervals_from_gaps(distances)
        intervals = [(s, e) for s, e in intervals if (e - s) >= self.min_section_len]

        if not intervals:
            return []
        
        split_groups = self._split_on_gaps(intervals)

        results: list[tuple[LineString, np.ndarray, str]] = []

        for group in split_groups:
            group_start = group[0][0]
            group_end = group[-1][1]
            if group_end - group_start < self.min_line_len:
                continue

            dashed_start = group_start
            for start, end in group:
                if end-start > self.solid_line_threshold:
                    if dashed_start != start:
                        dashed = substring(line, dashed_start, start)
                        mask = (distances >= dashed_start) & (distances <= start)
                        results.append((dashed, inliers[mask],"dashed"))
                    solid = substring(line, start, end)
                    mask = (distances>=start) & (distances<=end)
                    results.append((solid,inliers[mask],"solid"))
                    dashed_start = end
            if dashed_start != group_end:
                dashed = substring(line, dashed_start, group_end)
                mask = (distances >= dashed_start) & (distances <= group_end)
                results.append((dashed, inliers[mask],"dashed"))

        return results

    def _extract_intervals_from_gaps(
        self,
        distances: np.ndarray,
    ) -> list[tuple[float, float]]:
        """Extract continuous intervals separated by large empty gaps."""

        if len(distances) == 0:
            return []

        # Distance between consecutive projected points
        gaps = np.diff(distances)

        # Find large empty spaces
        split_idxs = np.where(gaps > self.max_point_gap)[0]

        # Build interval boundaries
        starts = np.concatenate(([0], split_idxs + 1))
        ends = np.concatenate((split_idxs, [len(distances) - 1]))

        intervals: list[tuple[float, float]] = []

        for s, e in zip(starts, ends):

            start_d = distances[s]
            end_d = distances[e]

            intervals.append((start_d, end_d))

        return intervals
    
    def _split_on_gaps( self, intervals: list[tuple[float, float]] ) -> list[list[tuple[float, float]]]: 
        """Split intervals into groups separated by gaps > empty_len_threshold.""" 
        if not intervals: 
            return []
        groups: list[list[tuple[float, float]]] = [[intervals[0]]] 
        for i in range(1, len(intervals)): 
            gap = intervals[i][0] - intervals[i - 1][1] 
            if gap > self.empty_len_threshold:
                groups.append([intervals[i]]) 
            else: groups[-1].append(intervals[i])
        return groups
    