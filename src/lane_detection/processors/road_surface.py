from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from collections import deque

import shapely
from shapely import contains_xy
from shapely.ops import linemerge

from lane_detection.pipeline.pipeline import Context
from lane_detection.processors.base import Processor
from lane_detection.utils.grid import build_grid
from lane_detection.utils.transform import scale_along_trace_mtx

class RoadSurfaceFilter(Processor):
    """Road Surface Filter Stage
    Use after intensity thresholding.

    Method: splits each window into grid cells.
    With region growing from the car trace, selects the cells that are part of the road surface, then applies a buffer to them.
    """

    def __init__(self, rect_width: float = 0.5, rect_len: float = 0.5, road_surface_buffer: float = 0.2, distance_clip: float = 0):
        """
        :param square_size: the size of squares to run the region growing with (in the transformed space)
        :parem scale_along_trace: the window size is scaled along the trace by this amount
        :param road_surface_buffer: once the road surface cells are determined, a buffer is added.
        :param distance_clip: road surface will be clipped by this distance from car trace. In case of 0, no clip is applied.
        """
        super().__init__()
        self.rect_width = rect_width
        if road_surface_buffer < 0:
            raise ValueError("Road surface buffer should be non-negative.")
        self.road_surface_buffer = road_surface_buffer

        if rect_len <= 0:
            raise ValueError("Scale along trace should be positive.")
        self.rect_len = rect_len

        if distance_clip < 0:
            raise ValueError("Distance clip should be non-negative.")
        self.distance_clip = distance_clip

    def _point_to_grid(self, x, y):
        return int(np.floor(x / self.rect_width)), int(np.floor(y / self.rect_width))
    
    def _neighbors(self, cell):
        x, y = cell
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
    
    def process_window(self, bin_indices, indices: np.ndarray, context: Context):
        rect_width = self.rect_width
        xy = np.stack((context.las[indices].x, context.las[indices].y),axis=1)
        hull_polygon = shapely.Polygon(xy[ConvexHull(xy).vertices])
        gps_time = np.asarray(context.las.gps_time, dtype=np.float64)
        min_time = gps_time[indices].min()
        max_time = gps_time[indices].max()
        trace_raw = shapely.LineString([loc[:2] for loc, t in context.trace if min_time <= t <= max_time]).intersection(hull_polygon)
        if trace_raw.geom_type == 'MultiLineString':
            trace = linemerge(trace_raw)
            if trace.geom_type == 'MultiLineString':
                trace = max(trace.geoms, key=lambda g: g.length)
        else:
            trace = trace_raw
        if len(trace.coords) <= 1:
            return np.zeros(len(indices), dtype=bool)
        S, E = trace.coords[0], trace.coords[-1]
        M, M_inv = scale_along_trace_mtx(S,E, self.rect_width/self.rect_len)

        if self.distance_clip > 0:
            hull_polygon = shapely.intersection(hull_polygon, trace.buffer(self.distance_clip+2*rect_width))
        hull_coords = np.array(hull_polygon.exterior.coords)
        transformed_coords = hull_coords @ M.T
        hull_polygon_transformed = shapely.Polygon(transformed_coords)
        
        xy_transformed = xy @ M.T
        
        trace_cells_transformed = set()
        if not trace.is_empty:
            for d in np.arange(0, trace.length, rect_width/2):
                p = trace.interpolate(d)
                p_vec = np.array([p.x, p.y])
                p_transformed = p_vec @ M.T
                trace_cells_transformed.add(
                    self._point_to_grid(p_transformed[0], p_transformed[1])
                )

        grid = build_grid(xy_transformed, square_size=self.rect_width)

        queue = deque(trace_cells_transformed)
        visited = set()

        while queue:
            cell = queue.popleft()

            if cell in visited:
                continue
            if cell in grid:
                continue

            wx = (cell[0]+0.5) * rect_width
            wy = (cell[1]+0.5) * rect_width
            if not hull_polygon_transformed.contains(shapely.Point(wx, wy)):
                continue
            
            visited.add(cell)

            for nb in self._neighbors(cell):
                if nb not in grid and nb not in visited:
                    queue.append(nb)

        shapely_pts = set()
        for cell in visited:
            wx = cell[0] * rect_width
            wy = cell[1] * rect_width

            cell_pts = np.array([
                (wx, wy),
                (wx + rect_width, wy),
                (wx + rect_width, wy + rect_width),
                (wx, wy + rect_width),
            ])

            shapely_pts.update([tuple(p) for p in cell_pts @ M_inv.T])

        new_hull = shapely.MultiPoint(shapely_pts).convex_hull.buffer(self.road_surface_buffer)
        context.road_surfaces.append(new_hull)

        xy = np.stack(
            (context.las[indices].x, context.las[indices].y),
            axis=1
        )

        keep_mask = contains_xy(new_hull, xy[:, 0], xy[:, 1])
        return keep_mask
        