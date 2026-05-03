from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull
from collections import deque

import shapely
from shapely import contains_xy

from lane_detection.processors.base import Processor
from lane_detection.utils.grid import build_grid

class RoadSurfaceFilter(Processor):
    """Road Surface Filter Stage
    Use after intensity thresholding.

    Method: splits each window into grid cells.
    With region growing from the car trace, selects the cells that are part of the road surface, then applies a buffer to them.
    """

    def __init__(self, square_size: float = 0.5, scale_along_trace: float = 0.5, road_surface_buffer: float = 0.2):
        """
        :param square_size: the size of squares to run the region growing with (in the transformed space)
        :parem scale_along_trace: the window size is scaled along the trace by this amount
        :param road_surface_buffer: once the road surface cells are determined, a buffer is added.
        """
        super().__init__()
        self.square_size = square_size
        self.road_surface_buffer = road_surface_buffer
        if scale_along_trace <= 0:
            raise ValueError("Scale along trace should be positive.")
        self.scale_along_trace = scale_along_trace

    def _point_to_grid(self, x, y):
        return int(np.floor(x / self.square_size)), int(np.floor(y / self.square_size))
    
    def _neighbors(self, cell):
        x, y = cell
        return [
            (x + 1, y),
            (x - 1, y),
            (x, y + 1),
            (x, y - 1),
        ]
    
    def scale_along_trace_mtx(self,S, E, scale: float):
        """
        :param S: start of the trace vector
        :param E: end of the trace
        :param scale: the amount distances in SE direction will be scaled by
        Returns:
            M      : 2x2 scaling matrix along direction SE
            M_inv  : its inverse
        """
        v = np.asarray(E) - np.asarray(S)
        norm = np.linalg.norm(v)
        if norm == 0:
            raise ValueError("S and E cannot be the same point")

        u = v / norm  # unit direction vector

        # perpendicular vector
        u_perp = np.array([-u[1], u[0]])

        # rotation matrix (basis change)
        R = np.stack([u, u_perp], axis=1)  # columns are basis vectors

        # scaling in aligned space
        S_mat = np.array([
            [scale, 0],
            [0, 1]
        ])

        # forward transform
        M = R @ S_mat @ R.T

        # inverse scaling
        S_inv = np.array([
            [1/scale, 0],
            [0, 1]
        ])

        M_inv = R @ S_inv @ R.T

        return M, M_inv

    def process_window(self, indices, context):
        square_size = self.square_size
        xy = np.stack((context.las[indices].x, context.las[indices].y),axis=1)
        hull_polygon = shapely.Polygon(xy[ConvexHull(xy).vertices])
        trace = shapely.LineString([loc[:2] for loc,t in context.trace]).intersection(hull_polygon)
        S, E = trace.coords[0], trace.coords[-1]
        M, M_inv = self.scale_along_trace_mtx(S,E,self.scale_along_trace)

        hull_polygon = hull_polygon.buffer(-square_size)
        hull_coords = np.array(hull_polygon.exterior.coords)
        transformed_coords = hull_coords @ M.T
        hull_polygon_transformed = shapely.Polygon(transformed_coords)
        
        xy_transformed = xy @ M.T
        
        trace_cells_transformed = set()
        if not trace.is_empty:
            for d in np.arange(0, trace.length, square_size/2):
                p = trace.interpolate(d)
                p_vec = np.array([p.x, p.y])
                p_transformed = p_vec @ M.T
                trace_cells_transformed.add(
                    self._point_to_grid(p_transformed[0], p_transformed[1])
                )

        grid = build_grid(xy_transformed, square_size=self.square_size)

        queue = deque(trace_cells_transformed)
        visited = set()

        while queue:
            cell = queue.popleft()

            if cell in visited:
                continue
            if cell in grid:
                continue

            wx = (cell[0]+0.5) * square_size
            wy = (cell[1]+0.5) * square_size
            if not hull_polygon_transformed.contains(shapely.Point(wx, wy)):
                continue
            
            visited.add(cell)

            for nb in self._neighbors(cell):
                if nb not in grid and nb not in visited:
                    queue.append(nb)

        shapely_pts = []
        for cell in visited:
            wx = cell[0] * square_size
            wy = cell[1] * square_size

            cell_pts = np.array([
                (wx, wy),
                (wx + square_size, wy),
                (wx + square_size, wy + square_size),
                (wx, wy + square_size),
            ])

            shapely_pts.extend(cell_pts @ M_inv.T)

        new_hull = shapely.MultiPoint(shapely_pts).convex_hull.buffer(self.road_surface_buffer)
        context.hulls.append(new_hull)

        xy = np.stack(
            (context.las[indices].x, context.las[indices].y),
            axis=1
        )

        keep_mask = contains_xy(new_hull, xy[:, 0], xy[:, 1])
        return keep_mask
        