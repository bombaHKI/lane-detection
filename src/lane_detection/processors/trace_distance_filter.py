import numpy as np

from lane_detection.pipeline.pipeline import Stage

def _in_quad(px: np.ndarray, py: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Test whether each point (px[i], py[i]) lies inside a convex quadrilateral.

    Parameters
    ----------
    px, py   : (N,) float arrays of point coordinates.
    corners  : (4, 2) array of quad vertices in consistent winding order.

    Returns
    -------
    (N,) bool array — True if the point is inside (or on the boundary).
    """
    inside = np.ones(len(px), dtype=bool)
    for k in range(4):
        ax, ay = corners[k]
        bx, by = corners[(k + 1) % 4]
        cross = (bx - ax) * (py - ay) - (by - ay) * (px - ax)
        inside &= cross >= 0
    return inside

class TraceDistanceFilterStage(Stage):
    """Filter each bin's points to those within *distance* metres of the car
    trace (2-D XY).

    Each bin is compared only against the nearby trace segments (determined by
    the bin's position along the trace), keeping memory usage small.

    Reads  ``context.bins``, ``context.las``, ``context.trace``.
    Writes ``context.bins`` with filtered index arrays.
    """

    def __init__(self, distance: float = 20.0):
        super().__init__()
        if distance <= 0.0:
            raise ValueError("Distance should be positive.")
        self.distance = float(distance)

    def run(self, context):
        if context.bins is not None:
            num_bins = len(context.bins)
        
        # Pull full arrays once — avoids re-materialising ScaledArrayView per bin.
        xs = np.asarray(context.las.x, dtype=np.float64)
        ys = np.asarray(context.las.y, dtype=np.float64)
        global_mask = np.zeros(len(xs), dtype=bool)

        d = self.distance
        log_step = max(1, num_bins // 10)
        for i, bin in enumerate(context.bins):
            if i % log_step == 0 or i == num_bins - 1:
                self.logger.info(f"Bin {i + 1}/{num_bins} ({(i + 1) / num_bins * 100:.0f}%)")

            indices = bin.indices
            if indices.size == 0:
                continue

            c_s = np.array(bin.perp_S.coords)   # (2, 2)
            mid_s = c_s.mean(axis=0)
            unit_s = c_s[0] - mid_s
            unit_s /= np.linalg.norm(unit_s)

            c_e = np.array(bin.perp_E.coords)   # (2, 2)
            mid_e = c_e.mean(axis=0)
            unit_e = c_e[0] - mid_e
            unit_e /= np.linalg.norm(unit_e)

            corners = np.array([
                mid_s + unit_s * d,
                mid_s - unit_s * d,
                mid_e - unit_e * d,
                mid_e + unit_e * d,
            ])  # (4, 2), winding order: S-left, S-right, E-right, E-left

            mask = _in_quad(xs[indices], ys[indices], corners)
            global_mask[indices[mask]] = True
        
        context.prev_mask = context.global_mask
        context.global_mask = global_mask
