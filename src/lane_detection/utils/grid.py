import numpy as np


def build_grid(points: np.ndarray, square_size: float) -> dict[tuple[int, int], np.ndarray]:
    """Group point-indices by their (gx, gy) grid cell.

    *points* must be (N, >=2). Only the first two columns (x, y) are used.
    Returns ``{(gx, gy): indices}`` with ``indices`` an int64 ``np.ndarray``.
    """
    gx = np.floor(points[:, 0] / square_size).astype(np.int64)
    gy = np.floor(points[:, 1] / square_size).astype(np.int64)

    # Pack (gx, gy) into a single int64 key, sort, then split at boundaries
    gy_range = int(gy.max() - gy.min()) + 1
    keys = (gx - gx.min()) * gy_range + (gy - gy.min())

    order = np.argsort(keys, kind='stable')
    sorted_keys = keys[order]
    splits = np.flatnonzero(np.diff(sorted_keys)) + 1
    groups = np.split(order, splits)
    unique_keys = sorted_keys[np.concatenate(([0], splits))]

    gx_min = int(gx.min())
    gy_min = int(gy.min())
    return {
        (int(k // gy_range) + gx_min, int(k % gy_range) + gy_min): g
        for k, g in zip(unique_keys, groups)
    }
