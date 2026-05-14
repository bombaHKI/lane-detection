import numpy as np

def scale_along_trace_mtx(S, E, scale: float = 1):
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
    M = S_mat @ R.T

    # inverse scaling
    S_inv = np.array([
        [1/scale, 0],
        [0, 1]
    ])

    M_inv = R @ S_inv
    return M, M_inv
