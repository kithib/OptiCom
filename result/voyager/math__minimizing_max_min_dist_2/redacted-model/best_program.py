import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2

    # 4x4 regular grid arrangement in [0,1] x [0,1]
    # This provides equal spacing between adjacent points and maximizes dmin/dmax
    grid_size = int(np.sqrt(n))
    x_coords = np.linspace(0, 1, grid_size)
    y_coords = np.linspace(0, 1, grid_size)
    xx, yy = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xx.ravel(), yy.ravel()])

    return points