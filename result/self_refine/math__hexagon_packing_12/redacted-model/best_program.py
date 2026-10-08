import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12
    sqrt3 = np.sqrt(3)
    dx = 1.5
    inner_hex_data = np.array(
        [
            # Compact symmetric offset-grid placement
            [-2 * dx, 1 * sqrt3, 0],
            [0 * dx, 1 * sqrt3, 0],
            [2 * dx, 1 * sqrt3, 0],
            [-3 * dx, 0 * sqrt3, 0],
            [-1 * dx, 0 * sqrt3, 0],
            [1 * dx, 0 * sqrt3, 0],
            [3 * dx, 0 * sqrt3, 0],
            [-2 * dx, -1 * sqrt3, 0],
            [0 * dx, -1 * sqrt3, 0],
            [2 * dx, -1 * sqrt3, 0],
            [-1 * dx, -2 * sqrt3, 0],
            [1 * dx, -2 * sqrt3, 0],
        ],
        dtype=float
    )

    # Proper unit hex vertex calculation with edge length = 1
    thetas = np.deg2rad(np.arange(0, 360, 60))
    unit_hex_vertices = np.column_stack([np.cos(thetas), np.sin(thetas)])
    
    # Correct containment calculation for a regular hexagon with edge length R
    max_extent = 0.0
    for i in range(n):
        cx, cy, ang_deg = inner_hex_data[i]
        ang_rad = np.deg2rad(ang_deg)
        cos_a = np.cos(ang_rad)
        sin_a = np.sin(ang_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        world_verts = unit_hex_vertices @ rot.T + np.array([cx, cy])
        for v in world_verts:
            x, y = v
            # Required edge length R to contain this vertex:
            # R must satisfy:
            #   x + y / sqrt(3) <= R
            #   x - y / sqrt(3) <= R
            #   -x + y / sqrt(3) <= R
            #   -x - y / sqrt(3) <= R
            #   y <= (sqrt(3)/2) R -> 2y / sqrt(3) <= R
            #   -y <= (sqrt(3)/2) R -> -2y / sqrt(3) <= R
            constraints = np.array([
                x + y / sqrt3,
                x - y / sqrt3,
                -x + y / sqrt3,
                -x - y / sqrt3,
                2 * y / sqrt3,
                -2 * y / sqrt3
            ])
            current_max = np.max(constraints)
            if current_max > max_extent:
                max_extent = current_max

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    # Use exact containment (remove safety margin to maximize objective)
    outer_hex_side_length = float(max_extent)

    return inner_hex_data, outer_hex_data, outer_hex_side_length