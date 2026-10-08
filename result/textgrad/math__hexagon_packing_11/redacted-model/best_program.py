# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 11
    sqrt3 = np.sqrt(3)
    
    # Improved compact hexagonal lattice arrangement
    inner_hex_data = np.array(
        [
            [0, 0, 0],                 # center
            [-1.5, 0.5 * sqrt3, 0],    # hex ring (row -1, col -1)
            [0, sqrt3, 0],              # hex ring (row -1, col 0)
            [1.5, 0.5 * sqrt3, 0],      # hex ring (row -1, col 1)
            [-1.5, -0.5 * sqrt3, 0],    # hex ring (row 1, col -1)
            [0, -sqrt3, 0],              # hex ring (row 1, col 0)
            [1.5, -0.5 * sqrt3, 0],      # hex ring (row 1, col 1)
            [-3, 0, 0],                  # row 0, col -2 (left extension)
            [3, 0, 0],                   # row 0, col 2 (right extension)
            [-2.25, (3 * sqrt3) / 2, 0],  # row -2, col -1 (upper-left)
            [2.25, -(3 * sqrt3) / 2, 0],  # row 2, col 1 (lower-right)
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 4.2  # significantly tighter containment

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END