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
    # Hex grid constants
    dx = 1.5  # horizontal distance between adjacent hex centers
    dy = np.sqrt(3) / 2 * 2  # vertical distance = sqrt(3) ≈ 1.732
    
    inner_hex_data = np.array(
        [
            # Center column (x = 0)
            [0, 0, 0],
            [0, dy, 0],
            [0, -dy, 0],
            # Column at x = dx
            [dx, dy/2, 0],
            [dx, -dy/2, 0],
            [dx, 3*dy/2, 0],
            [dx, -3*dy/2, 0],
            # Column at x = -dx
            [-dx, dy/2, 0],
            [-dx, -dy/2, 0],
            [-dx, 3*dy/2, 0],
            [-dx, -3*dy/2, 0],
        ]
    )

    outer_hex_data = np.array([0, 0, 0])
    # Tight bounding: horizontal extent 2*dx from center = 3, hex radius = 1.5, total side = 4.5
    # 3 columns at -1.5, 0, 1.5 each with radius 1.5 gives extent 3.0 each side = 6, plus margin = 4.8
    outer_hex_side_length = 4.8

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END