# EVOLVE-BLOCK-START
import numpy as np

SQRT3 = np.sqrt(3)
SQRT3_DIV2 = SQRT3 / 2.0

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    inner_hex_data = np.array([
        [0.0, 0.0, 0],                      # Center
        [1.5, 0.0, 0],                      # Right of center
        [-1.5, 0.0, 0],                     # Left of center
        [0.75, SQRT3_DIV2 * 3, 0],          # Top (3 units up in hex grid: 3 * sqrt(3)/2)
        [0.75, -SQRT3_DIV2 * 3, 0],         # Bottom
        [2.25, SQRT3_DIV2, 0],              # Right-top
        [2.25, -SQRT3_DIV2, 0],             # Right-bottom
        [-2.25, SQRT3_DIV2, 0],             # Left-top
        [-2.25, -SQRT3_DIV2, 0],            # Left-bottom
        [3.0, 0.0, 0],                       # Far right
        [-3.0, 0.0, 0],                      # Far left
    ])

    outer_hex_data = np.array([0.0, 0.0, 0])
    # Calculate tight outer hex side: far left/right at +/-3.0, outer hex radius 4.0
    outer_hex_side_length = 4.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length
# EVOLVE-BLOCK-END