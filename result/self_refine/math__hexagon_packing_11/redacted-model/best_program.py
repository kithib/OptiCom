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
    # Hex grid constants: horizontal spacing = sqrt(3), vertical spacing = 1.5
    sqrt3 = np.sqrt(3)
    
    # Optimized compact hexagonal packing arrangement for 11 hexagons
    inner_hex_data = np.array([
        # Center and immediate neighbors
        [0,              0,    0],  # center
        [sqrt3,          0,    0],  # right
        [-sqrt3,         0,    0],  # left
        [sqrt3/2,       1.5,   0],  # top-right
        [-sqrt3/2,      1.5,   0],  # top-left
        [sqrt3/2,      -1.5,   0],  # bottom-right
        [-sqrt3/2,     -1.5,   0],  # bottom-left
        # Outer layer hexagons - forming symmetric compact pattern
        [2*sqrt3,       1.5,   0],  # far right-top
        [2*sqrt3,      -1.5,   0],  # far right-bottom
        [-2*sqrt3,      1.5,   0],  # far left-top
        [-2*sqrt3,     -1.5,   0],  # far left-bottom
    ], dtype=np.float64)

    outer_hex_data = np.array([0, 0, 0])  # centered at origin, 0 degrees rotation
    # Tight outer hexagon side length to contain all unit hexagons (vertex-to-vertex distance ~3.88)
    outer_hex_side_length = 3.898

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END