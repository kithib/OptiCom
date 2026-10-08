# EVOLVE-BLOCK-START
import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Hexagonal lattice constants for unit hexagons (side length 1)
    dx = 1.5  # Horizontal distance between centers (1.5 * side)
    dy = np.sqrt(3) / 2  # Vertical distance between rows (sqrt(3)/2 * side)
    
    # Optimized arrangement in a tighter hexagonal lattice pattern with proper alignment
    inner_hex_data = np.array([
        # Row y = +2*dy (top row)
        [-dx, 4*dy, 0],
        [dx, 4*dy, 0],
        # Row y = +dy
        [-2*dx, 2*dy, 0],
        [0, 2*dy, 0],
        [2*dx, 2*dy, 0],
        # Row y = 0 (middle row)
        [-3*dx, 0, 0],
        [-dx, 0, 0],
        [dx, 0, 0],
        [3*dx, 0, 0],
        # Row y = -dy
        [-2*dx, -2*dy, 0],
        [0, -2*dy, 0],
        [2*dx, -2*dy, 0],
    ], dtype=np.float64)

    outer_hex_data = np.array([0, 0, 0], dtype=np.float64)  # centered at origin
    # Calculate minimal outer hexagon side length based on actual extent:
    # - Rightmost hex center at +3dx with unit side requires +1 clearance
    # - Horizontal extent for outer hexagon: 3dx + 1 = 5.5
    # - Proper clearance gives tight valid containment
    outer_hex_side_length = 5.5

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END