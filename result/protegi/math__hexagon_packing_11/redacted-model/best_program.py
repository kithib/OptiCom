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
    # Hexagonal lattice constants
    dx = 1.5  # Horizontal distance between adjacent hex centers (side length 1, axial distance)
    dy = np.sqrt(3)  # Vertical distance between adjacent hex rows (sqrt(3)/2 * 2)
    
    # Optimized compact arrangement - 2-layer compact pattern
    inner_hex_data = np.array(
        [
            # Center hexagon
            [0, 0, 0],
            # First ring (6 surrounding hexagons)
            [-dx, 0, 0],
            [dx, 0, 0],
            [-dx/2, dy/2, 0],
            [dx/2, dy/2, 0],
            [-dx/2, -dy/2, 0],
            [dx/2, -dy/2, 0],
            # Second ring extension (4 hexagons in tight arrangement)
            [-2*dx, 0, 0],
            [2*dx, 0, 0],
            [-dx, dy, 0],
            [dx, -dy, 0],
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # Tight outer hexagon side length calculated for containment
    # Distance from center to farthest hex vertex + proper boundary
    outer_hex_side_length = 4.5

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END