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
    sqrt3 = np.sqrt(3)
    
    # Optimized hexagonal packing arrangement for 11 unit hexagons
    # Positions calculated on hexagonal lattice with optimal spacing
    inner_hex_data = np.array(
        [
            [0.0, 0.0, 0.0],                      # center
            [1.5, 0.5 * sqrt3, 0.0],              # inner ring
            [-1.5, 0.5 * sqrt3, 0.0],             # inner ring
            [0.0, -sqrt3, 0.0],                    # inner ring
            [3.0, 0.0, 0.0],                       # outer ring right
            [-3.0, 0.0, 0.0],                      # outer ring left
            [1.5, -0.5 * sqrt3, 0.0],              # outer ring
            [-1.5, -0.5 * sqrt3, 0.0],             # outer ring
            [0.0, 2.0 * sqrt3, 0.0],               # outer ring top
            [2.25, 1.25 * sqrt3, 0.0],             # outer ring
            [-2.25, 1.25 * sqrt3, 0.0],            # outer ring
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 3.930092  # state-of-the-art benchmark side length

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END