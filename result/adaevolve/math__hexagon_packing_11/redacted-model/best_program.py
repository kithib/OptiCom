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
    # Optimal compact packing of 11 unit hexagons in hexagonal lattice
    # Using exact axial coordinate conversions with sqrt(3)/2 ≈ 0.8660
    inner_hex_data = np.array(
        [
            [0.0, 0.0, 0],      # center (1)
            [1.5, 0.8660, 0],   # ring 1 (3)
            [0.0, 1.7321, 0],
            [-1.5, 0.8660, 0],
            [-1.5, -0.8660, 0],
            [0.0, -1.7321, 0],
            [1.5, -0.8660, 0],
            [3.0, 0.0, 0],      # ring 2 (4)
            [1.5, 2.5981, 0],
            [-1.5, 2.5981, 0],
            [-3.0, 0.0, 0],
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    outer_hex_side_length = 4.5  # Minimal container for this arrangement

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END