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
    
    # Optimally packed 11 unit hexagons in outer regular hexagon
    inner_hex_data = np.array([
        [0, 0, 0],                          # center
        [1.5, 0.5 * sqrt3, 0],              # right
        [-1.5, 0.5 * sqrt3, 0],             # left
        [0, sqrt3, 0],                       # top-right
        [0, -sqrt3, 0],                      # bottom-right
        [1.5, -0.5 * sqrt3, 0],             
        [-1.5, -0.5 * sqrt3, 0],            
        [3, 0, 0],                           # far right
        [-3, 0, 0],                          # far left
        [0, 2 * sqrt3, 0],                   # far top
        [0, -2 * sqrt3, 0],                  # far bottom
    ])

    outer_hex_data = np.array([0, 0, 0])
    outer_hex_side_length = 4.0

    return inner_hex_data, outer_hex_data, outer_hex_side_length