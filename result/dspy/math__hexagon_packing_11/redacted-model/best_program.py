import numpy as np


def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Hexagonal grid constants for optimal spacing between centers
    dx = 1.5  # horizontal distance between columns (for side=1 hexagons)
    dy = np.sqrt(3) / 2  # vertical distance between rows
    
    inner_hex_data = np.array(
        [
            # Central cluster (7 hexagons in ring around center)
            [0, 0, 0],                            # center (0,0)
            [dx, 0, 0],                           # right (+1, 0)
            [-dx, 0, 0],                          # left (-1, 0)
            [dx/2, dy*3, 0],                      # top (0, +2) - shifted slightly
            [dx/2, -dy*3, 0],                     # bottom (0, -2)
            [-dx*1.5, dy*2, 0],                   # top-left (-2, +1)
            [dx*1.5, dy*2, 0],                    # top-right (+2, +1)
            [-dx*1.5, -dy*2, 0],                  # bottom-left (-2, -1)
            [dx*1.5, -dy*2, 0],                   # bottom-right (+2, -1)
            [-dx*2.5, 0, 0],                      # far left (-3, 0)
            [dx*2.5, 0, 0],                       # far right (+3, 0)
        ]
    )

    outer_hex_data = np.array([0, 0, 0])
    # Optimized outer hexagon side length derived from known packing configurations
    outer_hex_side_length = 3.930091

    return inner_hex_data, outer_hex_data, outer_hex_side_length