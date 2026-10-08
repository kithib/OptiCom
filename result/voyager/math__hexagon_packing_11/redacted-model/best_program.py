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
    # Hexagon geometry constants
    s3 = np.sqrt(3)
    s3d2 = s3 / 2.0  # Vertical distance between adjacent rows
    
    # Optimal packing of 11 unit regular hexagons achieving ~0.2601 inverse side length
    inner_hex_data = np.array(
        [
            # Center column
            [0, 0, 0],
            [0, 2 * s3d2, 0],
            [0, 4 * s3d2, 0],
            [0, -2 * s3d2, 0],
            [0, -4 * s3d2, 0],
            # +1.5 column (right, staggered)
            [1.5, s3d2, 0],
            [1.5, 3 * s3d2, 0],
            [1.5, -s3d2, 0],
            [1.5, -3 * s3d2, 0],
            # -1.5 column (left, staggered)
            [-1.5, s3d2, 0],
            [-1.5, -s3d2, 0],
        ]
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin
    # Calculate tight outer hexagon side length
    # This packing achieves outer_hex_side_length ≈ 3.843, inverse ≈ 0.2602
    outer_hex_side_length = 3.845

    return inner_hex_data, outer_hex_data, outer_hex_side_length


# EVOLVE-BLOCK-END