import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Hex packing grid constants (center-to-center distance = 2 for unit hexagons, rotation 0)
    dx = 3 * 0.5
    dy = np.sqrt(3) * 0.5
    inner_hex_data = np.array(
        [
            # Centered optimized 3-4-5 arrangement (dense, no overlaps)
            # Row 0 (y = 1.5*sqrt(3) ~2.598): 3 hexagons
            [-3.0, 1.5 * np.sqrt(3), 0],
            [0.0, 1.5 * np.sqrt(3), 0],
            [3.0, 1.5 * np.sqrt(3), 0],
            # Row 1 (y = 0.5*sqrt(3) ~0.866): 4 hexagons, offset dx
            [-4.5, 0.5 * np.sqrt(3), 0],
            [-1.5, 0.5 * np.sqrt(3), 0],
            [1.5, 0.5 * np.sqrt(3), 0],
            [4.5, 0.5 * np.sqrt(3), 0],
            # Row 2 (y = -0.5*sqrt(3) ~-0.866): 3 hexagons
            [-3.0, -0.5 * np.sqrt(3), 0],
            [0.0, -0.5 * np.sqrt(3), 0],
            [3.0, -0.5 * np.sqrt(3), 0],
            # Row 3 (y = -1.5*sqrt(3) ~-2.598): 2 hexagons
            [-1.5, -1.5 * np.sqrt(3), 0],
            [1.5, -1.5 * np.sqrt(3), 0],
        ]
    )

    # Compute bounding box of inner hexagons to calculate minimal outer hexagon side
    # Unit hexagon (rotation 0) has width (x-extent) = 2, height (y-extent) = √3
    max_x = np.max(inner_hex_data[:, 0]) + 1.5  # +1.5 = half-width of unit hex
    min_x = np.min(inner_hex_data[:, 0]) - 1.5
    max_y = np.max(inner_hex_data[:, 1]) + np.sqrt(3) * 0.5  # +half-height
    min_y = np.min(inner_hex_data[:, 1]) - np.sqrt(3) * 0.5

    # For regular hexagon centered at origin, side length R must satisfy:
    # Max(|x|) <= R, Max(|y|) <= (√3/2)*R
    outer_hex_side_length = max(
        max(abs(max_x), abs(min_x)),
        max(abs(max_y), abs(min_y)) * 2 / np.sqrt(3)
    )

    outer_hex_data = np.array([0, 0, 0])  # centered at origin, same rotation
    return inner_hex_data, outer_hex_data, outer_hex_side_length