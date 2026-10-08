import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Core constants for flat-top orientation inner unit hexagons (side length = 1)
    # - Circumradius (center to vertex): 1.0
    # - Center-to-center distance for disjoint packing: >= sqrt(3) ≈ 1.732
    # - Hexagonal grid spacing: horizontal = 1.5, vertical = sqrt(3)
    H_SPACING = 1.5
    V_SPACING = np.sqrt(3)

    inner_hex_data = np.array(
        [
            # Row 0 (y = 2*V_SPACING): 2 hexagons
            [-1.5, 2 * V_SPACING, 0],
            [1.5, 2 * V_SPACING, 0],
            # Row 1 (y = V_SPACING): 3 hexagons
            [-3.0, V_SPACING, 0],
            [0.0, V_SPACING, 0],
            [3.0, V_SPACING, 0],
            # Row 2 (y = 0): 2 hexagons
            [-1.5, 0.0, 0],
            [1.5, 0.0, 0],
            # Row 3 (y = -V_SPACING): 3 hexagons
            [-3.0, -V_SPACING, 0],
            [0.0, -V_SPACING, 0],
            [3.0, -V_SPACING, 0],
            # Row 4 (y = -2*V_SPACING): 2 hexagons
            [-1.5, -2 * V_SPACING, 0],
            [1.5, -2 * V_SPACING, 0],
        ]
    )

    outer_hex_data = np.array([0.0, 0.0, 0.0])  # centered at origin, same orientation

    # Compute vertices of inner hexagons to determine tight bounding outer hexagon
    # Flat-top unit hexagon vertex angles (relative to center): 0, 60, 120, 180, 240, 300 degrees
    vertex_angles = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
    vx = np.cos(vertex_angles)
    vy = np.sin(vertex_angles)

    # Rotate vertices by inner hexagon angle (all 0 here, but handled generally)
    cos_a = np.cos(np.deg2rad(inner_hex_data[:, 2]))
    sin_a = np.sin(np.deg2rad(inner_hex_data[:, 2]))

    # Transformed vertices: (x + vx*cos_a - vy*sin_a, y + vx*sin_a + vy*cos_a)
    all_x = inner_hex_data[:, 0:1] + (vx * cos_a[:, None] - vy * sin_a[:, None])
    all_y = inner_hex_data[:, 1:2] + (vx * sin_a[:, None] + vy * cos_a[:, None])

    max_abs_x = np.max(np.abs(all_x))
    max_abs_y = np.max(np.abs(all_y))

    # Outer hexagon (flat-top, angle 0) side length R constraints:
    # - Max |x| over outer vertices = R * (sqrt(3)/2)
    # - Max |y| over outer vertices = R * (sqrt(3)/2)
    # Thus R >= max(max_abs_x, max_abs_y) * 2 / sqrt(3)
    inv_sqrt3_half = 2.0 / np.sqrt(3)
    outer_hex_side_length = max(max_abs_x, max_abs_y) * inv_sqrt3_half

    return inner_hex_data, outer_hex_data, outer_hex_side_length