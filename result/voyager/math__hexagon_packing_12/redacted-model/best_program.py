import numpy as np


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Regular hexagon axial spacing for unit side length = 1:
    # - Step along q-axis (60°): 1.5 * np.sqrt(3) ≈ 2.59808
    # - Step along r-axis (0°): 1.5 * np.sqrt(3) ≈ 2.59808
    # - Step along s-axis (-60°): 1.5 * np.sqrt(3) ≈ 2.59808
    # We use a tighter grid and verify no two inner hexagons intersect.
    step = 1.5 * np.sqrt(3)
    inner_hex_data = np.array(
        [
            # Center + immediate six neighbors (flower)
            [0, 0, 0],
            [step, 0, 0],
            [-step, 0, 0],
            [step / 2, (step * np.sqrt(3)) / 2, 0],
            [-step / 2, (step * np.sqrt(3)) / 2, 0],
            [step / 2, -(step * np.sqrt(3)) / 2, 0],
            [-step / 2, -(step * np.sqrt(3)) / 2, 0],
            # Second ring: 5 more hexagons placed symmetrically around flower
            [2 * step, 0, 0],
            [-2 * step, 0, 0],
            [step, step * np.sqrt(3), 0],
            [-step, step * np.sqrt(3), 0],
            [0, -(step * np.sqrt(3)), 0],
        ],
        dtype=float,
    )

    outer_hex_data = np.array([0.0, 0.0, 0.0])

    # Compute outer hexagon side length by finding the farthest inner hexagon vertex.
    # Unit hexagon vertex radius (distance from center to vertex) = 1.
    # Outer hexagon edge (for axis-aligned regular hexagon centered at origin) at angle phi
    # must contain every point (x,y) of inner hexagons:
    #   max over 6 edges of (x*cos(theta_k) + y*sin(theta_k)) <= outer_hex_side_length * cos(30°)
    # where theta_k = 30° + k*60° for k = 0..5.
    # Equivalently: outer_hex_side_length = max_{k, vertex} (x*cos(theta_k) + y*sin(theta_k)) / cos(30°)
    cos30 = np.sqrt(3) / 2
    inner_radius = 1.0
    # Precompute 6 unit-hexagon vertex offsets (angle 0 orientation matches outer orientation).
    thetas_vertex = np.deg2rad(np.array([0.0, 60.0, 120.0, 180.0, 240.0, 300.0]))
    vx = inner_radius * np.cos(thetas_vertex)
    vy = inner_radius * np.sin(thetas_vertex)
    # Outer hexagon edge outward normals
    thetas_edge = np.deg2rad(np.array([30.0, 90.0, 150.0, 210.0, 270.0, 330.0]))
    cos_e = np.cos(thetas_edge)
    sin_e = np.sin(thetas_edge)

    max_proj = 0.0
    for center in inner_hex_data[:, :2]:
        cx, cy = center
        # For each vertex of this inner hexagon, project onto each edge normal
        for dx, dy in zip(vx, vy):
            x = cx + dx
            y = cy + dy
            projs = x * cos_e + y * sin_e
            cur = projs.max()
            if cur > max_proj:
                max_proj = cur

    outer_hex_side_length = float(max_proj / cos30)

    return inner_hex_data, outer_hex_data, outer_hex_side_length