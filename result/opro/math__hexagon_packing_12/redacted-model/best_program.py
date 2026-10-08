# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def get_hex_vertices(x, y, angle_deg, side_length=1):
    """Get vertices of a regular hexagon."""
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + np.pi / 3 * i
        vx = x + side_length * np.cos(theta)
        vy = y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return np.array(vertices)


def hex_within_outer(inner_vertices, outer_R):
    """Check if all inner hexagon vertices lie within outer hexagon (centered at origin, 0 rotation)."""
    max_violation = 0.0
    for (vx, vy) in inner_vertices:
        for i in range(6):
            theta = np.pi / 3 * i + np.pi / 6
            normal = np.array([np.cos(theta), np.sin(theta)])
            proj = vx * normal[0] + vy * normal[1]
            limit = outer_R * np.sqrt(3) / 2
            violation = proj - limit
            if violation > max_violation:
                max_violation = violation
    return max_violation


def hex_separation(xyza1, xyza2):
    """Compute non-overlap constraint (positive = separated, negative = overlapping)."""
    x1, y1, a1 = xyza1
    x2, y2, a2 = xyza2
    dx = x2 - x1
    dy = y2 - y1
    dist = np.sqrt(dx ** 2 + dy ** 2)
    if dist < 1e-8:
        return -10.0
    dir_vec = np.array([dx, dy]) / dist
    proj1 = 0.0
    proj2 = 0.0
    for i in range(6):
        angle1 = np.radians(a1) + i * np.pi / 3
        edge_normal1 = np.array([np.cos(angle1 + np.pi / 2), np.sin(angle1 + np.pi / 2)])
        proj1 = max(proj1, np.abs(np.dot(dir_vec, edge_normal1)))
        angle2 = np.radians(a2) + i * np.pi / 3
        edge_normal2 = np.array([np.cos(angle2 + np.pi / 2), np.sin(angle2 + np.pi / 2)])
        proj2 = max(proj2, np.abs(np.dot(dir_vec, edge_normal2)))
    min_required = (proj1 + proj2) / 2.0 * np.sqrt(3)
    return dist - min_required


def all_contained(inner_hex_data, R):
    """Check all inner hexagons are contained in outer hexagon of side R."""
    for i in range(12):
        x, y, angle = inner_hex_data[i]
        verts = get_hex_vertices(x, y, angle, 1)
        if hex_within_outer(verts, R) > 1e-6:
            return False
    return True


def no_overlaps(inner_hex_data):
    """Check no inner hexagons overlap."""
    for i in range(12):
        for j in range(i + 1, 12):
            sep = hex_separation(inner_hex_data[i], inner_hex_data[j])
            if sep < -1e-6:
                return False
    return True


def packing_objective(params):
    """Objective to minimize: outer hex side length + constraint penalties."""
    outer_R = params[-1]
    penalty = 0.0
    for i in range(12):
        x, y, a = params[i * 3], params[i * 3 + 1], params[i * 3 + 2]
        verts = get_hex_vertices(x, y, a)
        violation = hex_within_outer(verts, outer_R)
        if violation > 0:
            penalty += 1000 * violation ** 2
    for i in range(12):
        for j in range(i + 1, 12):
            hex1 = params[i * 3:i * 3 + 3]
            hex2 = params[j * 3:j * 3 + 3]
            sep = hex_separation(hex1, hex2)
            if sep < 0:
                penalty += 1000 * sep ** 2
    return outer_R + penalty


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    # Initial guess based on near-optimal hexagonal packing arrangement
    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 1.732, 0.0],
        [0.0, -1.732, 0.0],
        [1.5, 0.866, 0.0],
        [1.5, -0.866, 0.0],
        [-1.5, 0.866, 0.0],
        [-1.5, -0.866, 0.0],
        [3.0, 0.0, 0.0],
        [-3.0, 0.0, 0.0],
        [2.0, 2.598, 0.0],
        [-2.0, 2.598, 0.0],
        [0.0, 3.464, 0.0]
    ])
    
    init_params = np.concatenate([inner_hex_data.flatten(), [5.0]])
    bounds = []
    for i in range(12):
        bounds.extend([(-6, 6), (-6, 6), (0, 180)])
    bounds.append((3.5, 6.0))
    result = minimize(packing_objective, init_params, method='L-BFGS-B', bounds=bounds, options={'maxiter': 1000, 'ftol': 1e-9})
    opt_params = result.x
    optimized_inner = opt_params[:36].reshape(12, 3)
    optimized_R = opt_params[-1]
    
    if not (all_contained(optimized_inner, optimized_R) and no_overlaps(optimized_inner)):
        optimized_inner = inner_hex_data
        optimized_R = 5.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return optimized_inner.astype(float), outer_hex_data.astype(float), float(optimized_R)


# EVOLVE-BLOCK-END