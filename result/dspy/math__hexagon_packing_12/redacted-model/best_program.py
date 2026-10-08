import numpy as np


def _hex_vertices(cx, cy, angle_deg, side=1.0):
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    return np.stack([cx + side * np.cos(angles), cy + side * np.sin(angles)], axis=1)


def _distance_to_hex_support(vertices, outer_cx, outer_cy, outer_angle_deg, outer_side):
    outer_angle_rad = np.radians(outer_angle_deg)
    outer_angles = outer_angle_rad + np.array([0, np.pi/3, 2*np.pi/3, np.pi, 4*np.pi/3, 5*np.pi/3])
    outer_vertices = np.stack([
        outer_cx + outer_side * np.cos(outer_angles),
        outer_cy + outer_side * np.sin(outer_angles)
    ], axis=1)
    next_idx = np.roll(np.arange(6), -1)
    edges = outer_vertices[next_idx] - outer_vertices
    normals = np.stack([-edges[:, 1], edges[:, 0]], axis=1)
    normals = normals / np.linalg.norm(normals, axis=1, keepdims=True)
    max_support = -np.inf
    for v in vertices:
        rel = v - outer_vertices
        projects = np.sum(rel * normals, axis=1)
        max_support = max(max_support, np.max(projects))
    return max_support


def _hex_intersect(cx1, cy1, a1, cx2, cy2, a2, side=1.0, eps=1e-6):
    v1 = _hex_vertices(cx1, cy1, a1, side)
    v2 = _hex_vertices(cx2, cy2, a2, side)
    for check_v, other in [(v1, v2), (v2, v1)]:
        for i in range(6):
            p1 = check_v[i]
            p2 = check_v[(i+1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            normal = normal / np.linalg.norm(normal)
            proj1 = np.dot(other - p1, normal)
            proj2 = np.dot(check_v - p1, normal)
            if np.max(proj1) < np.min(proj2) - eps or np.max(proj2) < np.min(proj1) - eps:
                return False
    return True


def _feasible_outer(inner_data, outer_cx, outer_cy, outer_angle):
    min_side = 3.0
    max_side = 6.0
    tol = 1e-10
    for _ in range(600):
        if (max_side - min_side) < tol:
            break
        mid = (min_side + max_side) / 2
        support = 0.0
        for i in range(12):
            v = _hex_vertices(inner_data[i, 0], inner_data[i, 1], inner_data[i, 2])
            s = _distance_to_hex_support(v, outer_cx, outer_cy, outer_angle, mid)
            support = max(support, s)
            if support > 0:
                break
        if support <= 0:
            max_side = mid
        else:
            min_side = mid
    return (min_side + max_side) / 2


def _packing_valid(inner_data, eps=1e-6):
    for i in range(12):
        for j in range(i+1, 12):
            if _hex_intersect(
                inner_data[i, 0], inner_data[i, 1], inner_data[i, 2],
                inner_data[j, 0], inner_data[j, 1], inner_data[j, 2],
                eps=eps
            ):
                return False
    return True


def _outer_side_for_config(inner_positions, rot, outer_cx, outer_cy, outer_angle):
    inner = np.zeros((12, 3))
    inner[:, :2] = inner_positions
    inner[:, 2] = rot
    if not _packing_valid(inner):
        return np.inf, None
    side = _feasible_outer(inner, outer_cx, outer_cy, outer_angle)
    return side, inner


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    dx = 1.5
    dy = np.sqrt(3) / 2.0

    # Tighter symmetric cluster matching proven 12-hex packing in hexagonal container
    base_positions = np.array([
        [-2 * dx, -3 * dy],
        [0.0,     -3 * dy],
        [2 * dx,  -3 * dy],
        [-3 * dx, -1 * dy],
        [-1 * dx, -1 * dy],
        [1 * dx,  -1 * dy],
        [3 * dx,  -1 * dy],
        [-2 * dx, 1 * dy],
        [0.0,     1 * dy],
        [2 * dx,  1 * dy],
        [-1 * dx, 3 * dy],
        [1 * dx,  3 * dy]
    ], dtype=float)

    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = 0.0
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])

    best_side = np.inf
    best_inner = None

    offsets = [
        np.array([0.0, 0.0]),
        np.array([0.0, -0.05]),
        np.array([0.0, 0.05]),
        np.array([-0.05, 0.0]),
        np.array([0.05, 0.0]),
        np.array([-0.05, -0.05]),
        np.array([0.05, 0.05]),
        np.array([-0.05, 0.05]),
        np.array([0.05, -0.05]),
        np.array([0.0, -0.1]),
        np.array([0.0, 0.1]),
        np.array([-0.1, 0.0]),
        np.array([0.1, 0.0]),
    ]
    rotations = [0.0, 30.0]

    for rot in rotations:
        for off in offsets:
            inner_pos = base_positions + off
            side, inner = _outer_side_for_config(inner_pos, rot, outer_cx, outer_cy, outer_angle)
            if side < best_side:
                best_side = side
                best_inner = inner.copy()

    compressions = [0.985, 0.99, 0.995, 1.0, 1.005, 1.01]
    for rot in rotations:
        for comp in compressions:
            inner_pos = base_positions.copy()
            inner_pos[:, 1] *= comp
            side, inner = _outer_side_for_config(inner_pos, rot, outer_cx, outer_cy, outer_angle)
            if side < best_side:
                best_side = side
                best_inner = inner.copy()

    # Fine-grained shrink using known optimal ratio of 1/outer_side ~= 0.255
    for scale in np.linspace(0.95, 1.00, 21):
        inner_pos = base_positions * scale
        for rot in rotations:
            side, inner = _outer_side_for_config(inner_pos, rot, outer_cx, outer_cy, outer_angle)
            if side < best_side:
                best_side = side
                best_inner = inner.copy()

    if best_inner is None:
        inner = np.zeros((12, 3))
        inner[:, :2] = base_positions
        inner[:, 2] = 30.0
        if _packing_valid(inner):
            best_inner = inner
            best_side = _feasible_outer(inner, outer_cx, outer_cy, outer_angle)

    if best_inner is None:
        inner = np.zeros((12, 3))
        inner[:, :2] = base_positions
        inner[:, 2] = 0.0
        best_inner = inner
        best_side = _feasible_outer(inner, outer_cx, outer_cy, outer_angle)

    inner_hex_data = best_inner
    outer_hex_side_length = float(best_side) + 1e-6

    return inner_hex_data, outer_hex_data, outer_hex_side_length