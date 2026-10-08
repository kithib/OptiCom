import numpy as np
from scipy.optimize import minimize, basinhopping


def _hex_vertices(cx, cy, side, angle_deg):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for i in range(6):
        theta = angle_rad + i * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        vertices.append([x, y])
    return np.array(vertices)


def _project_polygon(axis, vertices):
    dots = vertices @ axis
    return np.min(dots), np.max(dots)


def _polygons_intersect(vertices_a, vertices_b):
    n_a = len(vertices_a)
    n_b = len(vertices_b)
    for i in range(n_a):
        edge = vertices_a[(i + 1) % n_a] - vertices_a[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        min_a, max_a = _project_polygon(axis, vertices_a)
        min_b, max_b = _project_polygon(axis, vertices_b)
        if max_a < min_b - 1e-6 or max_b < min_a - 1e-6:
            return False
    for i in range(n_b):
        edge = vertices_b[(i + 1) % n_b] - vertices_b[i]
        axis = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(axis)
        if norm < 1e-9:
            continue
        axis = axis / norm
        min_a, max_a = _project_polygon(axis, vertices_a)
        min_b, max_b = _project_polygon(axis, vertices_b)
        if max_a < min_b - 1e-6 or max_b < min_a - 1e-6:
            return False
    return True


def _vertices_in_hex(inner_vertices, outer_cx, outer_cy, outer_side, outer_angle_deg):
    outer_vertices = _hex_vertices(outer_cx, outer_cy, outer_side, outer_angle_deg)
    n_outer = len(outer_vertices)
    for v in inner_vertices:
        inside = True
        for i in range(n_outer):
            p1 = outer_vertices[i]
            p2 = outer_vertices[(i + 1) % n_outer]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            center_vec = np.array([outer_cx, outer_cy]) - p1
            if normal @ center_vec < 0:
                normal = -normal
            point_vec = v - p1
            if normal @ point_vec < -1e-6:
                inside = False
                break
        if not inside:
            return False
    return True


def _packing_valid(positions, inner_angles, outer_angle_deg, outer_side_target):
    for i in range(12):
        vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, inner_angles[i])
        if not _vertices_in_hex(vi, 0.0, 0.0, outer_side_target, outer_angle_deg):
            return False
    for i in range(11):
        for j in range(i + 1, 12):
            vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, inner_angles[i])
            vj = _hex_vertices(positions[j, 0], positions[j, 1], 1.0, inner_angles[j])
            if _polygons_intersect(vi, vj):
                return False
    return True


def _min_enclosing_hex_side(positions, inner_angles):
    max_side = 0.0
    for i in range(12):
        vi = _hex_vertices(positions[i, 0], positions[i, 1], 1.0, inner_angles[i])
        for v in vi:
            r = np.sqrt(v[0] ** 2 + v[1] ** 2)
            side = r * 2.0 / np.sqrt(3.0)
            if side > max_side:
                max_side = side
    return max_side


def _objective(params, inner_angles):
    positions = params.reshape(12, 2)
    return _min_enclosing_hex_side(positions, inner_angles)


def _fix_overlaps(positions, inner_angles, outer_angle_deg, outer_side, max_iter=80):
    pos = positions.copy()
    for _ in range(max_iter):
        overlap = False
        for i in range(11):
            for j in range(i + 1, 12):
                vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, inner_angles[i])
                vj = _hex_vertices(pos[j, 0], pos[j, 1], 1.0, inner_angles[j])
                if _polygons_intersect(vi, vj):
                    overlap = True
                    dx = pos[i, 0] - pos[j, 0]
                    dy = pos[i, 1] - pos[j, 1]
                    dist = np.hypot(dx, dy)
                    if dist < 1e-6:
                        dx, dy = 0.1, 0.1
                        dist = np.hypot(dx, dy)
                    pos[i, 0] += 0.14 * dx / dist
                    pos[i, 1] += 0.14 * dy / dist
                    pos[j, 0] -= 0.14 * dx / dist
                    pos[j, 1] -= 0.14 * dy / dist
        if not overlap:
            break
    for i in range(12):
        for _ in range(30):
            vi = _hex_vertices(pos[i, 0], pos[i, 1], 1.0, inner_angles[i])
            if _vertices_in_hex(vi, 0.0, 0.0, outer_side, outer_angle_deg):
                break
            dist = np.hypot(pos[i, 0], pos[i, 1])
            if dist > 1e-6:
                pos[i, 0] *= 0.94
                pos[i, 1] *= 0.94
    return pos


def _get_candidate_0():
    sqrt3 = np.sqrt(3.0)
    return np.array([
        [0.0, 0.0],
        [-1.5 * sqrt3, 0.0],
        [1.5 * sqrt3, 0.0],
        [-0.75 * sqrt3, 2.25],
        [0.75 * sqrt3, 2.25],
        [-0.75 * sqrt3, -2.25],
        [0.75 * sqrt3, -2.25],
        [-2.25 * sqrt3, 1.5],
        [2.25 * sqrt3, 1.5],
        [-2.25 * sqrt3, -1.5],
        [2.25 * sqrt3, -1.5],
        [0.0, -4.0],
    ])


def _get_candidate_1():
    sqrt3 = np.sqrt(3.0)
    return np.array([
        [0.0, 0.0],
        [-1.5 * sqrt3, 0.5],
        [1.5 * sqrt3, 0.5],
        [-0.75 * sqrt3, 2.25],
        [0.75 * sqrt3, 2.25],
        [-1.5 * sqrt3, -2.5],
        [1.5 * sqrt3, -2.5],
        [-2.75 * sqrt3, 1.0],
        [2.75 * sqrt3, 1.0],
        [-2.75 * sqrt3, -1.5],
        [2.75 * sqrt3, -1.5],
        [0.0, -3.8],
    ])


def _get_candidate_2():
    sqrt3 = np.sqrt(3.0)
    return np.array([
        [0.0, 0.0],
        [-1.5 * sqrt3, 0.0],
        [1.5 * sqrt3, 0.0],
        [-0.75 * sqrt3, 2.25],
        [0.75 * sqrt3, 2.25],
        [-0.75 * sqrt3, -2.25],
        [0.75 * sqrt3, -2.25],
        [-2.25 * sqrt3, 2.25],
        [2.25 * sqrt3, 2.25],
        [-2.25 * sqrt3, -2.25],
        [2.25 * sqrt3, -2.25],
        [0.0, -4.5],
    ])


def _get_candidate_3():
    sqrt3 = np.sqrt(3.0)
    return np.array([
        [0.0, 0.0],
        [-1.6 * sqrt3, 0.3],
        [1.6 * sqrt3, 0.3],
        [-0.8 * sqrt3, 2.35],
        [0.8 * sqrt3, 2.35],
        [-0.8 * sqrt3, -2.35],
        [0.8 * sqrt3, -2.35],
        [-2.4 * sqrt3, 1.2],
        [2.4 * sqrt3, 1.2],
        [-2.4 * sqrt3, -1.2],
        [2.4 * sqrt3, -1.2],
        [0.0, -4.2],
    ])


def _get_candidate_4():
    sqrt3 = np.sqrt(3.0)
    return np.array([
        [-0.3, 0.2],
        [-1.8 * sqrt3, 0.3],
        [1.8 * sqrt3, 0.3],
        [-0.8 * sqrt3, 2.4],
        [0.8 * sqrt3, 2.4],
        [-0.8 * sqrt3, -2.4],
        [0.8 * sqrt3, -2.4],
        [-2.6 * sqrt3, 1.0],
        [2.6 * sqrt3, 1.0],
        [-2.6 * sqrt3, -1.0],
        [2.6 * sqrt3, -1.0],
        [0.0, -4.0],
    ])


def _search_smallest_R(opt_glob, init, inner_angles, outer_angle_deg, start_R):
    current_R = max(3.9, start_R)
    max_R = 5.5
    step = 0.015
    while current_R <= max_R:
        candidate = opt_glob * (current_R / (current_R + 0.12))
        candidate = _fix_overlaps(candidate, inner_angles, outer_angle_deg, current_R)
        if _packing_valid(candidate, inner_angles, outer_angle_deg, current_R):
            return candidate, current_R, True
        scaled_initial = init * (current_R / 5.5)
        scaled_initial = _fix_overlaps(scaled_initial, inner_angles, outer_angle_deg, current_R)
        if _packing_valid(scaled_initial, inner_angles, outer_angle_deg, current_R):
            return scaled_initial, current_R, True
        current_R += step
    return init, 8.0, False


def _tighten(opt_glob, inner_angles, outer_angle_deg, R):
    tighter_R = R - 0.005
    best_R = R
    best_pos = None
    while tighter_R >= 3.88:
        cand = opt_glob * (tighter_R / (tighter_R + 0.15))
        cand = _fix_overlaps(cand, inner_angles, outer_angle_deg, tighter_R)
        if _packing_valid(cand, inner_angles, outer_angle_deg, tighter_R):
            best_R = tighter_R
            best_pos = cand.copy()
            tighter_R -= 0.005
        else:
            break
    if best_pos is not None:
        return best_pos, best_R, True
    return None, R, False


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    inner_angles = np.zeros(12)
    outer_angle_deg = 0.0
    candidates = [_get_candidate_0(), _get_candidate_1(), _get_candidate_2(), _get_candidate_3(), _get_candidate_4()]
    best_global_positions = candidates[0].copy()
    best_global_side = 8.0
    best_found = False

    for idx, init in enumerate(candidates):
        try:
            result = minimize(
                lambda p: _objective(p, inner_angles),
                init.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 6000, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True},
            )
            opt_local = result.x.reshape(12, 2)
        except Exception:
            opt_local = init.copy()

        opt_glob = opt_local.copy()
        if idx <= 1:
            try:
                bh = basinhopping(
                    lambda p: _objective(p, inner_angles),
                    opt_local.flatten(),
                    niter=40,
                    T=0.13,
                    stepsize=0.28,
                    minimizer_kwargs={'method': 'Nelder-Mead', 'options': {'maxiter': 700, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}},
                    niter_success=12,
                    seed=123 + idx,
                )
                opt_glob = bh.x.reshape(12, 2)
            except Exception:
                opt_glob = opt_local.copy()

        start_R = _min_enclosing_hex_side(opt_glob, inner_angles)
        pos, R, found = _search_smallest_R(opt_glob, init, inner_angles, outer_angle_deg, start_R)
        if found and R < best_global_side:
            best_global_positions = pos
            best_global_side = R
            best_found = True
        if found and R > 3.88:
            tight_pos, tight_R, t_found = _tighten(opt_glob, inner_angles, outer_angle_deg, R)
            if t_found and tight_R < best_global_side:
                best_global_positions = tight_pos
                best_global_side = tight_R
                best_found = True

    if not best_found:
        current_R = 5.0
        scaled = _get_candidate_0() * (5.0 / 5.5)
        scaled = _fix_overlaps(scaled, inner_angles, outer_angle_deg, current_R)
        if _packing_valid(scaled, inner_angles, outer_angle_deg, current_R):
            best_global_positions = scaled
            best_global_side = current_R
        else:
            best_global_positions = _get_candidate_0().copy()
            best_global_side = 8.0

    inner_hex_data = np.zeros((12, 3))
    inner_hex_data[:, :2] = best_global_positions
    inner_hex_data[:, 2] = inner_angles
    outer_hex_data = np.array([0.0, 0.0, outer_angle_deg])
    return inner_hex_data, outer_hex_data, best_global_side