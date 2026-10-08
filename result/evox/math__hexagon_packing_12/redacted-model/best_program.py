import numpy as np


def _hex_corners(cx, cy, angle_deg, side=1.0):
    angle = np.deg2rad(angle_deg)
    corners = []
    for k in range(6):
        theta = angle + k * np.pi / 3.0
        x = cx + side * np.cos(theta)
        y = cy + side * np.sin(theta)
        corners.append((x, y))
    return corners


def _sat_poly_overlap(poly_a, poly_b, eps=1e-6):
    polys = [poly_a, poly_b]
    for poly in polys:
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            nx = -(y2 - y1)
            ny = (x2 - x1)
            norm = np.hypot(nx, ny)
            if norm < 1e-12:
                continue
            nxn, nyn = nx / norm, ny / norm
            min_a = min(x * nxn + y * nyn for x, y in poly_a)
            max_a = max(x * nxn + y * nyn for x, y in poly_a)
            min_b = min(x * nxn + y * nyn for x, y in poly_b)
            max_b = max(x * nxn + y * nyn for x, y in poly_b)
            if max_a + eps < min_b or max_b + eps < min_a:
                return False
    return True


def _point_in_hex(point, outer_corners):
    px, py = point
    for i in range(6):
        x1, y1 = outer_corners[i]
        x2, y2 = outer_corners[(i + 1) % 6]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-6:
            return False
    return True


def _min_circumscribed_side(inner_hex_data, outer_hex_data, inner_side):
    outer_angle = np.deg2rad(outer_hex_data[2])
    corners = []
    n = inner_hex_data.shape[0]
    for idx in range(n):
        cx, cy = inner_hex_data[idx, 0], inner_hex_data[idx, 1]
        theta = inner_hex_data[idx, 2]
        for corner in _hex_corners(cx, cy, theta, inner_side):
            corners.append(corner)
    lo, hi = 1.0, 100.0
    for _ in range(300):
        mid = 0.5 * (lo + hi)
        oc_x, oc_y = outer_hex_data[0], outer_hex_data[1]
        outer_corners = []
        for k in range(6):
            theta = outer_angle + k * np.pi / 3.0
            outer_corners.append((oc_x + mid * np.cos(theta), oc_y + mid * np.sin(theta)))
        all_in = True
        for c in corners:
            if not _point_in_hex(c, outer_corners):
                all_in = False
                break
        if all_in:
            hi = mid
        else:
            lo = mid
    return hi


def _all_pairs_nonoverlap(inner_hex_data, inner_side=1.0):
    n = inner_hex_data.shape[0]
    corners_list = []
    for i in range(n):
        corners_list.append(_hex_corners(inner_hex_data[i, 0], inner_hex_data[i, 1], inner_hex_data[i, 2], inner_side))
    for i in range(n):
        for j in range(i + 1, n):
            if _sat_poly_overlap(corners_list[i], corners_list[j]):
                return False
    return True


def _compute_forces_and_shrink(current, n, max_iter, seed_base, shrink_start=1.0):
    rate = 0.17
    shrink = shrink_start
    shrink_step = 0.9984
    for it in range(max_iter):
        force = np.zeros((n, 2))
        corners_list = []
        eff_side = 1.0 * shrink
        for i in range(n):
            corners_list.append(_hex_corners(current[i, 0], current[i, 1], current[i, 2], eff_side))
        colliding = False
        for i in range(n):
            for j in range(i + 1, n):
                if _sat_poly_overlap(corners_list[i], corners_list[j]):
                    diff = current[j, :2] - current[i, :2]
                    dist = np.hypot(diff[0], diff[1])
                    if dist < 1e-6:
                        diff = np.random.default_rng(i * 113 + j * 17 + seed_base + it * 3).standard_normal(2)
                        dist = np.hypot(diff[0], diff[1])
                    unit = diff / dist
                    force[i] -= 3.5 * unit
                    force[j] += 3.5 * unit
                    colliding = True
        current[:, :2] += rate * force
        rate *= 0.9935
        rate = max(rate, 4.2e-3)
        shrink *= shrink_step
        shrink = max(shrink, 0.77)
        if not colliding and it > 50:
            break
    return current


def _enforce_no_overlap(current, n, safety_margin=0.01, max_iter=450):
    corners_list = []
    for i in range(n):
        corners_list.append(_hex_corners(current[i, 0], current[i, 1], current[i, 2], 1.0))
    for _ in range(max_iter):
        moved = False
        for i in range(n):
            for j in range(i + 1, n):
                if _sat_poly_overlap(corners_list[i], corners_list[j]):
                    dx_vec = current[j, :2] - current[i, :2]
                    dist = np.hypot(dx_vec[0], dx_vec[1])
                    if dist < 1e-6:
                        rng = np.random.default_rng(i * 41 + j * 29)
                        dx_vec = rng.standard_normal(2)
                        dist = np.hypot(dx_vec[0], dx_vec[1])
                    unit = dx_vec / dist
                    current[i, :2] -= safety_margin * unit
                    current[j, :2] += safety_margin * unit
                    corners_list[i] = _hex_corners(current[i, 0], current[i, 1], current[i, 2], 1.0)
                    corners_list[j] = _hex_corners(current[j, 0], current[j, 1], current[j, 2], 1.0)
                    moved = True
        if not moved:
            break
    return current


def hexagon_packing_12():
    n = 12
    sqrt3 = np.sqrt(3.0)
    dx_step = 1.5
    dy_step = sqrt3 / 2.0

    base = np.array([
        [0.0, 0.0],
        [-dx_step, dy_step],
        [dx_step, dy_step],
        [-dx_step, -dy_step],
        [dx_step, -dy_step],
        [-2.0 * dx_step, 0.0],
        [2.0 * dx_step, 0.0],
        [-2.0 * dx_step, 2.0 * dy_step],
        [2.0 * dx_step, -2.0 * dy_step],
        [-2.0 * dx_step, -2.0 * dy_step],
        [2.0 * dx_step, 2.0 * dy_step],
        [0.0, -3.0 * dy_step],
    ], dtype=np.float64)

    best_candidates = None
    best_side = np.inf
    outer_hex_template = np.array([0.0, 0.0, 0.0])

    for outer in range(26):
        rng = np.random.default_rng(outer * 137 + 17)
        current = np.zeros((n, 3), dtype=np.float64)
        jitter = rng.standard_normal((n, 2)) * 0.065
        scale_x = 1.0 - 0.0105 * outer
        scale_y = 1.0 - 0.0125 * outer
        current[:, 0] = base[:, 0] * scale_x + jitter[:, 0]
        current[:, 1] = base[:, 1] * scale_y + jitter[:, 1]
        current[:, 2] = 0.0

        current = _compute_forces_and_shrink(current, n, 330, outer * 137 + 17)
        current = _enforce_no_overlap(current, n)

        center = np.mean(current[:, :2], axis=0)
        current[:, :2] -= center
        current = np.ascontiguousarray(current)

        if _all_pairs_nonoverlap(current, 1.0):
            side = _min_circumscribed_side(current, outer_hex_template, 1.0)
            if side < best_side:
                best_side = side
                best_candidates = current.copy()

    if best_candidates is not None:
        for sub in range(24):
            rng = np.random.default_rng(9991 + sub)
            cand = best_candidates.copy()
            cand[:, :2] *= (1.0 - 0.0032 * sub)
            cand[:, :2] += rng.standard_normal((n, 2)) * 0.016
            cand[:, 2] = rng.standard_normal(n) * 0.32
            cand = _compute_forces_and_shrink(cand, n, 270, 9991 + sub, shrink_start=0.95)
            cand[:, 2] = 0.0
            cand = _enforce_no_overlap(cand, n)
            center = np.mean(cand[:, :2], axis=0)
            cand[:, :2] -= center
            cand = np.ascontiguousarray(cand)
            if _all_pairs_nonoverlap(cand, 1.0):
                side = _min_circumscribed_side(cand, outer_hex_template, 1.0)
                if side < best_side:
                    best_side = side
                    best_candidates = cand.copy()

    if best_candidates is not None:
        for sub in range(16):
            rng = np.random.default_rng(7777 + sub)
            cand = best_candidates.copy()
            shrink_factor = 1.0 - 0.0027 * sub
            cand[:, :2] *= shrink_factor
            cand[:, :2] += rng.standard_normal((n, 2)) * 0.007
            cand = _compute_forces_and_shrink(cand, n, 250, 7777 + sub, shrink_start=0.92)
            cand = _enforce_no_overlap(cand, n, safety_margin=0.0075)
            center = np.mean(cand[:, :2], axis=0)
            cand[:, :2] -= center
            cand = np.ascontiguousarray(cand)
            if _all_pairs_nonoverlap(cand, 1.0):
                side = _min_circumscribed_side(cand, outer_hex_template, 1.0)
                if side < best_side:
                    best_side = side
                    best_candidates = cand.copy()

    if best_candidates is not None:
        for sub in range(14):
            rng = np.random.default_rng(5555 + sub)
            cand = best_candidates.copy()
            cand[:, 0] *= (1.0 - 0.0019 * sub)
            cand[:, 1] *= (1.0 - 0.0023 * sub)
            cand[:, :2] += rng.standard_normal((n, 2)) * 0.0045
            cand = _compute_forces_and_shrink(cand, n, 230, 5555 + sub, shrink_start=0.89)
            cand = _enforce_no_overlap(cand, n, safety_margin=0.0065)
            center = np.mean(cand[:, :2], axis=0)
            cand[:, :2] -= center
            cand = np.ascontiguousarray(cand)
            if _all_pairs_nonoverlap(cand, 1.0):
                side = _min_circumscribed_side(cand, outer_hex_template, 1.0)
                if side < best_side:
                    best_side = side
                    best_candidates = cand.copy()

    if best_candidates is None:
        best_candidates = np.zeros((n, 3), dtype=np.float64)
        best_candidates[:, :2] = base * 2.3
        center = np.mean(best_candidates[:, :2], axis=0)
        best_candidates[:, :2] -= center

    inner_hex_data = np.ascontiguousarray(best_candidates)
    outer_hex_data = np.array([0.0, 0.0, 0.0])

    if not _all_pairs_nonoverlap(inner_hex_data, 1.0):
        fallback = np.zeros((n, 3), dtype=np.float64)
        fallback[:, :2] = base * 2.4
        center = np.mean(fallback[:, :2], axis=0)
        fallback[:, :2] -= center
        inner_hex_data = np.ascontiguousarray(fallback)

    side = _min_circumscribed_side(inner_hex_data, outer_hex_data, 1.0)
    outer_hex_side_length = float(side * 1.0022)
    return inner_hex_data, outer_hex_data, outer_hex_side_length