import numpy as np


def _get_hex_vertices(x, y, angle_deg, side=1.0):
    angle_rad = np.deg2rad(angle_deg)
    base = np.array(
        [
            (np.cos(np.pi / 3 * k), np.sin(np.pi / 3 * k))
            for k in range(6)
        ]
    )
    rot = np.array(
        [
            [np.cos(angle_rad), -np.sin(angle_rad)],
            [np.sin(angle_rad), np.cos(angle_rad)],
        ]
    )
    return (base @ rot.T) * side + np.array([x, y])


def _point_in_hex(px, py, outer_xy, outer_angle_deg, outer_side):
    verts = _get_hex_vertices(outer_xy[0], outer_xy[1], outer_angle_deg, outer_side)
    n = len(verts)
    sign = 0
    eps = 1e-9
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if abs(cross) < eps:
            continue
        if sign == 0:
            sign = np.sign(cross)
        elif np.sign(cross) != sign:
            return False
    return True


def _hexagons_overlap(inner1, inner2):
    x1, y1, a1 = inner1
    x2, y2, a2 = inner2
    v1 = _get_hex_vertices(x1, y1, a1, 1.0)
    v2 = _get_hex_vertices(x2, y2, a2, 1.0)
    axes = []
    for verts in (v1, v2):
        for i in range(6):
            x1a, y1a = verts[i]
            x2a, y2a = verts[(i + 1) % 6]
            edge = np.array([x2a - x1a, y2a - y1a])
            axis = np.array([-edge[1], edge[0]])
            nrm = np.linalg.norm(axis)
            if nrm < 1e-12:
                continue
            axis = axis / nrm
            axes.append(axis)
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        if max(proj1) < min(proj2) - 1e-9 or max(proj2) < min(proj1) - 1e-9:
            return False
    return True


def _resolve_all_overlaps(inner_data):
    sqrt3 = np.sqrt(3)
    needed_base = sqrt3 * 1.10 + 1e-3
    for _ in range(500):
        moved = False
        for i in range(len(inner_data)):
            for j in range(i + 1, len(inner_data)):
                if _hexagons_overlap(inner_data[i], inner_data[j]):
                    dx = inner_data[j][0] - inner_data[i][0]
                    dy = inner_data[j][1] - inner_data[i][1]
                    dist = np.hypot(dx, dy)
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    move = (needed_base - dist) * 1.30
                    if move > 0:
                        ux = dx / dist
                        uy = dy / dist
                        half_move = move / 2
                        inner_data[j][0] += ux * half_move
                        inner_data[j][1] += uy * half_move
                        inner_data[i][0] -= ux * half_move
                        inner_data[i][1] -= uy * half_move
                        moved = True
        if not moved:
            break
    return inner_data


def _compute_min_outer_side(inner_data):
    max_val = 0.0
    sqrt3 = np.sqrt(3)
    inv_half_sqrt3 = 2.0 / sqrt3
    for k in range(len(inner_data)):
        x, y = inner_data[k][0], inner_data[k][1]
        verts = _get_hex_vertices(x, y, 0.0, 1.0)
        for (vx, vy) in verts:
            t1 = abs(vx)
            t2 = abs(vy)
            t3 = abs(sqrt3 * vx + vy) * 0.5
            t4 = abs(sqrt3 * vx - vy) * 0.5
            t5 = abs(-sqrt3 * vx + vy) * 0.5
            t6 = abs(-sqrt3 * vx - vy) * 0.5
            R = max(t1, t2, t3, t4, t5, t6) * inv_half_sqrt3
            if R > max_val:
                max_val = R
    return max_val


def _validate_all(inner_data, outer_side):
    n = len(inner_data)
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(inner_data[i], inner_data[j]):
                return False
    for k in range(n):
        x, y, _ = inner_data[k]
        verts = _get_hex_vertices(x, y, 0.0, 1.0)
        for (vx, vy) in verts:
            if not _point_in_hex(vx, vy, (0.0, 0.0), 0.0, outer_side):
                return False
    return True


def hexagon_packing_12():
    sqrt3 = np.sqrt(3)
    base_step_x = 1.5 * sqrt3
    base_step_y = 2.25
    base_positions = [
        (0, 0),
        (-base_step_x, 0),
        (base_step_x, 0),
        (-base_step_x / 2, base_step_y),
        (base_step_x / 2, base_step_y),
        (-base_step_x / 2, -base_step_y),
        (base_step_x / 2, -base_step_y),
        (-base_step_x * 1.5, base_step_y),
        (base_step_x * 1.5, base_step_y),
        (-base_step_x * 1.5, -base_step_y),
        (base_step_x * 1.5, -base_step_y),
        (0, -2 * base_step_y),
    ]

    best_positions = None
    best_side = float("inf")

    scale_range = [
        1.000, 0.994, 1.006, 0.988, 1.012, 0.982, 1.018, 0.976, 1.024,
        0.970, 1.030, 0.964, 1.036, 0.959, 1.041, 0.954, 1.046, 0.949,
        1.051, 0.944, 1.056, 0.940, 1.060, 0.936, 1.064
    ]
    offset_range = [
        -0.0375, -0.028125, -0.01875, -0.009375, 0.0,
        0.009375, 0.01875, 0.028125, 0.0375
    ]

    for scale in scale_range:
        for dx in offset_range:
            for dy in offset_range:
                test = np.array(
                    [
                        [x * scale + dx, y * scale + dy, 0.0]
                        for (x, y) in base_positions
                    ],
                    dtype=float,
                )
                test = _resolve_all_overlaps(test)
                current_side = _compute_min_outer_side(test)
                test_side = float(np.ceil(current_side * 10000) / 10000) + 1.1e-2
                if _validate_all(test, test_side):
                    if current_side < best_side:
                        best_positions = test.copy()
                        best_side = current_side

    if best_positions is None:
        best_positions = np.array(
            [[x, y, 0.0] for (x, y) in base_positions], dtype=float
        )
        best_positions = _resolve_all_overlaps(best_positions)
        best_side = _compute_min_outer_side(best_positions)

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = float(np.ceil(best_side * 10000) / 10000) + 1.5e-2

    for _ in range(500):
        if _validate_all(best_positions, outer_hex_side_length):
            break
        outer_hex_side_length += 0.003
        if outer_hex_side_length > 20:
            outer_hex_side_length = 20
            break

    return best_positions, outer_hex_data, outer_hex_side_length