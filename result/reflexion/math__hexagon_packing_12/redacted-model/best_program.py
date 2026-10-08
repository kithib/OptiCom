import numpy as np


def get_hex_vertices(center_x, center_y, angle_deg, side_length):
    angle_rad = np.deg2rad(angle_deg)
    vertices = []
    for k in range(6):
        theta = angle_rad + k * np.pi / 3
        vx = center_x + side_length * np.cos(theta)
        vy = center_y + side_length * np.sin(theta)
        vertices.append((vx, vy))
    return vertices


def point_in_regular_hex(px, py, outer_cx, outer_cy, outer_angle_deg, outer_side):
    vertices = get_hex_vertices(outer_cx, outer_cy, outer_angle_deg, outer_side)
    n = len(vertices)
    inside = True
    eps = -1e-8
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < eps:
            inside = False
            break
    return inside


def all_hexes_contained(inner_hex_data, outer_hex_data, outer_side):
    outer_cx, outer_cy, outer_angle = outer_hex_data
    for row in inner_hex_data:
        cx, cy, angle_deg = row
        vertices = get_hex_vertices(cx, cy, angle_deg, 1.0)
        for (vx, vy) in vertices:
            if not point_in_regular_hex(vx, vy, outer_cx, outer_cy, outer_angle, outer_side):
                return False
    return True


def hexes_overlap(hex_a, hex_b):
    cx_a, cy_a, angle_a = hex_a
    cx_b, cy_b, angle_b = hex_b
    verts_a = get_hex_vertices(cx_a, cy_a, angle_a, 1.0)
    verts_b = get_hex_vertices(cx_b, cy_b, angle_b, 1.0)
    for (v1, v2) in [(verts_a, verts_b), (verts_b, verts_a)]:
        n = len(v1)
        for i in range(n):
            x1, y1 = v1[i]
            x2, y2 = v1[(i + 1) % n]
            projections_a = []
            projections_b = []
            nx = -(y2 - y1)
            ny = (x2 - x1)
            norm = np.hypot(nx, ny)
            if norm < 1e-12:
                continue
            nx /= norm
            ny /= norm
            for (px, py) in v1:
                projections_a.append(px * nx + py * ny)
            for (px, py) in v2:
                projections_b.append(px * nx + py * ny)
            min_a = min(projections_a)
            max_a = max(projections_a)
            min_b = min(projections_b)
            max_b = max(projections_b)
            if max_a < min_b - 1e-8 or max_b < min_a - 1e-8:
                return False
    return True


def any_overlap(inner_hex_data):
    n = inner_hex_data.shape[0]
    for i in range(n):
        for j in range(i + 1, n):
            if hexes_overlap(inner_hex_data[i], inner_hex_data[j]):
                return True
    return False


def hexagon_packing_12():
    sqrt3 = np.sqrt(3)
    step_x = 1.5 * sqrt3
    step_y = 3.0

    inner_hex_data = np.array(
        [
            [-step_x * 1.5, -step_y * 0.5, 0.0],
            [-step_x * 0.5, -step_y * 0.5, 0.0],
            [step_x * 0.5, -step_y * 0.5, 0.0],
            [step_x * 1.5, -step_y * 0.5, 0.0],
            [-step_x, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [step_x, 0.0, 0.0],
            [-step_x * 0.5, step_y * 0.5, 0.0],
            [step_x * 0.5, step_y * 0.5, 0.0],
            [-step_x, step_y, 0.0],
            [0.0, step_y, 0.0],
            [step_x, step_y, 0.0],
        ],
        dtype=float,
    )

    outer_cx = 0.0
    outer_cy = 0.0
    outer_angle = 0.0
    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle])

    max_center_x = np.max(np.abs(inner_hex_data[:, 0]))
    max_center_y = np.max(np.abs(inner_hex_data[:, 1]))
    half_width_unit_hex = 1.5
    half_height_unit_hex = sqrt3 / 2

    R_from_x = (2.0 / sqrt3) * (max_center_x + half_width_unit_hex)
    R_from_y = (2.0 / sqrt3) * (max_center_y + half_height_unit_hex)
    outer_side = max(R_from_x, R_from_y)

    outer_side *= 1.01
    attempts = 0
    shrink_factor = 0.999
    while attempts < 500 and all_hexes_contained(inner_hex_data, outer_hex_data, outer_side):
        candidate = outer_side * shrink_factor
        if all_hexes_contained(inner_hex_data, outer_hex_data, candidate):
            outer_side = candidate
        else:
            break
        attempts += 1

    outer_side *= 1.0001
    while not all_hexes_contained(inner_hex_data, outer_hex_data, outer_side):
        outer_side *= 1.001

    return inner_hex_data, outer_hex_data, outer_side