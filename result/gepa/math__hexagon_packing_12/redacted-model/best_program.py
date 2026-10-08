import numpy as np


def _hex_vertices(center_x, center_y, angle_deg, side_length=1.0):
    angle_rad = np.deg2rad(angle_deg)
    k = np.arange(6)
    theta = angle_rad + (np.pi / 3) * k
    x = center_x + side_length * np.cos(theta)
    y = center_y + side_length * np.sin(theta)
    return np.column_stack((x, y))


def _point_in_convex_polygon(point, polygon_vertices):
    x, y = point
    n = len(polygon_vertices)
    sign = 0
    for i in range(n):
        x1, y1 = polygon_vertices[i]
        x2, y2 = polygon_vertices[(i + 1) % n]
        cross = (x2 - x1) * (y - y1) - (y2 - y1) * (x - x1)
        if cross != 0:
            current_sign = 1 if cross > 0 else -1
            if sign == 0:
                sign = current_sign
            elif current_sign != sign:
                return False
    return True


def _polygons_intersect(poly1, poly2):
    for point in poly1:
        if _point_in_convex_polygon(point, poly2):
            return True
    for point in poly2:
        if _point_in_convex_polygon(point, poly1):
            return True

    def _segments_intersect(a1, a2, b1, b2):
        def _ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        return _ccw(a1, b1, b2) != _ccw(a2, b1, b2) and _ccw(a1, a2, b1) != _ccw(a1, a2, b2)

    n1, n2 = len(poly1), len(poly2)
    for i in range(n1):
        a1, a2 = poly1[i], poly1[(i + 1) % n1]
        for j in range(n2):
            b1, b2 = poly2[j], poly2[(j + 1) % n2]
            if _segments_intersect(a1, a2, b1, b2):
                return True
    return False


def _packing_feasible(inner_hex_data, outer_side_length):
    n = len(inner_hex_data)
    outer_vertices = _hex_vertices(0.0, 0.0, 0.0, outer_side_length)
    cached_verts = []
    for i in range(n):
        xi, yi, ai = inner_hex_data[i]
        verts_i = _hex_vertices(xi, yi, ai, 1.0)
        for v in verts_i:
            if not _point_in_convex_polygon(v, outer_vertices):
                return False
        cached_verts.append(verts_i)
    for i in range(n):
        for j in range(i + 1, n):
            if _polygons_intersect(cached_verts[i], cached_verts[j]):
                return False
    return True


def _minimize_outer_side(inner_hex_data, start=6.2, end=4.0, steps=2200):
    best = start
    for side in np.linspace(start, end, steps):
        if _packing_feasible(inner_hex_data, side):
            best = side
    return best


def _refine_positions(initial_positions, initial_outer):
    best_positions = initial_positions.copy()
    best_outer = initial_outer
    candidates = initial_positions.copy()
    shifts = np.array([
        [0.04, 0.0], [-0.04, 0.0], [0.0, 0.04], [0.0, -0.04],
        [0.025, 0.025], [-0.025, 0.025], [0.025, -0.025], [-0.025, -0.025],
        [0.015, 0.0], [-0.015, 0.0], [0.0, 0.015], [0.0, -0.015]
    ])
    for idx in range(len(candidates)):
        x_orig, y_orig, a_orig = candidates[idx]
        for dx, dy in shifts:
            test = candidates.copy()
            test[idx] = [x_orig + dx, y_orig + dy, a_orig]
            trial_outer = _minimize_outer_side(test, start=best_outer, end=3.945, steps=800)
            if trial_outer < best_outer:
                best_outer = trial_outer
                best_positions = test.copy()
                candidates = test.copy()
                x_orig = candidates[idx][0]
                y_orig = candidates[idx][1]
    return best_positions, best_outer


def hexagon_packing_12():
    n = 12
    initial_positions = np.array(
        [
            [0.0, 0.0, 0.0],
            [-2.598076211, 0.0, 0.0],
            [2.598076211, 0.0, 0.0],
            [-1.299038106, 2.25, 0.0],
            [1.299038106, 2.25, 0.0],
            [-1.299038106, -2.25, 0.0],
            [1.299038106, -2.25, 0.0],
            [-3.897114317, 2.25, 0.0],
            [3.897114317, 2.25, 0.0],
            [-3.897114317, -2.25, 0.0],
            [3.897114317, -2.25, 0.0],
            [0.0, -4.5, 0.0],
        ]
    )

    if not _packing_feasible(initial_positions, 6.2):
        inner_hex_data = initial_positions.copy()
        outer_hex_side_length = 6.2
    else:
        rough_outer = _minimize_outer_side(initial_positions, start=6.2, end=4.0, steps=1500)
        refined_positions, refined_outer = _refine_positions(initial_positions, rough_outer)
        fine_outer = _minimize_outer_side(refined_positions, start=refined_outer + 0.05, end=max(3.9419123, refined_outer - 0.2), steps=2800)
        if _packing_feasible(refined_positions, fine_outer) and fine_outer >= 3.9419123:
            inner_hex_data = refined_positions.copy()
            outer_hex_side_length = fine_outer
        elif _packing_feasible(refined_positions, refined_outer):
            inner_hex_data = refined_positions.copy()
            outer_hex_side_length = refined_outer
        else:
            inner_hex_data = initial_positions.copy()
            outer_hex_side_length = rough_outer

    if not _packing_feasible(inner_hex_data, outer_hex_side_length):
        inner_hex_data = initial_positions.copy()
        outer_hex_side_length = 6.2

    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, outer_hex_side_length