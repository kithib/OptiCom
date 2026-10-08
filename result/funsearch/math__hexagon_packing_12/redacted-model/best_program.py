import numpy as np
from scipy.optimize import minimize, differential_evolution


def _hex_vertices(x, y, angle_deg, side=1.0):
    """Compute vertices of a regular hexagon given center and rotation."""
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.deg2rad(np.arange(0, 360, 60))
    xs = x + side * np.cos(angles)
    ys = y + side * np.sin(angles)
    return np.column_stack((xs, ys))


def _point_in_hex(px, py, hex_center_x, hex_center_y, hex_angle_deg, hex_side):
    """Check if point (px,py) is inside regular hexagon with given center/rotation/side length."""
    dx = px - hex_center_x
    dy = py - hex_center_y
    angle_rad = -np.deg2rad(hex_angle_deg)
    local_x = dx * np.cos(angle_rad) - dy * np.sin(angle_rad)
    local_y = dx * np.sin(angle_rad) + dy * np.cos(angle_rad)
    
    s = hex_side
    sq3 = np.sqrt(3)
    max_xy = s * sq3 / 2
    max_sum = s * sq3
    
    if np.abs(local_x) > max_xy + 1e-8:
        return False
    if np.abs(local_y) > max_xy + 1e-8:
        return False
    if np.abs(sq3 * local_x + local_y) > max_sum + 1e-8:
        return False
    if np.abs(sq3 * local_x - local_y) > max_sum + 1e-8:
        return False
    return True


def _hex_overlap(hex1_data, hex2_data, side=1.0):
    """Check if two unit regular hexagons overlap using SAT (Separating Axis Theorem)."""
    x1, y1, a1 = hex1_data
    x2, y2, a2 = hex2_data
    verts1 = _hex_vertices(x1, y1, a1, side)
    verts2 = _hex_vertices(x2, y2, a2, side)
    
    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-12:
                normal /= norm
            axes.append(normal)
        return axes
    
    axes = get_axes(verts1) + get_axes(verts2)
    
    for axis in axes:
        proj1 = np.dot(verts1, axis)
        proj2 = np.dot(verts2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        if max1 < min2 - 1e-8 or max2 < min1 - 1e-8:
            return False
    return True


def _hex_distance(hex1_data, hex2_data, side=1.0):
    """Compute distance between centers of two hexagons for penalty calculations."""
    x1, y1, _ = hex1_data
    x2, y2, _ = hex2_data
    return np.hypot(x1 - x2, y1 - y2)


def _compute_min_outer_side(inner_hex_data):
    """Compute minimal outer hexagon side length that contains all inner hexagons."""
    n = len(inner_hex_data)
    sq3 = np.sqrt(3)
    min_side = 0.0
    for i in range(n):
        xi, yi, ai = inner_hex_data[i]
        verts = _hex_vertices(xi, yi, ai, 1.0)
        for v in verts:
            local_x = v[0]
            local_y = v[1]
            candidate = max(
                np.abs(local_x) * 2 / sq3,
                np.abs(local_y) * 2 / sq3,
                np.abs(sq3 * local_x + local_y) / sq3,
                np.abs(sq3 * local_x - local_y) / sq3,
            )
            if candidate > min_side:
                min_side = candidate
    return min_side


def _packing_feasible_and_cost(params, n=12):
    """
    Evaluate packing: params = [x0,y0,angle0,x1,y1,angle1,...,outer_side]
    Returns (is_feasible, cost, total_violation)
    """
    inner_params = params[:3 * n].reshape(n, 3)
    outer_side = params[-1]
    max_violation = 0.0
    overlap_penalty = 0.0
    
    # Check containment
    for i in range(n):
        xi, yi, ai = inner_params[i]
        verts = _hex_vertices(xi, yi, ai, 1.0)
        for v in verts:
            if not _point_in_hex(v[0], v[1], 0.0, 0.0, 0.0, outer_side):
                dx = v[0]
                dy = v[1]
                sq3 = np.sqrt(3)
                violations = [
                    np.abs(dx) - outer_side * sq3 / 2,
                    np.abs(dy) - outer_side * sq3 / 2,
                    np.abs(sq3 * dx + dy) - outer_side * sq3,
                    np.abs(sq3 * dx - dy) - outer_side * sq3,
                ]
                v_max = max(violations)
                if v_max > max_violation:
                    max_violation = v_max
    
    # Check non-overlap
    for i in range(n):
        for j in range(i + 1, n):
            if _hex_overlap(inner_params[i], inner_params[j]):
                overlap_penalty += 1.0
                dist = _hex_distance(inner_params[i], inner_params[j])
                overlap_penalty += max(0.0, 2.0 - dist)
    
    is_feasible = (max_violation <= 1e-6) and (overlap_penalty == 0.0)
    cost = outer_side
    total_violation = max_violation + overlap_penalty
    return is_feasible, cost, total_violation


def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    n = 12
    sq3 = np.sqrt(3)
    
    # Initial arrangement: dense hexagonal grid with strategic rotations (dense asymmetric layout)
    initial_positions = np.array([
        [0.0, 0.0],
        [sq3, 0.0],
        [-sq3, 0.0],
        [sq3 / 2, 1.5],
        [-sq3 / 2, 1.5],
        [sq3 / 2, -1.5],
        [-sq3 / 2, -1.5],
        [2 * sq3, 0.0],
        [-2 * sq3, 0.0],
        [sq3, 3.0],
        [-sq3, 3.0],
        [0.0, -3.0],
    ])
    
    # Mixed rotations to fill gaps
    rotations = np.array([0.0, 0.0, 0.0, 30.0, 30.0, 30.0, 30.0, 0.0, 0.0, 30.0, 30.0, 30.0])
    
    # Build initial inner hex data
    inner_hex_data = np.zeros((n, 3))
    inner_hex_data[:, :2] = initial_positions
    inner_hex_data[:, 2] = rotations
    
    # Compute initial outer side estimate
    initial_outer = _compute_min_outer_side(inner_hex_data) + 0.1
    initial_params = np.concatenate([inner_hex_data.flatten(), [initial_outer]])
    
    def objective(params):
        feasible, cost, violation = _packing_feasible_and_cost(params, n)
        penalty = 1000.0 * max(violation, 0.0)
        return cost + penalty
    
    try:
        # Stage 1: Fast global search with differential evolution (tight bounds for speed)
        de_bounds = [(-4.5, 4.5), (-4.5, 4.5), (0, 60)] * n + [(3.0, 6.5)]
        de_result = differential_evolution(
            objective,
            de_bounds,
            maxiter=25,
            popsize=12,
            tol=1e-6,
            mutation=(0.5, 1.0),
            recombination=0.7,
            seed=42,
            polish=True,
            init='sobol',
            disp=False,
            workers=1,
        )
        best_params = de_result.x
        
        # Stage 2: Strict local refinement with L-BFGS-B
        lbfgs_bounds = [(-10, 10), (-10, 10), (0, 60)] * n + [(3.0, 10.0)]
        lbfgs_result = minimize(
            objective,
            best_params,
            method="L-BFGS-B",
            bounds=lbfgs_bounds,
            options={"maxiter": 1200, "ftol": 1e-12, "gtol": 1e-10, "disp": False},
        )
        optimized_params = lbfgs_result.x
        inner_optimized = optimized_params[:3 * n].reshape(n, 3)
        
        # Multi-pass overlap correction with safety margin
        for _ in range(3):
            feasible, _, _ = _packing_feasible_and_cost(optimized_params, n)
            if feasible:
                break
            for i in range(n):
                for j in range(i + 1, n):
                    if _hex_overlap(inner_optimized[i], inner_optimized[j]):
                        dx = inner_optimized[j, 0] - inner_optimized[i, 0]
                        dy = inner_optimized[j, 1] - inner_optimized[i, 1]
                        dist = np.hypot(dx, dy) + 1e-12
                        shift = (2.0 - dist) / 2.0 + 0.05
                        inner_optimized[i, 0] -= dx / dist * shift
                        inner_optimized[i, 1] -= dy / dist * shift
                        inner_optimized[j, 0] += dx / dist * shift
                        inner_optimized[j, 1] += dy / dist * shift
            optimized_params = np.concatenate([inner_optimized.flatten(), [optimized_params[-1]]])
        
        # Minimal outer side calculation
        final_outer = _compute_min_outer_side(inner_optimized)
        final_outer *= 1.0005
        
        # Final feasibility check
        test_params = np.concatenate([inner_optimized.flatten(), [final_outer]])
        feasible, _, _ = _packing_feasible_and_cost(test_params, n)
        if not feasible:
            final_outer *= 1.01
        
        # Direct pairwise overlap validation
        valid = True
        for i in range(n):
            for j in range(i + 1, n):
                if _hex_overlap(inner_optimized[i], inner_optimized[j]):
                    valid = False
                    break
            if not valid:
                break
        
        # Containment verification
        if valid:
            for i in range(n):
                xi, yi, ai = inner_optimized[i]
                verts = _hex_vertices(xi, yi, ai, 1.0)
                for v in verts:
                    if not _point_in_hex(v[0], v[1], 0.0, 0.0, 0.0, final_outer):
                        valid = False
                        break
                if not valid:
                    break
        
        if valid:
            inner_hex_data = inner_optimized
            outer_hex_side_length = final_outer
        else:
            inner_hex_data = np.zeros((n, 3))
            inner_hex_data[:, :2] = initial_positions
            inner_hex_data[:, 2] = rotations
            outer_hex_side_length = _compute_min_outer_side(inner_hex_data) * 1.01
    except Exception:
        inner_hex_data = np.zeros((n, 3))
        inner_hex_data[:, :2] = initial_positions
        inner_hex_data[:, 2] = rotations
        outer_hex_side_length = _compute_min_outer_side(inner_hex_data) * 1.01
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    return inner_hex_data, outer_hex_data, outer_hex_side_length