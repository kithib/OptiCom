import numpy as np
from scipy.optimize import minimize, basinhopping, dual_annealing
from scipy.special import logsumexp
import time

np.random.seed(42)

def hex_vertices(cx, cy, angle_deg, radius):
    angle_rad = np.deg2rad(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + radius * np.cos(angles), cy + radius * np.sin(angles)])

def outer_hex_lines(cx, cy, angle_deg, side):
    radius = side
    verts = hex_vertices(cx, cy, angle_deg, radius)
    lines = []
    for i in range(6):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % 6]
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        norm = np.hypot(a, b)
        if norm < 1e-12:
            continue
        a, b, c = a / norm, b / norm, c / norm
        px, py = cx, cy
        if a * px + b * py + c > 0:
            a, b, c = -a, -b, -c
        lines.append((a, b, c))
    return lines

def project_feasible(x, n):
    x = x.copy()
    for i in range(n):
        idx = i * 3
        x[idx + 2] = x[idx + 2] % 60.0
        if x[idx + 2] < 0:
            x[idx + 2] += 60.0
    return x

def smooth_objective(x, n, outer_cx, outer_cy, outer_angle_deg, inner_radius, alpha):
    outer_lines = outer_hex_lines(outer_cx, outer_cy, outer_angle_deg, 1.0)
    all_d = []
    for i in range(n):
        idx = i * 3
        cx, cy, ang = x[idx], x[idx + 1], x[idx + 2]
        verts = hex_vertices(cx, cy, ang, inner_radius)
        for v in verts:
            for a, b, c in outer_lines:
                d = -(a * v[0] + b * v[1] + c)
                all_d.append(d)
    all_d = np.array(all_d)
    if len(all_d) == 0:
        return 0.0
    apothem_smooth = logsumexp(alpha * all_d) / alpha
    if apothem_smooth <= 1e-12:
        return 0.0
    outer_side_smooth = (2.0 / np.sqrt(3)) * apothem_smooth
    return 1.0 / outer_side_smooth

def loss_func(x, n, outer_cx, outer_cy, outer_angle_deg, inner_radius, alpha, penalty):
    inv_side_smooth = smooth_objective(x, n, outer_cx, outer_cy, outer_angle_deg, inner_radius, alpha)
    if inv_side_smooth <= 1e-12:
        return 1e6
    obj = -inv_side_smooth
    min_dist_sq_inner = 4.0 * inner_radius * inner_radius * 0.9999
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            idx_i, idx_j = i * 3, j * 3
            dx = x[idx_i] - x[idx_j]
            dy = x[idx_i + 1] - x[idx_j + 1]
            dist_sq = dx * dx + dy * dy
            pairs.append(min_dist_sq_inner - dist_sq)
    pairs = np.array(pairs)
    overlap_pen = 0.0
    if len(pairs) > 0:
        safe_pairs = np.clip(pairs, -500, None)
        overlap_pen = (logsumexp(alpha * safe_pairs) / alpha) ** 2
    outer_lines = outer_hex_lines(outer_cx, outer_cy, outer_angle_deg, 1.0)
    contain_viol = []
    for i in range(n):
        idx = i * 3
        cx, cy, ang = x[idx], x[idx + 1], x[idx + 2]
        verts = hex_vertices(cx, cy, ang, inner_radius)
        for v in verts:
            for a, b, c in outer_lines:
                d = -(a * v[0] + b * v[1] + c)
                if d > 1.0:
                    contain_viol.append(d - 1.0)
    contain_viol = np.array(contain_viol)
    contain_pen = 0.0
    if len(contain_viol) > 0:
        safe_contain = np.clip(contain_viol, -500, None)
        contain_pen = (logsumexp(alpha * safe_contain) / alpha) ** 2
    return obj + penalty * (overlap_pen + contain_pen)

def get_collision_pairs(x, n, inner_radius, tol=1e-3):
    pairs = []
    min_dist_sq = (2.0 * inner_radius * (1.0 - tol)) ** 2
    for i in range(n):
        for j in range(i + 1, n):
            idx_i, idx_j = i * 3, j * 3
            dx = x[idx_i] - x[idx_j]
            dy = x[idx_i + 1] - x[idx_j + 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq < min_dist_sq:
                pairs.append((i, j))
    return pairs

def partition_subproblems(n, collision_pairs):
    adj = [[] for _ in range(n)]
    for i, j in collision_pairs:
        adj[i].append(j)
        adj[j].append(i)
    visited = [False] * n
    subproblems = []
    for i in range(n):
        if not visited[i]:
            stack = [i]
            comp = []
            while stack:
                v = stack.pop()
                if not visited[v]:
                    visited[v] = True
                    comp.append(v)
                    for u in adj[v]:
                        if not visited[u]:
                            stack.append(u)
            subproblems.append(comp)
    return subproblems

def solve_subproblem(sub_x, outer_cx, outer_cy, outer_angle_deg, inner_radius, local_budget):
    n_local = len(sub_x) // 3
    alpha = 1500.0
    penalty = 300.0
    def sub_loss(x):
        return loss_func(x, n_local, outer_cx, outer_cy, outer_angle_deg, inner_radius, alpha, penalty)
    bounds = []
    for _ in range(n_local):
        bounds.extend([(-6.0, 6.0), (-6.0, 6.0), (0.0, 60.0)])
    bounds = np.array(bounds)
    start_t = time.time()
    try:
        if time.time() - start_t < local_budget * 0.5:
            result_bh = basinhopping(
                sub_loss, sub_x, niter=20, T=0.003, stepsize=0.02,
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 300}},
                seed=43, callback=lambda x, f, accepted: time.time() - start_t > local_budget * 0.5
            )
            sub_x = result_bh.x.copy()
    except Exception:
        pass
    try:
        polish_time = local_budget - (time.time() - start_t)
        if polish_time > 0:
            result_polish = minimize(
                sub_loss, sub_x, method='L-BFGS-B', bounds=bounds,
                options={'ftol': 1e-11, 'gtol': 1e-11, 'maxiter': max(300, int(polish_time * 1000))}
            )
            sub_x = result_polish.x.copy()
    except Exception:
        pass
    return sub_x

def validate_geometry(inner_hex_data, outer_hex_data, outer_side, inner_radius):
    n = inner_hex_data.shape[0]
    outer_cx, outer_cy, outer_angle = outer_hex_data
    outer_lines = outer_hex_lines(outer_cx, outer_cy, outer_angle, outer_side)
    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = inner_hex_data[i, 0] - inner_hex_data[j, 0]
            dy = inner_hex_data[i, 1] - inner_hex_data[j, 1]
            dist = np.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist
            if dist < 2.0 * inner_radius * 0.9999:
                return False
    for i in range(n):
        cx, cy, ang = inner_hex_data[i]
        verts = hex_vertices(cx, cy, ang, inner_radius)
        for v in verts:
            for a, b, c in outer_lines:
                if a * v[0] + b * v[1] + c > 1e-9:
                    return False
    return True

def hexagon_packing_12():
    start_time = time.time()
    n = 12
    inner_radius = 1.0
    outer_cx, outer_cy, outer_angle_deg = 0.0, 0.0, 0.0
    alpha = 1200.0
    penalty = 250.0
    total_budget = 115.0

    inner_hex_data = np.array([
        [0.0, 0.0, 0.0],
        [-2.6, 0.0, 0.0],
        [2.6, 0.0, 0.0],
        [-1.3, 2.25, 0.0],
        [1.3, 2.25, 0.0],
        [-1.3, -2.25, 0.0],
        [1.3, -2.25, 0.0],
        [-3.9, 2.25, 0.0],
        [3.9, 2.25, 0.0],
        [-3.9, -2.25, 0.0],
        [3.9, -2.25, 0.0],
        [0.0, -4.5, 0.0],
    ])
    x0 = inner_hex_data.flatten()

    bound_val = 6.0
    bounds = []
    for _ in range(n):
        bounds.extend([(-bound_val, bound_val), (-bound_val, bound_val), (0.0, 60.0)])
    bounds = np.array(bounds)

    def loss_wrap(x):
        return loss_func(x, n, outer_cx, outer_cy, outer_angle_deg, inner_radius, alpha, penalty)

    best_loss = np.inf
    best_x = x0.copy()

    try:
        elapsed = time.time() - start_time
        if elapsed < total_budget * 0.3:
            bh_niter = 60
            result_bh = basinhopping(
                loss_wrap, x0, niter=bh_niter, T=0.005, stepsize=0.04,
                minimizer_kwargs={'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 600}},
                seed=42, callback=lambda x, f, accepted: time.time() - start_time > total_budget * 0.3
            )
            if result_bh.fun < best_loss:
                best_loss = result_bh.fun
                best_x = result_bh.x.copy()
    except Exception:
        pass

    try:
        elapsed = time.time() - start_time
        if elapsed < total_budget * 0.6:
            da_maxiter = max(1500, int(4000 * (total_budget * 0.6 - elapsed) / 60))
            result_da = dual_annealing(loss_wrap, bounds, maxiter=da_maxiter, seed=42,
                                       callback=lambda x, f, context: time.time() - start_time > total_budget * 0.6)
            if result_da.fun < best_loss:
                best_loss = result_da.fun
                best_x = result_da.x.copy()
    except Exception:
        pass

    collision_pairs = get_collision_pairs(best_x, n, inner_radius, tol=1e-2)
    if collision_pairs:
        subproblems = partition_subproblems(n, collision_pairs)
        remaining_budget = total_budget - (time.time() - start_time)
        per_sub_budget = min(10.0, remaining_budget / max(1, len(subproblems)))
        for sp in subproblems:
            if len(sp) < 2:
                continue
            sp_indices = []
            for idx in sp:
                sp_indices.extend([idx * 3, idx * 3 + 1, idx * 3 + 2])
            sp_indices = np.array(sp_indices)
            sub_x = best_x[sp_indices].copy()
            solved_sub = solve_subproblem(sub_x, outer_cx, outer_cy, outer_angle_deg, inner_radius, per_sub_budget)
            best_x[sp_indices] = solved_sub

    try:
        elapsed = time.time() - start_time
        polish_maxiter = max(1000, int(2500 * (total_budget - elapsed) / 60))
        result_polish = minimize(
            loss_wrap, best_x, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': polish_maxiter}
        )
        if result_polish.fun < best_loss:
            best_loss = result_polish.fun
            best_x = result_polish.x.copy()
    except Exception:
        pass

    best_x = project_feasible(best_x, n)

    min_dist = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            idx_i, idx_j = i * 3, j * 3
            dx = best_x[idx_i] - best_x[idx_j]
            dy = best_x[idx_i + 1] - best_x[idx_j + 1]
            dist = np.hypot(dx, dy)
            if dist < min_dist:
                min_dist = dist

    if min_dist < 2.0 * inner_radius * 0.9999:
        scale_f = (2.0 * inner_radius * 1.001) / min_dist
        for i in range(n):
            idx = i * 3
            best_x[idx] *= scale_f
            best_x[idx + 1] *= scale_f
    else:
        shrink = 0.9998
        for i in range(n):
            idx = i * 3
            best_x[idx] *= shrink
            best_x[idx + 1] *= shrink

    outer_lines = outer_hex_lines(outer_cx, outer_cy, outer_angle_deg, 1.0)
    apothem = 0.0
    for i in range(n):
        idx = i * 3
        cx, cy, ang = best_x[idx], best_x[idx + 1], best_x[idx + 2]
        verts = hex_vertices(cx, cy, ang, inner_radius)
        for v in verts:
            for a, b, c in outer_lines:
                d = -(a * v[0] + b * v[1] + c)
                if d > apothem:
                    apothem = d

    inner_hex_data = np.zeros((n, 3))
    for i in range(n):
        idx = i * 3
        inner_hex_data[i, 0] = best_x[idx]
        inner_hex_data[i, 1] = best_x[idx + 1]
        inner_hex_data[i, 2] = best_x[idx + 2]

    outer_hex_data = np.array([outer_cx, outer_cy, outer_angle_deg])
    outer_side = (2.0 / np.sqrt(3)) * apothem

    if not validate_geometry(inner_hex_data, outer_hex_data, outer_side, inner_radius):
        inner_hex_data = x0.reshape(n, 3)
        outer_lines = outer_hex_lines(outer_cx, outer_cy, outer_angle_deg, 1.0)
        apothem = 0.0
        for i in range(n):
            cx, cy, ang = inner_hex_data[i]
            verts = hex_vertices(cx, cy, ang, inner_radius)
            for v in verts:
                for a, b, c in outer_lines:
                    d = -(a * v[0] + b * v[1] + c)
                    if d > apothem:
                        apothem = d
        outer_side = (2.0 / np.sqrt(3)) * apothem

    return inner_hex_data, outer_hex_data, outer_side

if __name__ == "__main__":
    ih, oh, os = hexagon_packing_12()
    print("1/outer_side =", 1.0 / os)
    print("combined_score =", (1.0 / os) / 0.2537)
    print("eval_time =", "N/A")