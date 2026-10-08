"""Constructor-based circle packing for n=26 circles"""
import numpy as np
from scipy.optimize import basinhopping, dual_annealing, minimize

np.random.seed(42)
n_circles = 26
n_vars = n_circles * 3  # [x_0, y_0, r_0, ..., x_{25}, y_{25}, r_{25}]
alpha_smooth = 800.0
penalty = 5e3


def _unpack(x):
    x_flat = np.asarray(x, dtype=np.float64).reshape(-1)
    xs = x_flat[0::3]
    ys = x_flat[1::3]
    rs = x_flat[2::3]
    return xs, ys, rs


def _smooth_max(vals, alpha):
    m = np.max(vals)
    shifted = vals - m
    return m + np.log(np.sum(np.exp(alpha * shifted))) / alpha


def _smooth_min(vals, alpha):
    return -_smooth_max(-vals, alpha)


def _loss(x):
    xs, ys, rs = _unpack(x)
    sum_r = np.sum(rs)
    violations = []
    # Border constraints: r + x <= 1, r - x <= 0, r + y <= 1, r - y <= 0
    violations.append(rs + xs - 1.0)
    violations.append(rs - xs)
    violations.append(rs + ys - 1.0)
    violations.append(rs - ys)
    # Pairwise constraints: r_i + r_j - d_{ij} <= 0
    x_arr = xs[:, None]
    y_arr = ys[:, None]
    dx = x_arr - x_arr.T
    dy = y_arr - y_arr.T
    d_sq = dx * dx + dy * dy
    d = np.sqrt(d_sq + 1e-18)
    # upper triangle pairs
    i_upper, j_upper = np.triu_indices(n_circles, k=1)
    pair_viol = rs[i_upper] + rs[j_upper] - d[i_upper, j_upper]
    violations.append(pair_viol)
    all_v = np.concatenate(violations)
    smoothed_max_v = _smooth_max(all_v, alpha_smooth)
    penalty_term = penalty * (np.maximum(smoothed_max_v, 0.0)) ** 2
    return -sum_r + penalty_term


def _true_objective_and_feasibility(x):
    xs, ys, rs = _unpack(x)
    sum_r = np.sum(rs)
    eps = 1e-9
    max_v = 0.0
    if np.any(rs < -eps):
        return sum_r, False, np.inf
    for i in range(n_circles):
        v1 = rs[i] + xs[i] - 1.0
        v2 = rs[i] - xs[i]
        v3 = rs[i] + ys[i] - 1.0
        v4 = rs[i] - ys[i]
        max_v = max(max_v, v1, v2, v3, v4)
    for i in range(n_circles):
        for j in range(i + 1, n_circles):
            dx = xs[i] - xs[j]
            dy = ys[i] - ys[j]
            d = np.sqrt(dx * dx + dy * dy)
            v = rs[i] + rs[j] - d
            if v > max_v:
                max_v = v
    feasible = max_v <= 1e-6
    return sum_r, feasible, max_v


def _heuristic_initial():
    x0 = np.zeros(n_vars)
    r_hex = 1.0 / (4 * np.sqrt(3) + 2)
    row_spacing = np.sqrt(3) * r_hex
    idx = 0
    for row in range(5):
        y = r_hex + row * row_spacing
        for col in range(5):
            x_pos = r_hex + col * (2 * r_hex)
            x0[3 * idx] = x_pos
            x0[3 * idx + 1] = y
            x0[3 * idx + 2] = r_hex * 0.9
            idx += 1
    x0[3 * 25] = 0.5 + r_hex
    x0[3 * 25 + 1] = 0.5
    x0[3 * 25 + 2] = r_hex * 0.8
    return x0


def construct_packing():
    """
    Construct a specific arrangement of 26 circles in a unit square
    that attempts to maximize the sum of their radii.

    Returns:
        Tuple of (centers, radii, sum_of_radii)
        centers: np.array of shape (26, 2) with (x, y) coordinates
        radii: np.array of shape (26) with radius of each circle
        sum_of_radii: Sum of all radii
    """
    bounds = [(0.0, 1.0), (0.0, 1.0), (0.0, 0.5)] * n_circles
    x0 = _heuristic_initial()

    # Strategy 1: basinhopping from heuristic start
    bh_kwargs = {
        "method": "L-BFGS-B",
        "bounds": bounds,
        "options": {"maxiter": 600, "ftol": 1e-9, "gtol": 1e-8},
    }
    try:
        res_bh = basinhopping(
            _loss,
            x0,
            niter=60,
            T=0.005,
            stepsize=0.04,
            minimizer_kwargs=bh_kwargs,
            seed=42,
            niter_success=15,
            interval=20,
        )
        x_bh = res_bh.x
        loss_bh = float(res_bh.fun)
    except Exception:
        x_bh = x0.copy()
        loss_bh = float(_loss(x0))

    # Strategy 2: dual_annealing global search
    try:
        res_da = dual_annealing(
            _loss,
            bounds,
            maxiter=1200,
            seed=42,
            initial_temp=2.0,
            restart_temp_ratio=1e-5,
            maxfun=150000,
        )
        x_da = res_da.x
        loss_da = float(res_da.fun)
    except Exception:
        x_da = x0.copy()
        loss_da = float(_loss(x0))

    # Pick winner of two strategies
    if loss_bh <= loss_da:
        x_candidate = x_bh.copy()
    else:
        x_candidate = x_da.copy()

    # Polish candidate tightly with L-BFGS-B
    try:
        polish_res = minimize(
            _loss,
            x_candidate,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-12, "gtol": 1e-10, "maxiter": 2500, "maxcor": 50},
        )
        x_polish = polish_res.x
    except Exception:
        x_polish = x_candidate.copy()

    # Project to feasible region: clip centers/radii to bounds, shrink radii iteratively
    xs, ys, rs = _unpack(x_polish)
    xs = np.clip(xs, 1e-8, 1 - 1e-8)
    ys = np.clip(ys, 1e-8, 1 - 1e-8)
    rs = np.clip(rs, 1e-9, None)

    # Border-bound radius limit
    for i in range(n_circles):
        rb = min(xs[i], ys[i], 1 - xs[i], 1 - ys[i])
        if rs[i] > rb:
            rs[i] = max(rb, 1e-9)

    # Iterative pairwise constraint enforcement (non-smoothed true constraints)
    for _ in range(50):
        changed = False
        for i in range(n_circles):
            for j in range(i + 1, n_circles):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                d = np.sqrt(dx * dx + dy * dy)
                if rs[i] + rs[j] > d + 1e-12:
                    scale = d / (rs[i] + rs[j])
                    rs[i] *= scale
                    rs[j] *= scale
                    changed = True
        if not changed:
            break

    # Build flat vector to report true objective
    x_final = np.zeros(n_vars)
    for i in range(n_circles):
        x_final[3 * i] = xs[i]
        x_final[3 * i + 1] = ys[i]
        x_final[3 * i + 2] = rs[i]

    sum_r, feasible, max_v = _true_objective_and_feasibility(x_final)
    # Fallback safeguard: if still infeasible, shrink radii uniformly
    if not feasible:
        shrink_factor = 1.0 - max(0.0, max_v) * 1.1
        shrink_factor = max(min(shrink_factor, 1.0), 0.01)
        rs = rs * shrink_factor
        sum_r = float(np.sum(rs))

    centers = np.column_stack([xs, ys])
    return centers, rs, sum_r


def compute_max_radii(centers):
    """
    Compute the maximum possible radii for each circle position
    such that they don't overlap and stay within the unit square.

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates

    Returns:
        np.array of shape (n) with radius of each circle
    """
    n = centers.shape[0]
    radii = np.ones(n)

    for i in range(n):
        x, y = centers[i]
        radii[i] = min(x, y, 1 - x, 1 - y)

    for _ in range(20):
        updated = False
        for i in range(n):
            for j in range(i + 1, n):
                dist = np.sqrt(np.sum((centers[i] - centers[j]) ** 2))
                if radii[i] + radii[j] > dist:
                    scale = dist / (radii[i] + radii[j])
                    radii[i] *= scale
                    radii[j] *= scale
                    updated = True
        if not updated:
            break

    return radii


def run_packing():
    """Run the circle packing constructor for n=26"""
    centers, radii, sum_radii = construct_packing()
    return centers, radii, sum_radii


def visualize(centers, radii):
    """
    Visualize the circle packing

    Args:
        centers: np.array of shape (n, 2) with (x, y) coordinates
        radii: np.array of shape (n) with radius of each circle
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(True)

    for i, (center, radius) in enumerate(zip(centers, radii)):
        circle = Circle(center, radius, alpha=0.5)
        ax.add_patch(circle)
        ax.text(center[0], center[1], str(i), ha="center", va="center")

    plt.title(f"Circle Packing (n={len(centers)}, sum={sum(radii):.6f})")
    plt.show()


if __name__ == "__main__":
    centers, radii, sum_radii = run_packing()
    print(f"Sum of radii: {sum_radii:.6f}")
    # AlphaEvolve improved this to 2.635
    # visualize(centers, radii)