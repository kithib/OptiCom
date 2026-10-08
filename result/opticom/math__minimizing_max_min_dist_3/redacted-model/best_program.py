import numpy as np
from scipy.optimize import basinhopping, dual_annealing, minimize


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    n = 14
    d = 3
    dim_x = n * d
    
    # Upper triangle indices for pairwise distances (i < j) - computed once
    triu_idx = np.triu_indices(n, k=1)
    
    # TRUE objective function (for final evaluation)
    def true_objective(x_flat):
        pts = x_flat.reshape(n, d)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=2)
        dists = np.sqrt(np.maximum(sq_dists[triu_idx], 1e-16))
        dmin = np.min(dists)
        dmax = np.max(dists)
        return dmin / dmax
    
    # Cuboctahedron vertices (12 points) - permutations of (±1, ±1, 0)
    cuboctahedron = np.array([
        [1, 1, 0], [1, -1, 0], [-1, 1, 0], [-1, -1, 0],
        [1, 0, 1], [1, 0, -1], [-1, 0, 1], [-1, 0, -1],
        [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1]
    ])
    axial_scale = 1.3
    axial_points = np.array([
        [0, 0, axial_scale],
        [0, 0, -axial_scale]
    ])
    x0_points = np.vstack([cuboctahedron, axial_points])
    max_radius = np.max(np.linalg.norm(x0_points, axis=1))
    x0_points = x0_points / max_radius
    x0 = x0_points.flatten()
    
    # LogSumExp smoothing base parameter
    alpha_base = 600.0
    penalty = 150.0
    
    def compute_pairwise_sq_dists(x_flat):
        pts = x_flat.reshape(n, d)
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diffs**2, axis=2)
        return sq_dists[triu_idx]
    
    def loss(x_flat, alpha=alpha_base):
        sq_dists = compute_pairwise_sq_dists(x_flat)
        dists = np.sqrt(np.maximum(sq_dists, 1e-16))
        
        # Smooth dmax using LogSumExp: smooth_max(d) ≈ LSE(d, alpha)/alpha
        log_sum_exp_max = np.log(np.sum(np.exp(alpha * dists))) / alpha
        
        # Smooth dmin using LogSumExp on negatives: smooth_min(d) ≈ -LSE(-d, alpha)/alpha
        log_sum_exp_min = -np.log(np.sum(np.exp(-alpha * dists))) / alpha
        
        smooth_objective = log_sum_exp_min / log_sum_exp_max
        
        # Constraint violation: keep points in unit bounding sphere to constrain search
        pts = x_flat.reshape(n, d)
        sq_norms = np.sum(pts**2, axis=1)
        constraint_viol = np.sum(np.maximum(sq_norms - 1.0, 0.0)**2)
        
        return -smooth_objective + penalty * constraint_viol
    
    # Bounds: keep within [-1.3, 1.3] to allow optimization near unit sphere
    bounds = [(-1.3, 1.3)] * dim_x
    
    # Multi-start candidates
    starts = []
    starts.append(x0)
    
    # 12-point equatorial ring + 2 poles
    ring_angles = np.linspace(0, 2*np.pi, 12, endpoint=False)
    ring_pts = np.column_stack([np.cos(ring_angles), np.sin(ring_angles), np.zeros(12)])
    ring_14 = np.vstack([ring_pts, np.array([[0, 0, 1], [0, 0, -1]])])
    ring_14 = ring_14 / np.max(np.linalg.norm(ring_14, axis=1))
    starts.append(ring_14.flatten())
    
    # Hexagonal bipyramid variant: 6 top hexagon, 6 bottom hexagon, 2 poles
    hex_angles = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hex_top = np.column_stack([np.cos(hex_angles), np.sin(hex_angles), 0.7 * np.ones(6)])
    hex_bot = np.column_stack([np.cos(hex_angles + np.pi/6), np.sin(hex_angles + np.pi/6), -0.7 * np.ones(6)])
    hex_poles = np.array([[0, 0, 1.2], [0, 0, -1.2]])
    hex_14 = np.vstack([hex_top, hex_bot, hex_poles])
    hex_14 = hex_14 / np.max(np.linalg.norm(hex_14, axis=1))
    starts.append(hex_14.flatten())
    
    # Additional starts with Gaussian noise around current best
    rng = np.random.default_rng(seed=42)
    for _ in range(8):
        starts.append(x0 + rng.normal(0.0, 0.02, size=dim_x))
    
    # Run optimization for all multi-start candidates first
    best_ratio = -1
    best_x = None
    for s in starts:
        opt_result = minimize(
            loss, s,
            method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1500}
        )
        current_ratio = true_objective(opt_result.x)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_x = opt_result.x
    
    # 1. Run basinhopping starting from the best multi-start point
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds
    }
    bh_result = basinhopping(
        loss, best_x,
        niter=80, T=0.005, stepsize=0.05,
        minimizer_kwargs=minimizer_kwargs,
        seed=42
    )
    x_bh = bh_result.x
    loss_bh = bh_result.fun
    
    # 2. Run dual_annealing as global fallback
    da_result = dual_annealing(
        lambda x: loss(x, alpha=500.0), bounds,
        maxiter=2800, seed=42
    )
    x_da = da_result.x
    loss_da = loss(x_da)
    
    # 3. Pick winner between global searches
    if loss_bh < loss_da:
        x_candidate = x_bh
    else:
        x_candidate = x_da
    
    # 4. Polish with tight L-BFGS-B
    polish_result = minimize(
        lambda x: loss(x, alpha=800.0), x_candidate,
        method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 2000}
    )
    x_opt = polish_result.x
    
    # 5. Project to feasible region: unit bounding sphere
    points_opt = x_opt.reshape(n, d)
    radii = np.linalg.norm(points_opt, axis=1, keepdims=True)
    max_radius = np.maximum(radii.max(), 1e-10)
    points_proj = points_opt / max_radius
    
    return points_proj