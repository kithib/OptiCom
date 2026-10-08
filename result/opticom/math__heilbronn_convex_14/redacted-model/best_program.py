import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import basinhopping, dual_annealing, minimize

def _compute_all_triangle_areas(points):
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                area = 0.5 * np.abs(
                    points[i,0] * (points[j,1] - points[k,1]) +
                    points[j,0] * (points[k,1] - points[i,1]) +
                    points[k,0] * (points[i,1] - points[j,1])
                )
                areas.append(area)
    return np.array(areas)

def _smooth_min(areas, alpha=2000.0):
    return -np.log(np.sum(np.exp(-alpha * areas))) / alpha

def _convex_hull_constraint_violations(points):
    try:
        hull = ConvexHull(points)
        eq = hull.equations
        violations = np.maximum(0.0, np.dot(points, eq[:,:-1].T) + eq[:,-1])
        return violations.flatten()
    except:
        return np.array([100.0] * len(points))

def _loss(x, n):
    points = x.reshape((n, 2))
    areas = _compute_all_triangle_areas(points)
    smooth_min_area = _smooth_min(areas)
    violations = _convex_hull_constraint_violations(points)
    penalty = 10000.0
    return -smooth_min_area + penalty * np.sum(violations**2)

def heilbronn_convex14() -> np.ndarray:
    n = 14
    np.random.seed(42)
    rng = np.random.default_rng(seed=42)
    
    # Initial configuration: regular polygon + Halton sequence
    boundary_count = 8
    angles = np.linspace(0, 2*np.pi, boundary_count, endpoint=False)
    boundary_points = np.column_stack((np.cos(angles), np.sin(angles)))
    hull = ConvexHull(boundary_points)
    scale = np.sqrt(1.0 / hull.volume)
    boundary_points *= scale
    
    def halton(n, base):
        seq = []
        for i in range(n):
            num, frac = i+1, 1.0/base
            val = 0.0
            while num > 0:
                val += (num % base) * frac
                num = num // base
                frac /= base
            seq.append(val)
        return np.array(seq)
    
    inner_count = n - boundary_count
    halton_x = halton(inner_count, 2)
    halton_y = halton(inner_count, 3)
    inner_points = np.column_stack((halton_x, halton_y))
    
    # Normalize inner points to fit within boundary convex hull
    inner_points = (inner_points - np.min(inner_points, axis=0)) / (np.max(inner_points, axis=0) - np.min(inner_points, axis=0))
    inner_points = inner_points * (np.max(boundary_points, axis=0) - np.min(boundary_points, axis=0)) + np.min(boundary_points, axis=0)
    
    initial_points = np.vstack([boundary_points, inner_points])
    
    # Add small jitter to break symmetry
    initial_points += rng.normal(0, 0.01, initial_points.shape)
    
    # Create optimization bounds
    min_x, max_x = np.min(initial_points[:,0]), np.max(initial_points[:,0])
    min_y, max_y = np.min(initial_points[:,1]), np.max(initial_points[:,1])
    bound_width_x = max_x - min_x
    bound_width_y = max_y - min_y
    bounds = [
        (min_x - 0.1*bound_width_x, max_x + 0.1*bound_width_x),
        (min_y - 0.1*bound_width_y, max_y + 0.1*bound_width_y)
    ] * n
    
    # Multi-start optimization
    x0 = initial_points.ravel()
    
    # Basinhopping
    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'bounds': bounds,
        'args': (n,),
        'options': {'ftol': 1e-8, 'gtol': 1e-8}
    }
    bh_result = basinhopping(
        _loss, x0, niter=50, T=0.005, stepsize=0.05,
        minimizer_kwargs=minimizer_kwargs, seed=42
    )
    
    # Dual annealing
    da_result = dual_annealing(
        _loss, bounds, args=(n,), maxiter=2000,
        seed=42, initial_temp=5000.0, visit=2.62
    )
    
    # Select best candidate
    if bh_result.fun < da_result.fun:
        best_x = bh_result.x
    else:
        best_x = da_result.x
    
    # Final polishing
    polish_result = minimize(
        _loss, best_x, args=(n,), method='L-BFGS-B',
        bounds=bounds, options={'ftol':1e-10, 'gtol':1e-10, 'maxiter':2000}
    )
    best_x = polish_result.x
    
    # Project to feasible region
    points = best_x.reshape((n,2))
    try:
        hull = ConvexHull(points)
        eq = hull.equations
        violations = np.dot(points, eq[:,:-1].T) + eq[:,-1]
        mask = violations > 1e-6
        if np.any(mask):
            for i in np.where(mask)[0]:
                max_viol_idx = np.argmax(violations[i])
                proj_dir = eq[max_viol_idx, :-1]
                proj_dir /= np.linalg.norm(proj_dir)
                points[i] -= violations[i, max_viol_idx] * proj_dir
    except:
        pass
    
    # Normalize convex hull to unit area
    try:
        hull = ConvexHull(points)
        current_area = hull.volume
        if current_area > 1e-9:
            scale_factor = np.sqrt(1.0 / current_area)
            points *= scale_factor
    except:
        pass
    
    return points