import numpy as np
from itertools import combinations
from scipy.optimize import basinhopping, dual_annealing, minimize
import warnings
warnings.filterwarnings('ignore')


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    np.random.seed(42)
    rng = np.random.default_rng(seed=42)
    
    # Precompute triangle indices once for efficiency
    triangle_indices = np.array(list(combinations(range(n), 3)), dtype=np.int64)
    i_idx = triangle_indices[:, 0]
    j_idx = triangle_indices[:, 1]
    k_idx = triangle_indices[:, 2]
    
    def compute_all_areas(pts_flat):
        """Compute all 286 triangle areas efficiently."""
        pts = pts_flat.reshape(n, 2)
        a = pts[i_idx]
        b = pts[j_idx]
        c = pts[k_idx]
        areas = 0.5 * np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - 
                              (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        return areas
    
    def compute_convex_hull_area(pts):
        """Compute convex hull area using Graham scan and shoelace formula."""
        if len(pts) < 3:
            return 1.0
        
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        
        # Graham scan
        hull_pts = pts[np.lexsort((pts[:, 1], pts[:, 0]))]
        lower = []
        for p in hull_pts:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-10:
                lower.pop()
            lower.append(p)
        upper = []
        for p in reversed(hull_pts):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-10:
                upper.pop()
            upper.append(p)
        hull = np.array(lower[:-1] + upper[:-1])
        
        if len(hull) < 3:
            return 1.0
        
        # Shoelace formula
        x_hull, y_hull = hull[:, 0], hull[:, 1]
        return 0.5 * np.abs(np.dot(x_hull, np.roll(y_hull, 1)) - np.dot(y_hull, np.roll(x_hull, 1)))
    
    def true_objective(x):
        """Non-smoothed true objective for final reporting."""
        pts = x.reshape(n, 2)
        areas = compute_all_areas(x)
        min_area = np.min(areas)
        hull_area = compute_convex_hull_area(pts)
        return min_area / max(hull_area, 1e-10)
    
    # Smooth differentiable loss function
    alpha = 1000.0
    penalty_weight = 150.0
    
    def loss(x):
        """loss(x) = -smooth_objective + penalty*constraint_violations"""
        pts = x.reshape(n, 2)
        areas = compute_all_areas(x) + 1e-14
        
        # Numerically stable LogSumExp smooth approximation to min
        lse_input = -alpha * areas
        max_lse = np.max(lse_input)
        lse = np.log(np.sum(np.exp(lse_input - max_lse))) + max_lse
        smooth_min_area = -lse / alpha
        
        hull_area = compute_convex_hull_area(pts) + 1e-12
        smooth_objective_val = smooth_min_area / hull_area
        
        # Bound constraint violations
        lb_viol = np.maximum(0.0, -x)
        ub_viol = np.maximum(0.0, x - 1.0)
        bound_penalty = np.sum(lb_viol**2 + ub_viol**2)
        
        # Point degeneracy penalty
        dists = np.sqrt(np.sum((pts[:, np.newaxis] - pts[np.newaxis, :])**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        degeneracy_penalty = np.exp(-150.0 * min_dist) if min_dist < 0.025 else 0.0
        
        return -smooth_objective_val + penalty_weight * (bound_penalty + degeneracy_penalty)
    
    # First: Run incumbent strategy to get best known parameter vector
    points = np.zeros((n, 2))
    idx = 0
    sqrt_n = int(np.ceil(np.sqrt(n)))
    scale = 0.95 / (sqrt_n - 0.5)
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if idx >= n:
                break
            x_coord = j * scale
            y_coord = i * scale * np.sqrt(3) / 2
            if i % 2 == 1:
                x_coord += scale / 2
            points[idx] = [x_coord + rng.random() * 0.01 - 0.005, 
                          y_coord + rng.random() * 0.01 - 0.005]
            idx += 1
    points = points[:n]
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    points = (points - min_coords) / (max_coords - min_coords) * 0.98 + 0.01
    x0 = points.flatten()
    
    bounds = [(0.0, 1.0)] * (2 * n)
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
    
    # Get incumbent best solution
    try:
        np.random.seed(42)
        bh_result = basinhopping(
            loss, x0, niter=100, T=0.005, stepsize=0.05,
            minimizer_kwargs=minimizer_kwargs, seed=42
        )
        x_bh, loss_bh = bh_result.x, bh_result.fun
    except Exception:
        x_bh, loss_bh = x0, loss(x0)
    
    try:
        da_result = dual_annealing(loss, bounds, maxiter=3500, seed=42)
        x_da, loss_da = da_result.x, da_result.fun
    except Exception:
        x_da, loss_da = x0, loss(x0)
    
    x_candidate = x_bh if loss_bh <= loss_da else x_da
    
    polish_result = minimize(
        loss, x_candidate, method='L-BFGS-B', bounds=bounds,
        options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 3000}
    )
    x_warm_start = np.clip(polish_result.x, 0.0, 1.0)
    
    # === MULTI-START REFINEMENT ===
    all_starts = []
    
    # 1. Primary warm-start
    all_starts.append(x_warm_start.copy())
    
    # 2. 16 additional starts with Gaussian noise (std=0.02)
    for _ in range(16):
        noisy = x_warm_start + rng.normal(0.0, 0.02, size=x_warm_start.shape)
        noisy = np.clip(noisy, 0.0, 1.0)
        all_starts.append(noisy)
    
    # 3. 4 fresh starts from symmetry-aware priors
    # Prior 1: Hexagonal lattice
    hex_pts = np.zeros((n, 2))
    idx = 0
    for i in range(4):
        for j in range(4):
            if idx >= n:
                break
            x = j * 0.25
            y = i * 0.25 * np.sqrt(3) / 2
            if i % 2 == 1:
                x += 0.125
            hex_pts[idx] = [x + rng.random()*0.02-0.01, y + rng.random()*0.02-0.01]
            idx += 1
    hex_pts = np.clip(hex_pts, 0.0, 1.0)
    all_starts.append(hex_pts.flatten())
    
    # Prior 2: Ring pattern with inner points
    ring_pts = np.zeros((n, 2))
    # Outer ring with 8 points
    for i in range(8):
        theta = 2 * np.pi * i / 8
        ring_pts[i] = [0.5 + 0.4 * np.cos(theta), 0.5 + 0.4 * np.sin(theta)]
    # Inner 5 points
    for i in range(5):
        theta = 2 * np.pi * i / 5 + np.pi/5
        ring_pts[8+i] = [0.5 + 0.18 * np.cos(theta), 0.5 + 0.18 * np.sin(theta)]
    ring_pts += rng.random(ring_pts.shape) * 0.02 - 0.01
    ring_pts = np.clip(ring_pts, 0.0, 1.0)
    all_starts.append(ring_pts.flatten())
    
    # Prior 3: Triangular lattice
    tri_pts = np.zeros((n, 2))
    idx = 0
    for i in range(5):
        for j in range(i+1):
            if idx >= n:
                break
            tri_pts[idx] = [0.1 + j*0.2 + (5-i)*0.1, 0.1 + i*0.18]
            idx += 1
    tri_pts += rng.random(tri_pts.shape) * 0.03 - 0.015
    tri_pts = np.clip(tri_pts, 0.0, 1.0)
    all_starts.append(tri_pts.flatten())
    
    # Prior 4: Square lattice perturbation
    sq_pts = np.zeros((n, 2))
    for i in range(n):
        sq_pts[i, 0] = (i % 4) * 0.26 + 0.05
        sq_pts[i, 1] = (i // 4) * 0.26 + 0.05
    sq_pts += rng.random(sq_pts.shape) * 0.04 - 0.02
    sq_pts = np.clip(sq_pts, 0.0, 1.0)
    all_starts.append(sq_pts.flatten())
    
    # Prior 5: Hexagonal lattice (from top exemplars)
    hex_pts = np.zeros((n, 2))
    idx = 0
    for i in range(4):
        for j in range(4):
            if idx >= n: break
            x = j * 0.24
            y = i * 0.21
            if i % 2 == 1: x += 0.12
            hex_pts[idx] = [x+0.03, y+0.06] + rng.random(2)*0.01-0.005
            idx += 1
    all_starts.append(hex_pts.flatten())

    # Run L-BFGS-B on each start
    best_score = -float('inf')
    best_x = None
    
    for start_idx, start_x in enumerate(all_starts):
        result = minimize(
            loss, start_x, method='L-BFGS-B', bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': 2500}
        )
        x_opt = np.clip(result.x, 0.0, 1.0)
        score = true_objective(x_opt)
        if score > best_score:
            best_score = score
            best_x = x_opt
    
    final_points = best_x.reshape(n, 2)
    return final_points


if __name__ == "__main__":
    pts = heilbronn_convex13()
    print("Points shape:", pts.shape)
    
    from scipy.spatial import ConvexHull
    hull = ConvexHull(pts)
    hull_area = hull.volume
    print("Convex hull area:", hull_area)
    
    min_area = float('inf')
    for i, j, k in combinations(range(13), 3):
        area = 0.5 * abs(
            (pts[j,0] - pts[i,0]) * (pts[k,1] - pts[i,1]) - 
            (pts[j,1] - pts[i,1]) * (pts[k,0] - pts[i,0])
        )
        if area < min_area:
            min_area = area
    print("Min triangle area:", min_area)
    norm_score = min_area / hull_area
    print("Normalized score:", norm_score)
    print("Combined score:", norm_score / 0.030936889034895654)