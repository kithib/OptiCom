import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import numba

@numba.njit
def triangle_area_numba(p1, p2, p3):
    """Numba-accelerated triangle area calculation using cross product."""
    return 0.5 * np.abs((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0]))

@numba.njit
def compute_all_triangle_areas_numba(points_flat, n):
    """Numba-accelerated computation of all triangle areas."""
    points = points_flat.reshape((n, 2))
    n_tri = n * (n - 1) * (n - 2) // 6
    areas = np.zeros(n_tri, dtype=np.float64)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                areas[idx] = triangle_area_numba(points[i], points[j], points[k])
                idx += 1
    return areas

@numba.njit
def get_min_area_numba(points_flat, n):
    """Get actual minimum triangle area without approximation."""
    areas = compute_all_triangle_areas_numba(points_flat, n)
    return np.min(areas)

def objective_lse_weighted(points_flat, n, alpha=100.0):
    """
    Smooth objective using LogSumExp approximation of minimum, normalized by hull area.
    Maximizes the minimum triangle area normalized by convex hull area.
    """
    areas = compute_all_triangle_areas_numba(points_flat, n)
    points = points_flat.reshape((n, 2))
    
    # Compute convex hull area for normalization
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume if hasattr(hull, 'volume') else hull.area
        if hull_area < 1e-10:
            hull_area = 1.0
    except:
        hull_area = 1.0
    
    # LogSumExp approximation for minimum
    eps = 1e-15
    areas_safe = np.maximum(areas, eps)
    lse = np.log(np.sum(np.exp(-alpha * areas_safe)))
    min_approx = -lse / alpha
    normalized = min_approx / hull_area
    return -normalized

def generate_hexagonal_lattice(n, scale=0.92):
    """Generate n points from a hexagonal lattice - good uniform distribution."""
    points = []
    rows = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    spacing = scale / (rows + 1)
    count = 0
    for i in range(rows):
        for j in range(rows + 1):
            x = 0.04 + (j + 0.5 * (i % 2)) * spacing
            y = 0.04 + i * spacing * np.sqrt(3) / 2
            points.append([x, y])
            count += 1
            if count >= n:
                return np.array(points)
    return np.array(points)[:n]

def generate_3fold_symmetric(n=13):
    """Generate 3-fold symmetric configuration: 1 center + 4 rings: 3+3+3+3 = 13 total."""
    rng = np.random.default_rng(seed=100)
    points = []
    points.append([0.5, 0.5])
    for layer_idx, (radius, count) in enumerate([(0.18, 3), (0.29, 3), (0.39, 3), (0.47, 3)]):
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + layer_idx * np.pi / 9
        for angle in angles:
            x = 0.5 + radius * np.cos(angle)
            y = 0.5 + radius * np.sin(angle)
            points.append([x, y])
    pts = np.array(points[:n])
    pts += rng.normal(0, 0.004, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def generate_force_relaxed(n=13, seed=200):
    """Generate configuration using force-based repulsion relaxation."""
    rng = np.random.default_rng(seed=seed)
    pts = rng.random((n, 2)) * 0.88 + 0.06
    for _ in range(120):
        forces = np.zeros_like(pts)
        for i in range(n):
            for j in range(i + 1, n):
                diff = pts[i] - pts[j]
                dist = np.linalg.norm(diff)
                if dist > 0.01:
                    force = diff / (dist ** 2) * 0.0012
                    forces[i] += force
                    forces[j] -= force
        pts = np.clip(pts + forces, 0.02, 0.98)
    return pts

def generate_grid_perturbed(n=13, seed=300):
    """Generate perturbed regular grid."""
    rng = np.random.default_rng(seed=seed)
    rows = int(np.ceil(np.sqrt(n)))
    pts = []
    for i in range(rows):
        for j in range(rows):
            pts.append([(j + 0.5) / rows, (i + 0.5) / rows])
    pts = np.array(pts[:n])
    pts *= 0.88
    pts += 0.06
    pts += rng.normal(0, 0.018, pts.shape)
    return np.clip(pts, 0.02, 0.98)

def normalize_unit_hull_area(points):
    """Normalize so convex hull has area close to 1 within bounds."""
    try:
        hull = ConvexHull(points)
        area = hull.volume
        if area > 0:
            scale = np.sqrt(0.88 / area)
            centroid = np.mean(points, axis=0)
            points = centroid + (points - centroid) * scale
            points = np.clip(points, 0.01, 0.99)
    except:
        pass
    return points

def heilbronn_convex13() -> np.ndarray:
    """
    Construct optimal arrangement of 13 points using multi-start L-BFGS-B optimization
    with LogSumExp smoothed objective and continuation method for sharper minima.

    Returns:
        points: np.ndarray of shape (13,2) with x,y coordinates.
    """
    n = 13
    np.random.seed(8675309)
    
    # Generate diverse initial candidate configurations
    init_candidates = [
        generate_hexagonal_lattice(n),
        generate_3fold_symmetric(n),
        generate_force_relaxed(n, seed=111),
        generate_grid_perturbed(n, seed=222),
        generate_force_relaxed(n, seed=333),
        generate_hexagonal_lattice(n) + np.random.normal(0, 0.02, (n, 2)),
    ]
    
    bounds = [(0.01, 0.99) for _ in range(2 * n)]
    best_overall_points = None
    best_overall_score = -np.inf
    
    # Optimize each candidate from diverse initializations
    for idx, x0 in enumerate(init_candidates):
        x0_flat = np.clip(x0.flatten(), 0.01, 0.99)
        current_x = x0_flat.copy()
        
        # Continuation method: progressively sharper minimum approximation with increasing alpha
        alphas = [30.0, 60.0, 100.0, 150.0]
        for alpha in alphas:
            result = minimize(
                objective_lse_weighted,
                current_x,
                args=(n, alpha),
                method="L-BFGS-B",
                bounds=bounds,
                options={"maxiter": 800, "ftol": 1e-11, "gtol": 1e-9}
            )
            current_x = result.x.copy()
            
            points_eval = current_x.reshape((n, 2))
            try:
                hull = ConvexHull(points_eval)
                hull_area = hull.volume
            except:
                hull_area = 1.0
            actual_min = get_min_area_numba(current_x, n)
            score = actual_min / hull_area if hull_area > 1e-10 else 0
            
            if score > best_overall_score:
                best_overall_score = score
                best_overall_points = points_eval.copy()
    
    # Final Nelder-Mead polishing on actual objective for best configuration
    if best_overall_points is not None:
        def pure_min_objective(pts_flat):
            points_eval = pts_flat.reshape((n, 2))
            try:
                hull = ConvexHull(points_eval)
                hull_area = hull.volume
            except:
                hull_area = 1.0
            actual_min = get_min_area_numba(pts_flat, n)
            return - (actual_min / hull_area) if hull_area > 1e-10 else 1e6
        
        result_nm = minimize(
            pure_min_objective,
            best_overall_points.flatten(),
            method="Nelder-Mead",
            options={"maxiter": 300, "fatol": 1e-10}
        )
        best_overall_points = result_nm.x.reshape((n, 2))
    else:
        best_overall_points = generate_hexagonal_lattice(n)
    
    return normalize_unit_hull_area(best_overall_points)