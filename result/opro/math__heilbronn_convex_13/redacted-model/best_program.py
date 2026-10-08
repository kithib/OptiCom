import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize


def triangle_area(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> float:
    """Compute twice the area of triangle with vertices p, q, r."""
    return 0.5 * np.abs((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))


def compute_min_area(points: np.ndarray) -> float:
    """Compute the area of the smallest triangle formed by any three points."""
    n = points.shape[0]
    min_area = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                area = triangle_area(points[i], points[j], points[k])
                if area < min_area:
                    min_area = area
    return min_area


def project_to_unit_square(points: np.ndarray) -> np.ndarray:
    """Project points to [0,1]x[0,1] while maintaining relative spacing."""
    points = points.copy()
    points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
    points[:, 1] = np.clip(points[:, 1], 0.0, 1.0)
    return points


def generate_hexagonal_lattice(n: int, seed: int = 12345) -> np.ndarray:
    """Generate points on a perturbed hexagonal lattice for good initial distribution."""
    rng = np.random.default_rng(seed=seed)
    points = []
    
    m = int(np.ceil(np.sqrt(n * 2 / np.sqrt(3))))
    dx = 1.0 / (m - 1) if m > 1 else 1.0
    dy = dx * np.sqrt(3) / 2
    
    for i in range(m):
        for j in range(m):
            x = i * dx
            y = j * dy
            if i % 2 == 1:
                y += dy / 2
            points.append([x, y])
    
    points = np.array(points)
    points += rng.normal(0, dx * 0.1, size=points.shape)
    points = project_to_unit_square(points)
    
    if len(points) > n:
        selected = [rng.integers(len(points))]
        for _ in range(n - 1):
            dists = np.min([np.sum((points - points[i])**2, axis=1) for i in selected], axis=0)
            selected.append(np.argmax(dists))
        points = points[selected]
    
    return points[:n]


def triangle_areas_vectorized(pts_flat):
    """Vectorized calculation of all triangle areas."""
    pts = pts_flat.reshape(-1, 2)
    n = len(pts)
    i_arr, j_arr, k_arr = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                i_arr.append(i)
                j_arr.append(j)
                k_arr.append(k)
    i_arr, j_arr, k_arr = np.array(i_arr), np.array(j_arr), np.array(k_arr)
    
    areas = 0.5 * np.abs(
        (pts[j_arr, 0] - pts[i_arr, 0]) * (pts[k_arr, 1] - pts[i_arr, 1]) -
        (pts[j_arr, 1] - pts[i_arr, 1]) * (pts[k_arr, 0] - pts[i_arr, 0])
    )
    return areas


def objective_lse(pts_flat):
    """Smooth objective using LogSumExp for better gradient properties."""
    areas = triangle_areas_vectorized(pts_flat)
    p = 60.0  # Increased sharpness parameter for better approximation
    log_sum = np.log(np.sum(np.exp(-p * areas)))
    smooth_min = -log_sum / p
    return -smooth_min  # Maximize smooth minimum


def objective_weighted_areas(pts_flat):
    """Alternative objective: weighted sum of smallest triangles."""
    areas = triangle_areas_vectorized(pts_flat)
    sorted_areas = np.sort(areas)
    # Weight smallest 20 triangles more heavily
    weights = np.exp(-np.arange(len(sorted_areas)) / 20)
    return -np.sum(sorted_areas * weights) / np.sum(weights)


def hill_climb_optimize(points: np.ndarray, iterations: int = 300, initial_step: float = 0.028) -> np.ndarray:
    """Hill climbing optimization with expanded directional moves and adaptive step."""
    n = points.shape[0]
    current_min = compute_min_area(points)
    step_size = initial_step
    best_points = points.copy()
    
    # Enhanced direction set with more variations
    directions = [
        (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size),
        (step_size*0.7, step_size*0.7), (-step_size*0.7, step_size*0.7),
        (step_size*0.7, -step_size*0.7), (-step_size*0.7, -step_size*0.7),
        (step_size*0.5, 0), (-step_size*0.5, 0), (0, step_size*0.5), (0, -step_size*0.5),
        (step_size*0.35, step_size*0.35), (-step_size*0.35, step_size*0.35),
        (step_size*0.35, -step_size*0.35), (-step_size*0.35, -step_size*0.35),
        (step_size*0.85, step_size*0.25), (step_size*0.25, step_size*0.85),
        (-step_size*0.85, step_size*0.25), (-step_size*0.25, step_size*0.85),
        (step_size*0.85, -step_size*0.25), (step_size*0.25, -step_size*0.85),
        (-step_size*0.85, -step_size*0.25), (-step_size*0.25, -step_size*0.85),
    ]
    
    for iteration in range(iterations):
        improved = False
        for i in range(n):
            for base_dx, base_dy in directions:
                dx = base_dx * (step_size / initial_step)
                dy = base_dy * (step_size / initial_step)
                test_points = best_points.copy()
                test_points[i, 0] += dx
                test_points[i, 1] += dy
                test_points = project_to_unit_square(test_points)
                
                new_min = compute_min_area(test_points)
                if new_min > current_min:
                    current_min = new_min
                    best_points = test_points
                    improved = True
        
        if not improved:
            step_size *= 0.82
            if step_size < 3e-5:
                break
    
    return best_points


def lbfgs_optimize(points: np.ndarray, use_alternative_objective: bool = False) -> np.ndarray:
    """L-BFGS-B refinement with smooth LogSumExp objective and tighter tolerances."""
    n = points.shape[0]
    x0 = points.flatten()
    bounds = [(0.003, 0.997)] * (2 * n)
    
    # Choose objective
    obj = objective_weighted_areas if use_alternative_objective else objective_lse
    
    res = minimize(
        obj,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 550, 'ftol': 1e-12, 'gtol': 1e-10}
    )
    optimized_points = res.x.reshape(-1, 2)
    return project_to_unit_square(optimized_points)


def normalize_by_hull(points: np.ndarray) -> np.ndarray:
    """Normalize points based on convex hull area for consistent scaling."""
    hull = ConvexHull(points)
    if hull.volume > 0:
        points = (points - np.mean(points, axis=0)) / np.sqrt(hull.volume)
        points = points - points.min(axis=0) + 0.01
        points = np.clip(points, 0.005, 0.995)
    return points


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Generate multiple candidate initializations - expanded hybrid multi-start strategy
    candidates = []
    
    # Candidate 1: Enhanced hybrid boundary + interior pattern (from top-performing configurations)
    points1 = []
    points1.append([0.0, 0.0])
    points1.append([1.0, 0.0])
    points1.append([1.0, 1.0])
    points1.append([0.0, 1.0])
    points1.append([0.5, 0.02])
    points1.append([0.98, 0.5])
    points1.append([0.5, 0.98])
    interior1 = [
        [0.25, 0.25], [0.75, 0.25], [0.5, 0.5],
        [0.25, 0.75], [0.75, 0.75], [0.5, 0.35],
        [0.35, 0.65], [0.65, 0.65], [0.65, 0.35],
    ]
    points1.extend(interior1[:n - len(points1)])
    points1 = np.array(points1, dtype=np.float64)
    points1 += rng.normal(0, 0.003, size=points1.shape)
    candidates.append(project_to_unit_square(points1))
    
    # Candidate 2: Optimized circular boundary + inner pattern
    angles = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    boundary = np.column_stack([0.5 * np.cos(angles), 0.5 * np.sin(angles)]) + 0.5
    inner = np.array([
        [0.25, 0.25], [0.5, 0.25], [0.75, 0.25],
        [0.35, 0.55], [0.65, 0.55], [0.5, 0.8]
    ])
    points2 = np.vstack([boundary, inner])[:13]
    points2 += rng.normal(0, 0.003, size=points2.shape)
    candidates.append(project_to_unit_square(points2))
    
    # Candidate 3: Hexagonal lattice initialization with different seeds
    candidates.append(generate_hexagonal_lattice(n, seed=42))
    candidates.append(generate_hexagonal_lattice(n, seed=12345))
    
    # Candidate 4: Enhanced 3-fold symmetric configuration (from Exemplar 2 improvements)
    angles3 = np.linspace(0, 2 * np.pi, 9, endpoint=False)
    outer_r, inner_r = 0.48, 0.25
    boundary3 = np.column_stack([outer_r * np.cos(angles3), outer_r * np.sin(angles3)]) + 0.5
    inner_angles3 = np.linspace(0, 2 * np.pi, 4, endpoint=False) + np.pi/4
    inner3 = np.column_stack([inner_r * np.cos(inner_angles3), inner_r * np.sin(inner_angles3)]) + 0.5
    center3 = np.array([[0.5, 0.5]])
    points4 = np.vstack([boundary3, inner3, center3])[:13]
    points4 += rng.normal(0, 0.004, size=points4.shape)
    candidates.append(project_to_unit_square(points4))
    
    # Candidate 5: Hexagonal boundary + multiple inner points pattern
    angles_hex = np.linspace(0, 2 * np.pi, 8, endpoint=False)
    outer_hex = np.column_stack([0.48 * np.cos(angles_hex), 0.48 * np.sin(angles_hex)]) + 0.5
    inner_hex = np.array([
        [0.5, 0.5], [0.35, 0.35], [0.65, 0.35],
        [0.35, 0.65], [0.65, 0.65]
    ])
    points5 = np.vstack([outer_hex, inner_hex])[:13]
    points5 += rng.normal(0, 0.003, size=points5.shape)
    candidates.append(project_to_unit_square(points5))
    
    # Candidate 6: Regular grid with alternating offset pattern - enhanced
    points6 = []
    for i in range(4):
        for j in range(4):
            x = (i + 0.5) / 4.0
            y = (j + 0.5) / 4.0
            if (i + j) % 3 == 0:
                x += 0.02
            elif (i + j) % 4 == 0:
                y -= 0.015
            points6.append([x, y])
    points6 = np.array(points6)[:13]
    points6 += rng.normal(0, 0.003, size=points6.shape)
    candidates.append(project_to_unit_square(points6))
    
    # Candidate 7: 10-3 split boundary/interior configuration
    angles7 = np.linspace(0, 2 * np.pi, 10, endpoint=False)
    outer7 = np.column_stack([0.49 * np.cos(angles7), 0.49 * np.sin(angles7)]) + 0.5
    inner_r7 = 0.18
    inner7_angles = np.linspace(0, 2 * np.pi, 3, endpoint=False)
    inner7 = np.column_stack([inner_r7 * np.cos(inner7_angles), inner_r7 * np.sin(inner7_angles)]) + 0.5
    points7 = np.vstack([outer7[:10], inner7])[:13]
    points7 += rng.normal(0, 0.003, size=points7.shape)
    candidates.append(project_to_unit_square(points7))
    
    # Optimize all candidates with enhanced comprehensive pipeline
    best_score = -1
    best_points = None
    
    for cand in candidates:
        # Enhanced hill climbing with more iterations
        cand_hill = hill_climb_optimize(cand, iterations=300)
        
        # Dual L-BFGS refinement with both objectives
        cand_opt1 = lbfgs_optimize(cand_hill, use_alternative_objective=False)
        cand_opt2 = lbfgs_optimize(cand_hill, use_alternative_objective=True)
        
        # Evaluate both to pick better intermediate
        min1, min2 = compute_min_area(cand_opt1), compute_min_area(cand_opt2)
        cand_opt = cand_opt1 if min1 >= min2 else cand_opt2
        
        # Normalize by hull for consistent comparison
        cand_norm = normalize_by_hull(cand_opt)
        
        # Two-stage post-optimization for maximum refinement
        cand_post1 = hill_climb_optimize(cand_norm, iterations=120, initial_step=0.015)
        cand_final = hill_climb_optimize(cand_post1, iterations=80, initial_step=0.008)
        
        # Final evaluation with full metric
        current_min = compute_min_area(cand_final)
        try:
            hull = ConvexHull(cand_final)
            score = current_min / hull.volume if hull.volume > 0 else 0
        except:
            score = 0
        
        if score > best_score:
            best_score = score
            best_points = cand_final
    
    # Final verification checks
    assert np.all((best_points >= 0) & (best_points <= 1)), "Points out of bounds"
    assert best_points.shape == (13, 2), "Wrong point count"
    
    return best_points