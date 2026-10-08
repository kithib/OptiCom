import numpy as np
from scipy.optimize import minimize


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    np.random.seed(42)
    n = 11
    sqrt3 = np.sqrt(3)
    sqrt3_over_6 = sqrt3 / 6.0
    sqrt3_over_4 = sqrt3 / 4.0
    sqrt3_over_3 = sqrt3 / 3.0
    sqrt3_over_2 = sqrt3 / 2.0
    
    # Candidate 1: Hexagonal grid-based initialization (strong symmetry)
    grid_init = np.array([
        # Vertices - important for maximizing spread
        [0.0, 0.0], [1.0, 0.0], [0.5, sqrt3_over_2],
        # Edge midpoints
        [0.5, 0.0], [0.25, sqrt3_over_4], [0.75, sqrt3_over_4],
        # Center
        [0.5, sqrt3_over_6],
        # Additional structured grid points on base edge
        [1/3, 0.0], [2/3, 0.0],
        # Mid-height inner point
        [0.5, sqrt3_over_3],
        # Extra optimized inner point (refined)
        [0.5, 0.28867513459]
    ], dtype=np.float64)
    
    # Candidate 2: Optimized geometric configuration from previous runs
    geom_init = np.array([
        # 3 corners
        [0.0, 0.0], [1.0, 0.0], [0.5, sqrt3_over_2],
        # Edge midpoints
        [0.5, 0.0], [0.25, sqrt3_over_4], [0.75, sqrt3_over_4],
        # Center
        [0.5, sqrt3_over_6],
        # Optimized inner points from best known configurations
        [0.17157287525, 0.09999999999],
        [0.82842712475, 0.09999999999],
        [0.33000000000, 0.50000000000],
        [0.67000000000, 0.50000000000]
    ], dtype=np.float64)
    
    # Candidate 3: Symmetrically-optimized 11-point known configuration
    sym_opt_init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, sqrt3_over_2],
        [0.1206147425, 0.0],
        [0.8793852575, 0.0],
        [0.2793852575, 0.4839595655],
        [0.7206147425, 0.4839595655],
        [0.06030737125, 0.1044915522],
        [0.93969262875, 0.1044915522],
        [0.3511334025, 0.2886751346],
        [0.6488665975, 0.2886751346]
    ], dtype=np.float64)
    
    # Halton sequence generator for low-discrepancy initializations
    def halton_sequence(num_points, base1=2, base2=3):
        points = []
        for i in range(num_points):
            x, y = 0.0, 0.0
            f1, f2 = 1.0 / base1, 1.0 / base2
            idx = i
            while idx > 0:
                x += (idx % base1) * f1
                idx //= base1
                f1 /= base1
            idx = i
            while idx > 0:
                y += (idx % base2) * f2
                idx //= base2
                f2 /= base2
            points.append([x, y])
        return np.array(points)
    
    # Candidate 4: Halton sequence initialization (base1=2, base2=3)
    unit_samples = halton_sequence(n)
    halton_init = np.zeros((n, 2))
    for i in range(n):
        s = np.sqrt(unit_samples[i, 0])
        t = unit_samples[i, 1]
        x = s * (1 - t) + 0.5 * s * t
        y = sqrt3_over_2 * s * t
        halton_init[i] = [x, y]
    
    # Candidate 5: Diversified alternate Halton sequence (base1=3, base2=5)
    unit_samples_alt = halton_sequence(n, base1=3, base2=5)
    halton_alt_init = np.zeros((n, 2))
    for i in range(n):
        s = np.sqrt(unit_samples_alt[i, 0])
        t = unit_samples_alt[i, 1]
        x = s * (1 - t) + 0.5 * s * t
        y = sqrt3_over_2 * s * t
        halton_alt_init[i] = [x, y]
    
    # Helper to compute min triangle area for initial candidate evaluation
    def compute_min_area(pts):
        min_area = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
                    area = 0.5 * cross
                    if area < min_area:
                        min_area = area
        return min_area
    
    # Evaluate ALL candidate configurations and select the BEST
    candidates = [grid_init, geom_init, sym_opt_init, halton_init, halton_alt_init]
    candidate_scores = [compute_min_area(c) for c in candidates]
    best_init_idx = np.argmax(candidate_scores)
    init_points = candidates[best_init_idx].copy()
    init_flat = init_points.flatten()
    
    # Vectorized constraints: Efficiently ensure points stay within equilateral triangle
    def in_equilateral_constraint(xy):
        xy = xy.reshape(-1, 2)
        x, y = xy[:, 0], xy[:, 1]
        return np.min(np.array([
            y + 1e-15,                      # Above base edge: y >= 0 with numerical buffer
            sqrt3 * x - y,                   # Below left edge: y <= sqrt(3)x
            sqrt3 * (1 - x) - y              # Below right edge: y <= sqrt(3)(1-x)
        ]), axis=0)
    
    constraints = [{'type': 'ineq', 'fun': in_equilateral_constraint}]
    
    # Objective function: Minimize NEGATIVE of min area to maximize min triangle area
    def objective(p_flat):
        pts = p_flat.reshape((n, 2))
        min_area = np.inf
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    v1 = pts[j] - pts[i]
                    v2 = pts[k] - pts[i]
                    cross = np.abs(v1[0] * v2[1] - v1[1] * v2[0])
                    area = 0.5 * cross
                    if area < min_area:
                        min_area = area
        return -min_area
    
    # Multi-stage robust optimization with varied methods to avoid local minima
    result = None
    optimization_methods = [
        {'method': 'SLSQP', 'maxiter': 3000, 'ftol': 1e-12},
        {'method': 'SLSQP', 'maxiter': 4000, 'ftol': 1e-10},
        {'method': 'COBYLA', 'maxiter': 5000, 'tol': 1e-10},
        {'method': 'COBYLA', 'maxiter': 6000, 'tol': 1e-11},
    ]
    
    for opt_config in optimization_methods:
        try:
            current_init = init_flat if result is None or not result.success else result.x
            result = minimize(
                objective,
                current_init,
                method=opt_config['method'],
                constraints=constraints,
                options={
                    'maxiter': opt_config['maxiter'],
                    'ftol': opt_config.get('ftol', opt_config.get('tol', 1e-10))
                }
            )
            if result.success:
                break
        except Exception:
            continue
    
    # Barycentric feasibility check for points
    def point_inside(p):
        A = np.array([0.0, 0.0])
        B = np.array([1.0, 0.0])
        C = np.array([0.5, sqrt3_over_2])
        v0, v1, v2 = C - A, B - A, p - A
        dot00, dot01, dot02 = np.dot(v0, v0), np.dot(v0, v1), np.dot(v0, v2)
        dot11, dot12 = np.dot(v1, v1), np.dot(v1, v2)
        inv_denom = 1.0 / (dot00 * dot11 - dot01 * dot01)
        u = (dot11 * dot02 - dot01 * dot12) * inv_denom
        v = (dot00 * dot12 - dot01 * dot02) * inv_denom
        return (u >= -1e-12) and (v >= -1e-12) and (u + v <= 1.0 + 1e-12)
    
    # Process result with robust fallback and numerical cleanup
    if result is not None and result.success:
        points = result.x.reshape((n, 2))
        # Numerical clipping for basic bounds
        points[:, 0] = np.clip(points[:, 0], 0.0, 1.0)
        points[:, 1] = np.clip(points[:, 1], 0.0, sqrt3_over_2)
        # Per-point feasibility correction with fallback to initial
        for i in range(n):
            if not point_inside(points[i]):
                points[i] = init_points[i]
    else:
        # Ultimate fallback: best evaluated initial configuration
        points = init_points
    
    return points