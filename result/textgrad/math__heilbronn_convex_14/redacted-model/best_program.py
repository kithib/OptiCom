import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    
    # Enhanced asymmetric initialization with further improved spacing and irregularity
    # 7 boundary points with optimized perturbations to minimize collinearities and expand hull
    theta = np.linspace(0, 2 * np.pi, 8)[:-1]
    theta += np.array([0.0, 0.0265, -0.0185, 0.0225, -0.0255, 0.0165, -0.0215])
    boundary = np.column_stack([0.492 * np.cos(theta) + 0.5, 0.492 * np.sin(theta) + 0.5])
    # Interior: Enhanced ring+center with varying asymmetric radii and refined phase/perturbations
    interior_theta = np.linspace(0, 2 * np.pi, 7)[:-1] + np.pi/6.7
    interior_theta += np.array([0.0085, -0.0075, 0.0065, -0.0085, 0.0075, -0.0065])
    interior_radii = np.array([0.254, 0.247, 0.253, 0.246, 0.252, 0.245])
    interior_ring = np.column_stack([interior_radii * np.cos(interior_theta) + 0.5, 
                                      interior_radii * np.sin(interior_theta) + 0.5])
    interior_center = np.array([[0.5007, 0.4993]])
    interior = np.vstack([interior_ring, interior_center])
    points = np.vstack([boundary, interior])
    
    # Precompute combination indices once
    _combo_cache = np.array(list(combinations(range(n), 3)))
    
    def min_triangle_area(pts):
        cross = np.cross(pts[_combo_cache[:, 1]] - pts[_combo_cache[:, 0]], 
                        pts[_combo_cache[:, 2]] - pts[_combo_cache[:, 0]])
        areas = np.abs(cross) / 2.0
        return np.min(areas)
    
    def compute_convex_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume if hull.volume > 0 else 1.0
        except:
            return 1.0
    
    # Initialize with normalized score for proper optimization
    best_min = min_triangle_area(points)
    current_hull_area = compute_convex_hull_area(points)
    best_normalized = best_min / current_hull_area
    best_points = points.copy()
    step = 0.076
    max_step_reductions = 18
    step_reductions = 0
    
    # Extended search iterations with improved seed
    rng = np.random.default_rng(seed=12791)
    for iteration in range(1200):
        improved = False
        for idx in rng.permutation(n):
            # Enhanced delta set with comprehensive coverage at multiple scales
            base_deltas = [(dx, dy) for dx in [-step, 0, step] for dy in [-step, 0, step] 
                          if not (dx == 0 and dy == 0)]
            deltas = base_deltas.copy()
            # Half-step deltas (full set)
            deltas += [(dx*0.5, dy*0.5) for dx, dy in base_deltas]
            # Quarter-step deltas (full set)
            deltas += [(dx*0.25, dy*0.25) for dx, dy in base_deltas]
            # Eighth-step deltas with extended oblique angles for hull expansion
            if step > 0.008:
                deltas += [(0.125*s, 0.125*t) for s, t in [(1,0), (-1,0), (0,1), (0,-1), 
                           (1,1), (-1,1), (1,-1), (-1,-1), (2,1), (-2,1), (1,2), (-1,2),
                           (3,1), (1,3), (-3,1), (3,-1), (-1,3), (1,-3), (2,3), (3,2), (-3,2), (3,-2), (-2,3), (2,-3)]]
            # Sixteenth-step for fine tuning
            if step > 0.015:
                deltas += [(0.0625*s, 0.0625*t) for s, t in [(1,0), (-1,0), (0,1), (0,-1),
                           (1,1), (-1,1), (1,-1), (-1,-1), (2,1), (1,2), (-2,1), (1,-2), (-1,2), (2,-1)]]
            # Thirty-second step for ultra-fine tuning early on
            if step > 0.025:
                deltas += [(0.03125*s, 0.03125*t) for s, t in [(1,0), (-1,0), (0,1), (0,-1),
                           (1,1), (-1,1), (1,-1), (-1,-1), (1,2), (2,1)]]
            
            orig_x, orig_y = float(points[idx, 0]), float(points[idx, 1])
            for dx, dy in deltas:
                new_x = np.clip(points[idx, 0] + dx, 0.004, 0.996)
                new_y = np.clip(points[idx, 1] + dy, 0.004, 0.996)
                points[idx, 0], points[idx, 1] = new_x, new_y
                current_min = min_triangle_area(points)
                current_hull = compute_convex_hull_area(points)
                current_normalized = current_min / current_hull
                # Explicit optimization for normalized score with proper hull awareness
                # Accept improvements in normalized score or equal score with expanded hull
                if (current_normalized > best_normalized + 1e-14 or
                    (abs(current_normalized - best_normalized) < 1e-12 and 
                     current_hull > current_hull_area * 1.00004)):
                    best_min = current_min
                    best_normalized = current_normalized
                    current_hull_area = current_hull
                    best_points = points.copy()
                    improved = True
                    orig_x, orig_y = float(new_x), float(new_y)
                else:
                    points[idx, 0], points[idx, 1] = orig_x, orig_y
        if not improved:
            points = best_points.copy()
            step *= 0.5035  # Balanced step reduction for finer convergence
            step_reductions += 1
            if step_reductions >= max_step_reductions:
                break
    
    return best_points