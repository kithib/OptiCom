# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    
    # Precompute all triangle combinations once (286 total) for efficiency
    all_combos = np.array(list(combinations(range(n), 3)))
    
    # Symmetry-aware initialization: staggered boundary + interior with 3-fold symmetry
    rng = np.random.default_rng(seed=42)
    
    boundary_count = 9
    interior_count = 4
    
    # Regular nonagon on boundary - rotated for optimal symmetry
    angles = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False) + np.pi/9
    boundary = 0.49 * np.column_stack([np.cos(angles), np.sin(angles)]) + 0.5
    
    # 3-fold symmetric interior with phase shift to stagger relative to boundary (60° offset)
    interior = []
    for k in range(3):
        angle = 2 * np.pi * k / 3 + np.pi / 6
        r = 0.22
        interior.append([0.5 + r * np.cos(angle), 0.5 + r * np.sin(angle)])
    interior.append([0.5, 0.5])
    interior = np.array(interior[:interior_count])
    
    # Combine and add controlled noise
    points = np.vstack([boundary, interior])
    points += rng.normal(0, 0.006, points.shape)
    points = np.clip(points, 0.01, 0.99)
    
    # Efficient convex hull area computation for normalization - use correct attribute volume for 2D area
    def convex_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.volume if hasattr(hull, 'volume') else 1.0
        except:
            return 1.0
    
    # Vectorized triangle area computation
    def compute_min_area(pts):
        i, j, k = all_combos[:, 0], all_combos[:, 1], all_combos[:, 2]
        cross = (pts[j, 0] - pts[i, 0]) * (pts[k, 1] - pts[i, 1]) - \
                (pts[j, 1] - pts[i, 1]) * (pts[k, 0] - pts[i, 0])
        areas = 0.5 * np.abs(cross)
        return np.min(areas), areas
    
    step = 0.008
    for iteration in range(350):
        current_min, areas = compute_min_area(points)
        hull_area = convex_hull_area(points)
        grad = np.zeros_like(points)
        
        # Adaptive focusing: threshold decreases from 2.4 to 1.05 - wider range for better coverage
        focus_factor = 2.4 - (iteration / 350.0) * 1.35
        
        threshold = current_min * focus_factor
        mask = areas < threshold
        
        if np.any(mask):
            filtered_combos = all_combos[mask]
            filtered_areas = areas[mask]
            
            # Adaptive weight exponent: increases from 1.3 to 2.7, plus small hull normalization
            weight_exp = 1.3 + (iteration / 350.0) * 1.4
            weights = 1.0 / (filtered_areas**weight_exp + 1e-10)
            weights = weights * (hull_area ** 0.25)
            
            i = filtered_combos[:, 0]
            j = filtered_combos[:, 1]
            k = filtered_combos[:, 2]
            
            vi, vj, vk = points[i], points[j], points[k]
            
            grad_i = weights[:, None] * np.column_stack([vk[:, 1] - vj[:, 1], vj[:, 0] - vk[:, 0]]) * 0.5
            grad_j = weights[:, None] * np.column_stack([vi[:, 1] - vk[:, 1], vk[:, 0] - vi[:, 0]]) * 0.5
            grad_k = weights[:, None] * np.column_stack([vj[:, 1] - vi[:, 1], vi[:, 0] - vj[:, 0]]) * 0.5
            
            np.add.at(grad, i, grad_i)
            np.add.at(grad, j, grad_j)
            np.add.at(grad, k, grad_k)
        
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-7:
            break
        grad = grad / (grad_norm + 1e-8)
        points += step * grad
        points = np.clip(points, 0.005, 0.995)
        if iteration % 70 == 69:
            step *= 0.66
    
    return points


# EVOLVE-BLOCK-END