import numpy as np
from itertools import combinations


def _compute_min_triangle_area(points: np.ndarray) -> float:
    """Compute the area of the smallest triangle formed by any three points."""
    min_area = float("inf")
    for tri in combinations(points, 3):
        a, b, c = tri
        area = 0.5 * abs(
            a[0] * (b[1] - c[1]) + b[0] * (c[1] - a[1]) + c[0] * (a[1] - b[1])
        )
        if area < min_area:
            min_area = area
            if min_area < 1e-15:
                return min_area
    return min_area


def _convex_hull(points: np.ndarray) -> np.ndarray:
    """Compute convex hull using Andrew's monotone chain algorithm."""
    points = points[np.lexsort(points.T)]
    lower = []
    for p in points:
        while len(lower) >= 2:
            a, b = lower[-2], lower[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 0:
                lower.pop()
            else:
                break
        lower.append(p)
    upper = []
    for p in reversed(points):
        while len(upper) >= 2:
            a, b = upper[-2], upper[-1]
            cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
            if cross <= 0:
                upper.pop()
            else:
                break
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def _convex_hull_area(hull: np.ndarray) -> float:
    """Compute area of a convex polygon using the shoelace formula."""
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i, 0] * hull[j, 1] - hull[j, 0] * hull[i, 1]
    return 0.5 * abs(area)


def _normalize_hull_to_unit_area(points: np.ndarray) -> np.ndarray:
    """Scale and translate points so convex hull has unit area and is centered in [0,1]^2."""
    hull = _convex_hull(points)
    hull_area = _convex_hull_area(hull)
    if hull_area <= 0:
        return points
    scale = np.sqrt(1.0 / hull_area)
    centroid = np.mean(points, axis=0)
    points = centroid + (points - centroid) * scale
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    span = max_coords - min_coords
    if span[0] > 0 and span[1] > 0:
        points = (points - min_coords) / np.maximum(span, 1e-8)
    return points


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    
    # Highly optimized starting configuration refined through extensive iterative optimization
    # Balances boundary and interior points for maximum minimum triangle area
    points = np.array([
        [0.052, 0.498],
        [0.128, 0.882],
        [0.158, 0.112],
        [0.292, 0.708],
        [0.308, 0.292],
        [0.425, 0.525],
        [0.488, 0.092],
        [0.512, 0.908],
        [0.622, 0.378],
        [0.688, 0.718],
        [0.768, 0.102],
        [0.862, 0.508],
        [0.912, 0.878],
        [0.958, 0.338]
    ], dtype=np.float64)
    
    # Apply normalization to ensure convex hull has ~unit area
    points = _normalize_hull_to_unit_area(points)
    
    # Refinement via hill climbing with balanced step schedule and reduced directions
    # Optimized for both quality and performance
    step_sizes = [0.04, 0.02, 0.01, 0.005, 0.0025]
    max_iters_per_step = 150
    
    hull = _convex_hull(points)
    hull_area = _convex_hull_area(hull)
    best_min_area = _compute_min_triangle_area(points)
    best_score = best_min_area / hull_area if hull_area > 0 else 0
    best_points = points.copy()
    
    # 24-direction search - balanced coverage and performance
    directions = []
    for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        directions.append((np.cos(angle), np.sin(angle)))
    
    # Multi-scale scale factors for each step - early coarse search, later finer search
    scale_factors_by_step = [[2.0, 1.0, 0.5], [1.5, 1.0, 0.5], [1.25, 1.0], [1.0, 0.75], [1.0]]
    
    for step_idx, step in enumerate(step_sizes):
        scale_factors = scale_factors_by_step[step_idx]
        for iter_count in range(max_iters_per_step):
            improved = False
            rng = np.random.default_rng(seed=42 + step_idx * 1000 + iter_count)
            point_indices = rng.permutation(n)
            
            for i in point_indices:
                for scale_factor in scale_factors:
                    for dx, dy in directions:
                        new_points = best_points.copy()
                        new_x = np.clip(new_points[i, 0] + dx * step * scale_factor, 0.005, 0.995)
                        new_y = np.clip(new_points[i, 1] + dy * step * scale_factor, 0.005, 0.995)
                        new_points[i] = [new_x, new_y]
                        
                        current_hull = _convex_hull(new_points)
                        current_hull_area = _convex_hull_area(current_hull)
                        if current_hull_area <= 1e-8:
                            continue
                        current_min_area = _compute_min_triangle_area(new_points)
                        current_score = current_min_area / current_hull_area
                        
                        if current_score > best_score + 1e-12:
                            best_points = new_points
                            best_score = current_score
                            best_min_area = current_min_area
                            hull_area = current_hull_area
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return best_points