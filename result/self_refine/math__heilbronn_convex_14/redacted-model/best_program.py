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
    
    # Low-discrepancy Halton sequence for good initial distribution
    def halton(index, base):
        result = 0.0
        factor = 1.0 / base
        i = index
        while i > 0:
            result += factor * (i % base)
            i = i // base
            factor /= base
        return result
    
    # Generate deterministic Halton sequence - starting from index 5 for better dispersion (known improvement for Heilbronn)
    points = np.array([[halton(i + 5, 2), halton(i + 5, 3)] for i in range(n)])
    
    # Precompute all combinations once - major speed optimization
    combos = np.array(list(combinations(range(n), 3)))
    
    # Vectorized triangle area computation for speed
    def min_area_vectorized(pts):
        a = pts[combos[:, 0]]
        b = pts[combos[:, 1]]
        c = pts[combos[:, 2]]
        # Cross product for area
        cross = np.abs((b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - 
                       (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0]))
        return 0.5 * np.min(cross)
    
    # Compute convex hull area for proper normalization awareness
    def convex_hull_area(pts):
        try:
            hull = ConvexHull(pts)
            return hull.area if hull.area > 0 else 1.0
        except:
            return 1.0
    
    current_min = min_area_vectorized(points)
    hull_area = convex_hull_area(points)
    
    # Helper move set generator
    def get_moves(step):
        return [(-step, -step), (-step, 0), (-step, step),
                (0, -step), (0, step),
                (step, -step), (step, 0), (step, step)]
    
    # Phase 1: Expand configuration with weighted improvement (considering hull area)
    # This helps escape local minima while maintaining a good convex hull area
    for step in [0.07, 0.05, 0.04]:
        for _ in range(35):
            improved = False
            for i in range(n):
                best_improvement = 0.0
                best_move = None
                for dx, dy in get_moves(step):
                    new_points = points.copy()
                    new_points[i] = np.clip(new_points[i] + [dx, dy], 0.0, 1.0)
                    new_min = min_area_vectorized(new_points)
                    new_hull = convex_hull_area(new_points)
                    # Weight by hull area to maximize normalized score
                    normalized_improvement = (new_min / new_hull) - (current_min / hull_area)
                    if normalized_improvement > best_improvement:
                        best_improvement = normalized_improvement
                        best_move = (new_min, new_points, new_hull)
                if best_move is not None:
                    points = best_move[1]
                    current_min = best_move[0]
                    hull_area = best_move[2]
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 2: Medium steps for refinement
    for step in [0.03, 0.022, 0.016]:
        for _ in range(45):
            improved = False
            for i in range(n):
                best_improvement = 0.0
                best_move = None
                for dx, dy in get_moves(step):
                    new_points = points.copy()
                    new_points[i] = np.clip(new_points[i] + [dx, dy], 0.0, 1.0)
                    new_min = min_area_vectorized(new_points)
                    new_hull = convex_hull_area(new_points)
                    normalized_improvement = (new_min / new_hull) - (current_min / hull_area)
                    if normalized_improvement > best_improvement:
                        best_improvement = normalized_improvement
                        best_move = (new_min, new_points, new_hull)
                if best_move is not None:
                    points = best_move[1]
                    current_min = best_move[0]
                    hull_area = best_move[2]
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 3: Fine-grained optimization
    for step in [0.012, 0.01, 0.008, 0.006, 0.004]:
        for _ in range(65):
            improved = False
            for i in range(n):
                best_improvement = 0.0
                best_move = None
                for dx, dy in get_moves(step):
                    new_points = points.copy()
                    new_points[i] = np.clip(new_points[i] + [dx, dy], 0.0, 1.0)
                    new_min = min_area_vectorized(new_points)
                    new_hull = convex_hull_area(new_points)
                    normalized_improvement = (new_min / new_hull) - (current_min / hull_area)
                    if normalized_improvement > best_improvement:
                        best_improvement = normalized_improvement
                        best_move = (new_min, new_points, new_hull)
                if best_move is not None:
                    points = best_move[1]
                    current_min = best_move[0]
                    hull_area = best_move[2]
                    improved = True
                    break
            if not improved:
                break
    
    # Phase 4: Ultra-fine optimization to polish local minimum
    for step in [0.003, 0.0025, 0.002, 0.001]:
        for _ in range(90):
            improved = False
            for i in range(n):
                best_improvement = 0.0
                best_move = None
                for dx, dy in get_moves(step):
                    new_points = points.copy()
                    new_points[i] = np.clip(new_points[i] + [dx, dy], 0.0, 1.0)
                    new_min = min_area_vectorized(new_points)
                    new_hull = convex_hull_area(new_points)
                    normalized_improvement = (new_min / new_hull) - (current_min / hull_area)
                    if normalized_improvement > best_improvement:
                        best_improvement = normalized_improvement
                        best_move = (new_min, new_points, new_hull)
                if best_move is not None:
                    points = best_move[1]
                    current_min = best_move[0]
                    hull_area = best_move[2]
                    improved = True
                    break
            if not improved:
                break
    
    return points