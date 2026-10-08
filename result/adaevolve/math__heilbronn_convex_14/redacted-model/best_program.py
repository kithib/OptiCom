import numpy as np
from itertools import combinations


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    
    # Start with deterministic low-discrepancy Halton sequence (better than random)
    def halton(index, base):
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i = i // base
            f /= base
        return result
    
    # Generate initial Halton points
    points = np.array([[halton(i+1, 2), halton(i+1, 3)] for i in range(n)])
    
    # Normalize to unit square [0,1]^2 (convex hull area = 1.0)
    # Simple local optimization to improve minimum triangle area
    def triangle_area(a, b, c):
        return 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
    
    def min_area(pts):
        min_a = float('inf')
        for i, j, k in combinations(range(n), 3):
            a = triangle_area(pts[i], pts[j], pts[k])
            if a < min_a:
                min_a = a
        return min_a
    
    # Deterministic coordinate ascent optimization
    current_min = min_area(points)
    step = 0.05
    
    for _ in range(20):  # Limited iterations for efficiency
        improved = False
        for i in range(n):
            for dim in [0, 1]:
                for delta in [-step, step]:
                    new_points = points.copy()
                    new_points[i, dim] = np.clip(new_points[i, dim] + delta, 0.0, 1.0)
                    new_min = min_area(new_points)
                    if new_min > current_min:
                        points = new_points
                        current_min = new_min
                        improved = True
        if not improved:
            step *= 0.5
    
    return points