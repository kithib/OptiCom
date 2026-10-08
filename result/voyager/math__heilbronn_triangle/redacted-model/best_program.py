import numpy as np
from itertools import combinations
from numba import jit

@jit(nopython=True)
def min_triangle_area_numba(pts):
    n = len(pts)
    min_area = 1e10
    for i in range(n):
        a = pts[i]
        for j in range(i+1, n):
            b = pts[j]
            for k in range(j+1, n):
                c = pts[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
                if area < min_area:
                    min_area = area
    return min_area

def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    np.random.seed(42)
    sqrt3 = np.sqrt(3)
    
    # Initialize with vertices plus midpoints (better symmetry and distribution)
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, sqrt3/2]])
    midpoints = np.array([[0.5, 0.0], [0.25, sqrt3/4], [0.75, sqrt3/4]])
    
    # Create additional interior points using barycentric coordinates
    interior = []
    # Improved pattern for better initial distribution (5 interior = 3+3+5 = 11 total)
    pattern = [
        (0.2, 0.2), (0.6, 0.2), (0.2, 0.6),
        (0.4, 0.4), (0.333, 0.333),
    ]
    for r1, r2 in pattern:
        interior.append((1 - r1 - r2) * vertices[0] + r1 * vertices[1] + r2 * vertices[2])
    
    points = np.vstack([vertices, midpoints, np.array(interior)])
    points += np.random.normal(0, 0.005, points.shape)
    
    # Proper projection of points into the triangle - also ensure x stays in valid range
    def project_to_triangle(pts):
        for i, (x, y) in enumerate(pts):
            x = max(0.0, min(1.0, x))  # Ensure x is within [0, 1] first
            y = max(0, min(y, sqrt3 * x, sqrt3 * (1 - x)))
            pts[i] = [x, y]
        return pts
    
    points = project_to_triangle(points)
    
    # Hill climbing optimization with adaptive step size and restarts
    best_points = points.copy()
    best_min_area = min_triangle_area_numba(points)
    step_size = 0.05
    iterations = 12000000  # Further increased iterations for deeper search
    no_improve = 0
    
    for it in range(iterations):
        idx = np.random.randint(n)
        delta = np.random.uniform(-step_size, step_size, 2)
        new_points = points.copy()
        new_points[idx] += delta
        new_points = project_to_triangle(new_points)
        new_min_area = min_triangle_area_numba(new_points)
        
        if new_min_area > best_min_area:
            points = new_points
            best_points = new_points.copy()
            best_min_area = new_min_area
            step_size = min(step_size * 1.1, 0.18)  # More aggressive step growth
            no_improve = 0
        else:
            no_improve += 1
            step_size = max(step_size * 0.992, 0.0000001)
            # Accept occasional worse moves to escape local optima
            if np.random.random() < 0.005:  # Slightly higher acceptance rate
                points = new_points
            # Periodic restart from best found
            if no_improve > 6000:  # Longer patience before restart
                points = best_points + np.random.normal(0, 0.015, best_points.shape)
                points = project_to_triangle(points)
                no_improve = 0
                step_size = 0.05
    
    return best_points.astype(np.float64)