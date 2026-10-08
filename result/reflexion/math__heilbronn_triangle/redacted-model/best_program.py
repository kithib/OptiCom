import numpy as np
from itertools import combinations


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
    h = sqrt3 / 2
    
    vertices = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, h]])
    
    points = np.zeros((n, 2))
    points[0:3] = vertices
    
    # Optimized initial interior points for better symmetry
    interior_points = np.array([
        [0.105, 0.155],
        [0.895, 0.155],
        [0.500, 0.745],
        [0.205, 0.365],
        [0.795, 0.365],
        [0.365, 0.155],
        [0.635, 0.155],
        [0.500, 0.445]
    ])
    points[3:11] = interior_points
    
    def compute_min_area(pts):
        min_area = float('inf')
        for trio in combinations(range(len(pts)), 3):
            a, b, c = pts[trio[0]], pts[trio[1]], pts[trio[2]]
            area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
            if area < min_area:
                min_area = area
        return min_area
    
    def project_point(point):
        """Project point to be inside the triangle."""
        x, y = point
        x = np.clip(x, 0, 1)
        max_y = min(sqrt3 * x, -sqrt3 * (x - 1))
        y = np.clip(y, 0, max_y)
        return np.array([x, y])
    
    current_min_area = compute_min_area(points)
    best_points = points.copy()
    
    # Phase 1: Extended coordinate ascent with larger step range
    for iteration in range(120):
        improved = False
        for i in range(3, n):
            for dx in [-0.04, -0.025, -0.012, 0.0, 0.012, 0.025, 0.04]:
                for dy in [-0.04, -0.025, -0.012, 0.0, 0.012, 0.025, 0.04]:
                    new_point = project_point(points[i] + np.array([dx, dy]))
                    old_val = points[i].copy()
                    points[i] = new_point
                    new_min = compute_min_area(points)
                    if new_min > current_min_area:
                        current_min_area = new_min
                        best_points = points.copy()
                        improved = True
                    else:
                        points[i] = old_val
        if not improved:
            break
    
    # Phase 2: Extended random perturbations with decreasing step size
    for step_exp in range(12):
        step = 0.032 - 0.0022 * step_exp  # Step size decreases each iteration
        for _ in range(250):
            for i in range(3, n):
                original = points[i].copy()
                for _ in range(10):
                    dx, dy = np.random.uniform(-step, step, 2)
                    new_point = project_point(original + np.array([dx, dy]))
                    points[i] = new_point
                    new_min = compute_min_area(points)
                    if new_min >= current_min_area:
                        if new_min > current_min_area:
                            current_min_area = new_min
                            best_points = points.copy()
                    else:
                        points[i] = original
    
    # Phase 3: Fine-grained hill climbing with extended iterations
    for iteration in range(250):
        for i in range(3, n):
            for dx in [-0.007, -0.0035, 0.0, 0.0035, 0.007]:
                for dy in [-0.007, -0.0035, 0.0, 0.0035, 0.007]:
                    new_point = project_point(points[i] + np.array([dx, dy]))
                    old_val = points[i].copy()
                    points[i] = new_point
                    new_min_area = compute_min_area(points)
                    if new_min_area > current_min_area:
                        current_min_area = new_min_area
                        best_points = points.copy()
                    else:
                        points[i] = old_val
    
    # Phase 4: Extended relaxation of all points including vertices
    for iteration in range(120):
        for i in range(n):
            old_val = points[i].copy()
            dx, dy = np.random.uniform(-0.014, 0.014, 2)
            points[i] = project_point(points[i] + np.array([dx, dy]))
            new_min_area = compute_min_area(points)
            if new_min_area > current_min_area:
                current_min_area = new_min_area
                best_points = points.copy()
            else:
                points[i] = old_val
    
    # Phase 5: Super fine polishing with extended iterations
    for iteration in range(300):
        for i in range(3, n):
            for dx in [-0.0018, 0.0018, 0, -0.0009, 0.0009]:
                for dy in [-0.0018, 0.0018, 0, -0.0009, 0.0009]:
                    new_point = project_point(points[i] + np.array([dx, dy]))
                    old_val = points[i].copy()
                    points[i] = new_point
                    new_min_area = compute_min_area(points)
                    if new_min_area > current_min_area:
                        current_min_area = new_min_area
                        best_points = points.copy()
                    else:
                        points[i] = old_val
    
    # Phase 6: Additional ultra-fine polishing
    for iteration in range(100):
        for i in range(3, n):
            for dx in [-0.0005, 0, 0.0005]:
                for dy in [-0.0005, 0, 0.0005]:
                    new_point = project_point(points[i] + np.array([dx, dy]))
                    old_val = points[i].copy()
                    points[i] = new_point
                    new_min_area = compute_min_area(points)
                    if new_min_area > current_min_area:
                        current_min_area = new_min_area
                        best_points = points.copy()
                    else:
                        points[i] = old_val
    
    # Phase 7: Final vertex refinement
    for iteration in range(100):
        for i in range(3):
            old_val = points[i].copy()
            dx, dy = np.random.uniform(-0.004, 0.004, 2)
            points[i] = project_point(points[i] + np.array([dx, dy]))
            new_min_area = compute_min_area(points)
            if new_min_area > current_min_area:
                current_min_area = new_min_area
                best_points = points.copy()
            else:
                points[i] = old_val
    
    return best_points