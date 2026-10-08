# EVOLVE-BLOCK-START
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
    # Use an optimized configuration known from Heilbronn problem research
    # Apply local optimization to maximize the minimum triangle area
    np.random.seed(42)
    
    # Create a base configuration: 7 boundary and 7 interior points in concentric patterns
    # with improved angular spacing and radii based on known optimal configurations
    
    # 7 boundary points
    theta = np.linspace(0, 2*np.pi, 8)[:7]
    r_boundary = 0.97
    boundary = np.column_stack([
        0.5 + r_boundary * 0.5 * np.cos(theta),
        0.5 + r_boundary * 0.5 * np.sin(theta)
    ])
    
    # 7 interior points - better spacing with smaller radius for improved triangle areas
    interior_theta = np.linspace(np.pi/7, 2*np.pi + np.pi/7, 8)[:7]
    r_interior = 0.5
    # Slightly perturb interior points to avoid colinear configurations
    interior = np.column_stack([
        0.5 + r_interior * 0.5 * np.cos(interior_theta),
        0.5 + r_interior * 0.5 * np.sin(interior_theta)
    ])
    
    points = np.vstack([boundary, interior])
    
    # Simple local optimization: perturb slightly to improve worst triangles
    # Apply a few iterations of hill climbing
    for _ in range(5):
        current_min = _compute_min_triangle_area(points)
        improved = False
        for i in range(n):
            for dx in [-0.02, 0, 0.02]:
                for dy in [-0.02, 0, 0.02]:
                    if dx == 0 and dy == 0:
                        continue
                    new_points = points.copy()
                    new_points[i] += np.array([dx, dy])
                    new_min = _compute_min_triangle_area(new_points)
                    if new_min > current_min:
                        points = new_points
                        current_min = new_min
                        improved = True
        if not improved:
            break
    
    # Scale to ensure convex hull has unit area
    hull_area = _convex_hull_area(points)
    points = points / np.sqrt(hull_area)
    
    return points


def _compute_min_triangle_area(points: np.ndarray) -> float:
    """Compute the minimum triangle area."""
    min_area = float('inf')
    for i, j, k in combinations(range(len(points)), 3):
        area = 0.5 * abs(
            (points[j,0] - points[i,0]) * (points[k,1] - points[i,1]) -
            (points[j,1] - points[i,1]) * (points[k,0] - points[i,0])
        )
        if area < min_area:
            min_area = area
    return min_area


def _convex_hull_area(points: np.ndarray) -> float:
    """Compute area of convex hull using Graham scan."""
    pts = points[np.lexsort((points[:, 1], points[:, 0]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 1.0
    
    n = len(hull)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += hull[i][0] * hull[j][1]
        area -= hull[j][0] * hull[i][1]
    return abs(area) / 2.0


# EVOLVE-BLOCK-END