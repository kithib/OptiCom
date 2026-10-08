# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize points to have convex hull area = 1 using affine transformation."""
    # Compute convex hull
    from scipy.spatial import ConvexHull
    hull = ConvexHull(points)
    hull_area = hull.volume
    if hull_area > 0:
        # Scale all points to make convex hull area = 1
        points = points / np.sqrt(hull_area)
    return points


def _min_triangle_area(points: np.ndarray) -> float:
    """Compute the area of the smallest triangle formed by any 3 points."""
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        # Cross product gives twice the area
        cross = np.cross(points[j] - points[i], points[k] - points[i])
        area = abs(cross) / 2.0
        if area < min_area:
            min_area = area
    return min_area


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    rng = np.random.default_rng(seed=12345)
    
    # Start with a quasi-random low-discrepancy sequence (better than pure random)
    # Generate points in [0,1] x [0,1]
    points = np.zeros((n, 2))
    for i in range(n):
        # Use additive recurrence for low-discrepancy
        points[i, 0] = (i * 0.7548776662466927) % 1.0
        points[i, 1] = (i * 0.5698402909980533) % 1.0
    
    # Simple hill-climbing optimization to improve min triangle area
    max_iterations = 500
    step_size = 0.05
    current_min = _min_triangle_area(points)
    
    for iteration in range(max_iterations):
        for idx in range(n):
            original_point = points[idx].copy()
            # Try small random perturbations
            for _ in range(5):
                perturbation = rng.uniform(-step_size, step_size, 2)
                points[idx] = np.clip(original_point + perturbation, 0.0, 1.0)
                new_min = _min_triangle_area(points)
                if new_min > current_min:
                    current_min = new_min
                    break
            else:
                points[idx] = original_point
        
        # Decay step size
        if (iteration + 1) % 100 == 0:
            step_size *= 0.5
    
    # Normalize to unit convex hull area
    points = _normalize_points(points)
    
    return points


# EVOLVE-BLOCK-END