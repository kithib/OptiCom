# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)

    # Improved initialization: 3-fold symmetric boundary + interior points
    angles = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    boundary_points = np.column_stack([np.cos(angles), np.sin(angles)]) * 0.48
    center_point = np.array([[0.5, 0.5]])
    points = np.vstack([center_point, boundary_points + 0.5])

    # Helper to compute triangle areas
    def compute_min_area(pts):
        triangles = np.array(list(combinations(pts, 3)))
        areas = 0.5 * np.abs(np.cross(triangles[:, 1] - triangles[:, 0],
                                       triangles[:, 2] - triangles[:, 0]))
        return np.min(areas), areas

    # Optimization phase 1: Expand smallest triangles directly
    learning_rate = 0.08
    iterations = 150

    for i in range(iterations):
        min_area, areas = compute_min_area(points)

        smallest_tri_idx = np.argmin(areas)
        tri_combs = list(combinations(range(n), 3))
        i0, i1, i2 = tri_combs[smallest_tri_idx]

        centroid = np.mean(points[[i0, i1, i2]], axis=0)
        for idx in [i0, i1, i2]:
            direction = points[idx] - centroid
            norm = np.linalg.norm(direction)
            if norm > 1e-12:
                points[idx] += learning_rate * direction / norm

        points = np.clip(points, 0.01, 0.99)

        if (i + 1) % 40 == 0:
            learning_rate *= 0.6

    # Add small perturbation to escape local optima
    points += rng.normal(0, 0.008, size=points.shape)
    
    # Ensure all points stay within [0, 1] bounds
    points = np.clip(points, 0.01, 0.99)
    
    # Normalize to unit convex hull area
    hull = ConvexHull(points)
    scale_factor = np.sqrt(1.0 / hull.area)
    points = (points - np.mean(points, axis=0)) * scale_factor + 0.5
    
    return points


# EVOLVE-BLOCK-END