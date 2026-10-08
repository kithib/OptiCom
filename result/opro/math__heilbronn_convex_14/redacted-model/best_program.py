# EVOLVE-BLOCK-START
import numpy as np


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    rng = np.random.default_rng(seed=42)
    
    def compute_min_area(points: np.ndarray) -> float:
        n = len(points)
        min_area = float('inf')
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    area = 0.5 * np.abs(
                        (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                    )
                    if area < min_area:
                        min_area = area
        return min_area
    
    # Start with perturbed hexagonal lattice (known good structure)
    sqrt_n = int(np.ceil(np.sqrt(n)))
    points = []
    for i in range(sqrt_n):
        for j in range(sqrt_n):
            if len(points) < n:
                x = (i + 0.5 * (j % 2)) / sqrt_n
                y = j / sqrt_n
                points.append([x, y])
    points = np.array(points)
    
    # Scale to roughly unit square, add small noise for diversity
    points = (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0))
    points = 0.05 + 0.9 * points  # Keep away from edges initially
    points += rng.normal(0, 0.01, points.shape)
    points = np.clip(points, 0, 1)
    
    # Simple hill climbing optimization
    best_points = points.copy()
    best_min_area = compute_min_area(points)
    learning_rate = 0.05
    
    for iteration in range(100):
        for i in range(n):
            for _ in range(10):  # Try multiple perturbations per point
                test_points = best_points.copy()
                test_points[i] += rng.normal(0, learning_rate, 2)
                test_points = np.clip(test_points, 0, 1)
                
                current_min = compute_min_area(test_points)
                if current_min > best_min_area:
                    best_min_area = current_min
                    best_points = test_points
        
        learning_rate *= 0.95  # Reduce step size over time
    
    return best_points


# EVOLVE-BLOCK-END