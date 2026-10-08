import numpy as np


def _compute_min_area(points: np.ndarray) -> float:
    """Compute the minimum normalized triangle area from all triplets of points."""
    n = len(points)
    min_area = float('inf')
    # Use vectorized cross product for efficiency: area = 0.5 * |(B-A) × (C-A)|
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                # Compute cross product directly for speed
                cross = (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) - \
                        (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
                area = 0.5 * abs(cross)
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
    sqrt3 = np.sqrt(3)
    rng = np.random.default_rng(42)  # Reproducible deterministic seed
    
    # Generate initial configuration: hand-tuned points from top exemplars (better spread)
    points = np.zeros((n, 2))
    # Include boundary vertices for good coverage
    points[0] = [0.0, 0.0]
    points[1] = [1.0, 0.0]
    points[2] = [0.5, sqrt3/2]
    # Interior points from exemplar's high-performing asymmetric configuration (better spacing)
    points[3] = [0.5, 0.0]
    points[4] = [0.25, sqrt3/4]
    points[5] = [0.75, sqrt3/4]
    points[6] = [0.5, sqrt3/3]
    points[7] = [0.15, sqrt3/6]
    points[8] = [0.85, sqrt3/6]
    points[9] = [0.35, sqrt3/2.8]
    points[10] = [0.65, sqrt3/2.8]
    
    # Hill-climbing optimization: refined with more iterations, larger initial step, better perturbation count
    best_min_area = _compute_min_area(points)
    for iteration in range(1000):  # Balanced iteration count between best variants
        step_size = 0.13 * (1 - iteration / 1000)  # Larger initial step size from exemplars
        for i in range(n):
            orig = points[i].copy()
            # Try more perturbations per point from top exemplar (25 vs original 30, better ratio)
            for __ in range(25):
                # Generate perturbation centered around original point
                perturb = rng.uniform(-step_size, step_size, 2)
                candidate = orig + perturb
                # Project candidate back into equilateral triangle bounds with proper order from exemplars
                x, y = candidate
                y = np.clip(y, 0, sqrt3/2)
                if y > sqrt3 * x:
                    y = sqrt3 * x
                if y > sqrt3 * (1 - x):
                    y = sqrt3 * (1 - x)
                x = np.clip(x, 0, 1)
                candidate = np.array([x, y])
                
                points[i] = candidate
                current_min = _compute_min_area(points)
                if current_min > best_min_area:
                    best_min_area = current_min
                    break
            else:
                # If no improvement with small perturbations, try barycentric random sample
                s, t = rng.uniform(0, 1, 2)
                if s + t > 1:
                    s, t = 1 - t, 1 - s
                candidate = np.array([s, t]) @ np.array([[1, 0], [0.5, sqrt3/2]])
                points[i] = candidate
                current_min = _compute_min_area(points)
                if current_min >= best_min_area:
                    best_min_area = current_min
                else:
                    points[i] = orig  # Revert if worse or equal (exemplar's better condition)
    
    return points