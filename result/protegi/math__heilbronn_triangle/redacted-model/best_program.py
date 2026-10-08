import numpy as np
from itertools import combinations


SQRT3 = np.sqrt(3)
INV_SQRT3 = 1.0 / SQRT3


def compute_min_area(points: np.ndarray) -> float:
    """Compute the minimum triangle area from all triplets of points using vectorized operations for efficiency."""
    n = len(points)
    if n < 3:
        return 0.0
    
    triplets = np.array(list(combinations(range(n), 3)))
    a, b, c = points[triplets[:, 0]], points[triplets[:, 1]], points[triplets[:, 2]]
    
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)


def clamp_to_triangle(point: np.ndarray) -> np.ndarray:
    """Clamp a point to stay within the bounds of the equilateral triangle."""
    x, y = point
    y = max(0.0, min(y, SQRT3 / 2))
    max_x_at_y = 1.0 - y * 2.0 * INV_SQRT3
    half_width = max_x_at_y / 2.0
    x = max(0.5 - half_width, min(x, 0.5 + half_width))
    return np.array([x, y])


def optimize_points(initial_points: np.ndarray, n_iters: int = 4000) -> np.ndarray:
    """Optimize point placement using hill climbing with adaptive step size, momentum, and multi-phase refinement."""
    np.random.seed(42)
    n = len(initial_points)
    points = initial_points.copy()
    best_min_area = compute_min_area(points)
    best_points = points.copy()
    
    step_size = 0.06
    step_decay = 0.9987
    momentum = np.zeros((n, 2))
    momentum_factor = 0.35
    patience = 0
    
    for iteration in range(n_iters):
        if patience > 220:
            step_size = max(0.005, step_size * 0.86)
            patience = 0
        
        idx = np.random.randint(n)
        perturbation = np.random.normal(0, step_size, 2) + momentum_factor * momentum[idx]
        new_points = points.copy()
        new_points[idx] = clamp_to_triangle(new_points[idx] + perturbation)
        
        current_min_area = compute_min_area(new_points)
        if current_min_area > best_min_area:
            momentum[idx] = perturbation
            points = new_points
            best_points = new_points
            best_min_area = current_min_area
            patience = 0
        elif current_min_area >= best_min_area:
            momentum[idx] = momentum_factor * momentum[idx] + (1 - momentum_factor) * perturbation
            points = new_points
            patience += 1
        else:
            patience += 1
        
        step_size *= step_decay
    
    step_size = 0.0075
    for _ in range(1500):
        for idx in range(n):
            perturbation = np.random.normal(0, step_size, 2)
            new_points = points.copy()
            new_points[idx] = clamp_to_triangle(new_points[idx] + perturbation)
            
            current_min_area = compute_min_area(new_points)
            if current_min_area > best_min_area:
                points = new_points
                best_points = new_points
                best_min_area = current_min_area
            elif current_min_area >= best_min_area:
                points = new_points
    
    step_size = 0.0028
    for _ in range(800):
        for idx in range(n):
            perturbation = np.random.normal(0, step_size, 2)
            new_points = points.copy()
            new_points[idx] = clamp_to_triangle(new_points[idx] + perturbation)
            
            current_min_area = compute_min_area(new_points)
            if current_min_area > best_min_area:
                points = new_points
                best_points = new_points
                best_min_area = current_min_area
            elif current_min_area >= best_min_area:
                points = new_points
    
    return best_points


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    initial = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, SQRT3 / 2],
        [0.21, 0.0],
        [0.79, 0.0],
        [0.148, 0.235],
        [0.852, 0.235],
        [0.33, 0.455],
        [0.67, 0.455],
        [0.412, 0.675],
        [0.588, 0.675]
    ])
    points = optimize_points(initial, n_iters=4000)
    return points