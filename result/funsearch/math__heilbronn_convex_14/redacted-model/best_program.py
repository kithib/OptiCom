import numpy as np
from itertools import combinations


def compute_min_triangle_area(points: np.ndarray) -> float:
    """Compute the area of the smallest triangle formed by any three points."""
    min_area = float('inf')
    for trio in combinations(points, 3):
        a, b, c = trio
        cross = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        area = 0.5 * abs(cross)
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
    rng = np.random.default_rng(seed=42)

    # Hybrid grid initialization: Per-point grid jitter (Exemplars 1&2) + dual noise approach
    grid_size = int(np.ceil(np.sqrt(n)))
    grid_points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5 + rng.uniform(-0.2, 0.2)) / grid_size
            y = (j + 0.5 + rng.uniform(-0.2, 0.2)) / grid_size
            grid_points.append([x, y])
    points = np.array(grid_points[:n])
    # Additional dual-approach jitter for extra variation (all exemplars + current)
    points += rng.normal(0, 0.02, points.shape)
    points = np.clip(points, 0.01, 0.99)  # Keep hull area near 1

    # Ensure we have exactly n points (safety from all exemplars)
    if len(points) < n:
        extra = rng.random((n - len(points), 2)) * 0.98 + 0.01
        points = np.vstack([points, extra])

    # Optimized threshold acceptance with annealed steps and balanced exploration
    current_min = compute_min_triangle_area(points)
    total_iterations = 700  # Further increased for deeper search
    thresholds = np.linspace(0.01, 0.0005, total_iterations)  # Smoother schedule
    rng_opt = np.random.default_rng(seed=123)
    base_step = 0.025

    for t in range(total_iterations):
        # Mixed point selection: cyclic thoroughness + occasional random exploration (Exemplar 1)
        if t % 5 == 0:
            point_idx = rng_opt.integers(0, n)  # Random every 5th step
        else:
            point_idx = t % n  # Cyclic through points for systematic coverage

        old_point = points[point_idx].copy()
        # Adaptive step size: linear decay (exploit) + annealed scaling (explore schedule)
        linear_step = 0.03 - (t / total_iterations) * 0.025
        anneal_scale = 0.5 ** (t // 130)  # Halve step every 130 iterations (Exemplar 1)

        improved = False
        # 12 direction trials (boosted exploration per iteration)
        for _ in range(12):
            direction = rng_opt.standard_normal(2)
            direction = direction / (np.linalg.norm(direction) + 1e-8)
            new_point = old_point + linear_step * anneal_scale * direction
            new_point = np.clip(new_point, 0.01, 0.99)

            test_points = points.copy()
            test_points[point_idx] = new_point
            new_min = compute_min_triangle_area(test_points)

            # Threshold acceptance: strict improvement + probabilistic escape from plateaus
            if new_min >= current_min or (new_min > current_min * (1 - thresholds[t])):
                points = test_points
                current_min = new_min
                improved = True
                break

        if not improved:
            points[point_idx] = old_point

    return points