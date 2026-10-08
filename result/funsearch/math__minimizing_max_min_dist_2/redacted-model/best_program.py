import numpy as np
from scipy.spatial.distance import pdist, squareform


def _compute_ratio(pts: np.ndarray) -> float:
    """Compute the min/max distance ratio for a set of points."""
    dists = pdist(pts)
    return np.min(dists) / np.max(dists)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a perturbed 4x4 grid initialization (slightly larger initial perturbations for exploration)
    followed by hybrid multi-phase gradient-based optimization (extended 700-iteration LR-annealed phase 
    with 0.6 max weight + 300-iteration finer constant LR phase with 0.5 weight) and comprehensive 
    enhanced hill climbing local search (combined multi-scale perturbation strategy from top exemplars, 
    asymptotic diminishing-scale fine-tuning, and extended ultra-fine micro-adjustment phase with 
    enhanced perturbation sets and tight epsilon comparisons) for optimal convergence.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    np.random.seed(42)
    n = 16
    # Initialize with a 4x4 grid with slight perturbation (better than random)
    grid_size = 4
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x, y)
    points = np.stack([xv.ravel(), yv.ravel()], axis=1).astype(np.float64)
    # Add controlled initial perturbation (slightly larger for more initial exploration)
    points += 0.025 * np.random.randn(n, 2)
    # Clip to unit square
    points = np.clip(points, 0.0, 1.0)

    triu_i, triu_j = np.triu_indices(n, k=1)  # Precompute for efficiency

    # Phase 1: Extended gradient-based optimization with LR decay (700 iterations) with 0.6 max weight
    iterations = 700
    initial_lr = 0.005
    for iter_idx in range(iterations):
        lr = initial_lr * (1 - iter_idx / iterations) * 0.6 + 0.002
        dists = pdist(points)
        dmin_idx = np.argmin(dists)
        dmax_idx = np.argmax(dists)
        i_min, j_min = triu_i[dmin_idx], triu_j[dmin_idx]
        i_max, j_max = triu_i[dmax_idx], triu_j[dmax_idx]
        diff_min = points[i_min] - points[j_min]
        if np.linalg.norm(diff_min) > 1e-8:
            grad_min = diff_min / np.linalg.norm(diff_min)
            points[i_min] += lr * grad_min
            points[j_min] -= lr * grad_min
        # 0.6 weight for max distance contraction
        diff_max = points[i_max] - points[j_max]
        if np.linalg.norm(diff_max) > 1e-8:
            grad_max = diff_max / np.linalg.norm(diff_max)
            points[i_max] -= lr * 0.6 * grad_max
            points[j_max] += lr * 0.6 * grad_max
        points = np.clip(points, 0.0, 1.0)
    
    # Phase 2: Lower learning rate for finer convergence (300 iterations, 0.5 weight)
    iterations = 300
    lr = 0.002
    for _ in range(iterations):
        dists = pdist(points)
        dmin_idx = np.argmin(dists)
        dmax_idx = np.argmax(dists)
        i_min, j_min = triu_i[dmin_idx], triu_j[dmin_idx]
        i_max, j_max = triu_i[dmax_idx], triu_j[dmax_idx]
        diff_min = points[i_min] - points[j_min]
        if np.linalg.norm(diff_min) > 1e-8:
            grad_min = diff_min / np.linalg.norm(diff_min)
            points[i_min] += lr * grad_min
            points[j_min] -= lr * grad_min
        diff_max = points[i_max] - points[j_max]
        if np.linalg.norm(diff_max) > 1e-8:
            grad_max = diff_max / np.linalg.norm(diff_max)
            points[i_max] -= lr * 0.5 * grad_max
            points[j_max] += lr * 0.5 * grad_max
        points = np.clip(points, 0.0, 1.0)

    current_ratio = _compute_ratio(points)
    
    # First enhanced hill climbing phase (comprehensive perturbation set from all exemplars)
    for hill_climb_phase in range(40):
        improved = False
        for i in range(n):
            # Comprehensive perturbation set: multiple scales (axis-aligned + diagonal + ultra-fine)
            for dx, dy in [(-0.02, 0), (0.02, 0), (0, -0.02), (0, 0.02),
                          (-0.015, 0), (0.015, 0), (0, -0.015), (0, 0.015),
                          (-0.01, 0), (0.01, 0), (0, -0.01), (0, 0.01),
                          (-0.0075, 0), (0.0075, 0), (0, -0.0075), (0, 0.0075),
                          (-0.005, 0), (0.005, 0), (0, -0.005), (0, 0.005),
                          (-0.0025, 0), (0.0025, 0), (0, -0.0025), (0, 0.0025),
                          (-0.015, -0.015), (-0.015, 0.015),
                          (0.015, -0.015), (0.015, 0.015),
                          (-0.01, -0.01), (-0.01, 0.01),
                          (0.01, -0.01), (0.01, 0.01),
                          (-0.0075, -0.0075), (-0.0075, 0.0075),
                          (0.0075, -0.0075), (0.0075, 0.0075),
                          (-0.005, -0.005), (-0.005, 0.005),
                          (0.005, -0.005), (0.005, 0.005),
                          (-0.0025, -0.0025), (-0.0025, 0.0025),
                          (0.0025, -0.0025), (0.0025, 0.0025)]:
                new_points = points.copy()
                new_points[i, 0] = np.clip(new_points[i, 0] + dx, 0, 1)
                new_points[i, 1] = np.clip(new_points[i, 1] + dy, 0, 1)
                new_ratio = _compute_ratio(new_points)
                if new_ratio > current_ratio + 1e-12:
                    points = new_points
                    current_ratio = new_ratio
                    improved = True
        if not improved:
            break

    # Diminishing perturbation fine-tuning phase (asymptotic convergence)
    for fine_tune_round in range(20):
        improved = False
        scale = 0.003 * (1 - fine_tune_round / 20)
        if scale < 0.0005:
            break
        for i in range(n):
            for dx, dy in [(-scale, 0), (scale, 0), (0, -scale), (0, scale),
                          (-scale, -scale), (-scale, scale), (scale, -scale), (scale, scale)]:
                new_points = points.copy()
                new_points[i, 0] = np.clip(new_points[i, 0] + dx, 0, 1)
                new_points[i, 1] = np.clip(new_points[i, 1] + dy, 0, 1)
                new_ratio = _compute_ratio(new_points)
                if new_ratio > current_ratio + 1e-12:
                    points = new_points
                    current_ratio = new_ratio
                    improved = True
        if not improved:
            break

    # Extended ultra-fine phase - micro adjustments with tight epsilon (enhanced from current artifact)
    for _ in range(30):  # Extended from 25
        improved = False
        for i in range(n):
            # Enhanced: more very small perturbation sizes for ultra-fine tuning (from current artifact)
            for dx, dy in [(-0.003, 0), (0.003, 0), (0, -0.003), (0, 0.003),
                          (-0.0025, 0), (0.0025, 0), (0, -0.0025), (0, 0.0025),
                          (-0.002, 0), (0.002, 0), (0, -0.002), (0, 0.002),
                          (-0.0015, 0), (0.0015, 0), (0, -0.0015), (0, 0.0015),
                          (-0.001, 0), (0.001, 0), (0, -0.001), (0, 0.001),
                          (-0.0005, 0), (0.0005, 0), (0, -0.0005), (0, 0.0005),
                          (-0.002, -0.002), (-0.002, 0.002),
                          (0.002, -0.002), (0.002, 0.002),
                          (-0.0015, -0.0015), (-0.0015, 0.0015),
                          (0.0015, -0.0015), (0.0015, 0.0015),
                          (-0.001, -0.001), (-0.001, 0.001),
                          (0.001, -0.001), (0.001, 0.001)]:
                new_points = points.copy()
                new_points[i, 0] = np.clip(new_points[i, 0] + dx, 0, 1)
                new_points[i, 1] = np.clip(new_points[i, 1] + dy, 0, 1)
                new_ratio = _compute_ratio(new_points)
                # Tight epsilon comparison for very small improvements
                if new_ratio > current_ratio + 1e-15:
                    points = new_points
                    current_ratio = new_ratio
                    improved = True
        if not improved:
            break

    return points