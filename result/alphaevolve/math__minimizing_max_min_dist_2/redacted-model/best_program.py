import numpy as np
from scipy.spatial.distance import pdist


def _compute_ratio(points: np.ndarray) -> float:
    """Compute dmin/dmax ratio for a set of points."""
    dists = pdist(points)
    dmin = np.min(dists)
    dmax = np.max(dists)
    return dmin / dmax if dmax > 0 else 0.0


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Normalize points to fit in [0,1]x[0,1] while preserving relative distances."""
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    span = max_coords - min_coords
    if np.max(span) > 0:
        points = (points - min_coords) / np.max(span)
    return points


def _push_topk_apart(points: np.ndarray, k: int, step: float, idx_map: tuple) -> None:
    """Push k closest pairs apart to improve dmin - replaces inefficient single-pair updates."""
    dists = pdist(points)
    closest_k_idx = np.argpartition(dists, k)[:k]
    for d_idx in closest_k_idx:
        i, j = idx_map[0][d_idx], idx_map[1][d_idx]
        vec = points[j] - points[i]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-8:
            vec /= vec_norm
            points[i] -= step * 0.5 * vec
            points[j] += step * 0.5 * vec


def _pull_topk_together(points: np.ndarray, k: int, step: float, idx_map: tuple) -> None:
    """Pull k farthest pairs together to improve dmax."""
    dists = pdist(points)
    farthest_k_idx = np.argpartition(dists, -k)[-k:]
    for d_idx in farthest_k_idx:
        i, j = idx_map[0][d_idx], idx_map[1][d_idx]
        vec = points[j] - points[i]
        vec_norm = np.linalg.norm(vec)
        if vec_norm > 1e-8:
            vec /= vec_norm
            points[i] += step * 0.5 * vec
            points[j] -= step * 0.5 * vec


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    np.random.seed(42)
    
    n = 16
    d = 2
    k = 3

    # Start with optimal base: regular 4x4 grid in unit square (Exemplar 1&2 improvement over random)
    grid_size = int(np.sqrt(n))
    x_coords = np.linspace(0, 1, grid_size)
    y_coords = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xv.ravel(), yv.ravel()])

    best_ratio = _compute_ratio(points)
    best_points = points.copy()
    
    # Phase 1: Extended gradient-based refinement with top-k operations (hybrid)
    # Now with annealing and additional iterations (1800 total)
    step = 0.0015
    idx_map = np.triu_indices(n, k=1)
    for iter_idx in range(1800):
        annealed_step = step * (1.0 - iter_idx / 1800)
        # Combined: both single pair AND top-k operations
        _push_topk_apart(points, 6, annealed_step, idx_map)
        _pull_topk_together(points, 4, annealed_step, idx_map)
        
        # Plus single extreme pair refinement
        dists = pdist(points)
        dmin_idx = np.argmin(dists)
        dmax_idx = np.argmax(dists)
        
        i_min, j_min = idx_map[0][dmin_idx], idx_map[1][dmin_idx]
        i_max, j_max = idx_map[0][dmax_idx], idx_map[1][dmax_idx]
        
        vec_min = points[j_min] - points[i_min]
        vec_min_norm = np.linalg.norm(vec_min)
        if vec_min_norm > 1e-8:
            vec_min /= vec_min_norm
            points[i_min] -= annealed_step * vec_min
            points[j_min] += annealed_step * vec_min
        
        vec_max = points[j_max] - points[i_max]
        vec_max_norm = np.linalg.norm(vec_max)
        if vec_max_norm > 1e-8:
            vec_max /= vec_max_norm
            points[i_max] += annealed_step * vec_max
            points[j_max] -= annealed_step * vec_max
        
        points = np.clip(points, 0, 1)
        
        ratio = _compute_ratio(points)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

    # Phase 2: Hill climbing with random perturbations - extended iterations, slower annealing
    points = best_points.copy()
    step_size = 0.02
    for iter_idx in range(2000):
        annealed_step = step_size * np.exp(-iter_idx / 400)
        candidate = points + np.random.uniform(-annealed_step, annealed_step, points.shape)
        candidate = np.clip(candidate, 0, 1)
        ratio = _compute_ratio(candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate
            points = candidate
    
    # Phase 3: Fine-grained hill climbing with smaller step
    points = best_points.copy()
    step_size = 0.005
    for iter_idx in range(2000):
        annealed_step = step_size * np.exp(-iter_idx / 400)
        candidate = points + np.random.uniform(-annealed_step, annealed_step, points.shape)
        candidate = np.clip(candidate, 0, 1)
        ratio = _compute_ratio(candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate
            points = candidate
    
    # Phase 4: Ultra-fine grained hill climbing with even smaller step
    points = best_points.copy()
    step_size = 0.001
    for iter_idx in range(2000):
        annealed_step = step_size * np.exp(-iter_idx / 400)
        candidate = points + np.random.uniform(-annealed_step, annealed_step, points.shape)
        candidate = np.clip(candidate, 0, 1)
        ratio = _compute_ratio(candidate)
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = candidate
            points = candidate
    
    # Phase 5: Per-point hill climbing - extended outer loops, finer grid
    for p in range(30):
        for i in range(n):
            original_point = best_points[i].copy()
            for dx in np.linspace(-0.015, 0.015, 20):
                for dy in np.linspace(-0.015, 0.015, 20):
                    if dx == 0 and dy == 0:
                        continue
                    best_points[i] = original_point + np.array([dx, dy])
                    best_points[i] = np.clip(best_points[i], 0, 1)
                    ratio = _compute_ratio(best_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        original_point = best_points[i].copy()
                    else:
                        best_points[i] = original_point.copy()
    
    # Phase 6: Angular gradient ascent refinement
    for iter_idx in range(800):
        for i in range(n):
            best_local = best_points[i].copy()
            for angle in np.linspace(0, 2 * np.pi, 36):
                step = 0.002 * np.exp(-iter_idx / 150)
                test_point = best_local + step * np.array([np.cos(angle), np.sin(angle)])
                test_point = np.clip(test_point, 0, 1)
                best_points[i] = test_point
                ratio = _compute_ratio(best_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_local = test_point.copy()
                else:
                    best_points[i] = best_local
    
    # Phase 7: Final ultra-fine per-point refinement with early exit
    for _ in range(8):
        improved = False
        for i in range(n):
            original_point = best_points[i].copy()
            for dx in np.linspace(-0.006, 0.006, 15):
                for dy in np.linspace(-0.006, 0.006, 15):
                    if dx == 0 and dy == 0:
                        continue
                    best_points[i] = original_point + np.array([dx, dy])
                    best_points[i] = np.clip(best_points[i], 0, 1)
                    ratio = _compute_ratio(best_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        original_point = best_points[i].copy()
                        improved = True
                    else:
                        best_points[i] = original_point.copy()
        if not improved:
            break
    
    # Final normalization to ensure optimal bounding box
    points = _normalize_points(best_points)
    
    return points