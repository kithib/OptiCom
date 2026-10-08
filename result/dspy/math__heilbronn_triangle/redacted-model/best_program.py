import numpy as np
from itertools import combinations


def _compute_min_area_fast(points: np.ndarray) -> float:
    """Compute the minimum triangle area from all triplets of points using vectorized operations."""
    n = len(points)
    diffs = points[:, np.newaxis] - points[np.newaxis, :]
    min_area2 = np.inf
    
    for i in range(n - 2):
        for j in range(i + 1, n - 1):
            dx_ij, dy_ij = diffs[i, j]
            dx_ik = diffs[i, j+1:, 0]
            dy_ik = diffs[i, j+1:, 1]
            areas2 = np.abs(dx_ij * dy_ik - dy_ij * dx_ik)
            curr_min = areas2.min()
            if curr_min < min_area2:
                min_area2 = curr_min
    
    return min_area2 / 2.0


def _get_worst_triplet(points: np.ndarray) -> tuple:
    """Find the indices of the triplet forming the smallest area triangle."""
    n = len(points)
    min_area2 = np.inf
    worst_triplet = None
    
    for i, j, k in combinations(range(n), 3):
        area2 = np.abs((points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                       (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0]))
        if area2 < min_area2:
            min_area2 = area2
            worst_triplet = (i, j, k)
    
    return worst_triplet


def _get_multiple_worst_triplets(points: np.ndarray, num: int = 3) -> set:
    """Get indices from multiple small-area triplets for better targeted improvement."""
    n = len(points)
    areas = []
    for i, j, k in combinations(range(n), 3):
        area2 = np.abs((points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                       (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0]))
        areas.append((area2, {i, j, k}))
    areas.sort(key=lambda x: x[0])
    result = set()
    for _, idx_set in areas[:num]:
        result.update(idx_set)
    return result


def _compute_all_areas2_with_points(points: np.ndarray) -> tuple:
    """Compute all twice-areas and track which triplet has minimum."""
    n = len(points)
    min_area2 = np.inf
    worst = None
    for i, j, k in combinations(range(n), 3):
        area2 = np.abs((points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
                       (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0]))
        if area2 < min_area2:
            min_area2 = area2
            worst = (i, j, k)
    return min_area2, worst


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    sqrt3 = np.sqrt(3)
    sqrt3_half = sqrt3 / 2.0
    
    def is_inside(p):
        x, y = p
        return (y >= -1e-12 and 
                y <= sqrt3 * x + 1e-12 and 
                y <= sqrt3 * (1 - x) + 1e-12)
    
    def project_to_triangle(p):
        """Project point to be inside/on the triangle boundary with proper boundary handling."""
        x, y = p
        y = max(0.0, y)
        max_y_left = sqrt3 * x
        max_y_right = sqrt3 * (1 - x)
        max_y = min(max_y_left, max_y_right)
        
        if y > max_y + 1e-12:
            if max_y_left < max_y_right:
                t = (y + sqrt3 * x) / (2 * sqrt3)
                x = max(0.0, min(0.5, t))
                y = sqrt3 * x
            else:
                t = (y - sqrt3 * x + sqrt3) / (2 * sqrt3)
                t = max(0.0, min(0.5, t))
                x = 1.0 - t
                y = sqrt3 * t
        
        x = max(0.0, min(1.0, x))
        y = max(0.0, min(sqrt3_half, y))
        return np.array([x, y])
    
    # Highly optimized symmetric initial configuration from best exemplars
    points = np.array([
        [0.0, 0.0],                              # Vertex A
        [1.0, 0.0],                              # Vertex B
        [0.5, sqrt3_half],                       # Vertex C
        [0.116, 0.0],                            # Edge AB, left - refined
        [0.884, 0.0],                            # Edge AB, right - symmetric
        [0.056, 0.0969948435562814],            # Left edge lower
        [0.944, 0.0969948435562814],            # Right edge lower
        [0.267, 0.4615198765991102],            # Left interior mid
        [0.733, 0.4615198765991102],            # Right interior mid
        [0.396, 0.6858689530403034],            # Near top left
        [0.604, 0.6858689530403034]             # Near top right
    ])
    
    # Ensure initial validity
    for i in range(len(points)):
        points[i] = project_to_triangle(points[i])
    
    rng = np.random.default_rng(seed=12345)
    current_min = _compute_min_area_fast(points)
    best_points = points.copy()
    best_min = current_min
    
    # Enhanced starting variants - more diversity with controlled perturbations
    restart_points = [points.copy()]
    
    # Config 2: Slightly different edge spacing from Exemplar 1
    variant2 = np.array([
        [0.0, 0.0], [1.0, 0.0], [0.5, sqrt3_half],
        [0.122, 0.0], [0.878, 0.0],
        [0.052, 0.092], [0.948, 0.092],
        [0.262, 0.456], [0.738, 0.456],
        [0.393, 0.682], [0.607, 0.682]
    ])
    for i in range(len(variant2)):
        variant2[i] = project_to_triangle(variant2[i])
    restart_points.append(variant2)
    
    # Additional perturbed variants
    for seed_offset, scale in enumerate([0.011, 0.019, 0.026, 0.009, 0.017, 0.032]):
        variant_rng = np.random.default_rng(seed=12345 + seed_offset * 100)
        variant = points.copy()
        variant[3:] += variant_rng.normal(0, scale, (8, 2))
        for i in range(3, n):
            variant[i] = project_to_triangle(variant[i])
        restart_points.append(variant)
    
    for start_points in restart_points:
        points = start_points.copy()
        current_min = _compute_min_area_fast(points)
        
        # Enhanced optimization schedule - extended iterations with better temperature profile
        for phase, (iterations, step_start, step_end, temp_base) in enumerate([
            (1450, 0.125, 0.038, 0.022),  # Extended global exploration with higher temp
            (1850, 0.048, 0.013, 0.0055),  # Deep refinement with moderate temp
            (1200, 0.020, 0.0055, 0.0016),  # Fine tuning
            (650, 0.007, 0.0012, 0.00045),  # Ultra-fine polishing
        ]):
            for it in range(iterations):
                alpha = it / iterations
                step = step_start * (1 - alpha) + step_end * alpha
                current_temp = temp_base * (1 - alpha)
                
                # Get multiple worst triplets frequently for targeted improvement
                target_indices = set()
                if it % 22 == 0:
                    target_indices = _get_multiple_worst_triplets(points, num=3)
                elif it % 35 == 0:
                    min_a2, worst = _compute_all_areas2_with_points(points)
                    if worst:
                        target_indices = set(worst)
                
                # Allow all points to move with adaptive constraints
                for idx in range(n):
                    # Freeze vertices in later phases for stability
                    if idx < 3 and phase > 1:
                        continue
                    
                    # More attempts for points in small-area triplets
                    base_tries = 6 if idx in target_indices else 3
                    # Extra attempts for interior points (more flexibility)
                    num_tries = base_tries + 2 if idx >= 3 else base_tries
                    
                    for _ in range(num_tries):
                        dir = rng.normal(0, 1, 2)
                        dir_norm = np.linalg.norm(dir)
                        if dir_norm > 0:
                            dir = dir / dir_norm
                        
                        # Smaller steps for vertices
                        effective_step = step * 0.45 if idx < 3 else step
                        # Adaptive step size based on boundary proximity
                        if idx >= 3:
                            px, py = points[idx]
                            if (py < 0.025 or 
                                abs(py - sqrt3 * px) < 0.025 or 
                                abs(py - sqrt3 * (1 - px)) < 0.025):
                                effective_step *= 0.65
                        
                        candidate_pt = points[idx] + dir * effective_step
                        candidate_pt = project_to_triangle(candidate_pt)
                        
                        if not is_inside(candidate_pt):
                            continue
                        
                        old_val = points[idx].copy()
                        points[idx] = candidate_pt
                        new_min = _compute_min_area_fast(points)
                        
                        # Always accept improvements
                        if new_min > current_min:
                            current_min = new_min
                            if current_min > best_min:
                                best_min = current_min
                                best_points = points.copy()
                        # Better acceptance probability function
                        elif current_temp > 0:
                            delta = (new_min - current_min) * 200  # Scale for better probability curve
                            if rng.random() < np.exp(delta / max(current_temp, 1e-10)):
                                current_min = new_min
                            else:
                                points[idx] = old_val
                        else:
                            points[idx] = old_val
    
    # Final validation and cleanup
    for idx in range(len(best_points)):
        best_points[idx] = project_to_triangle(best_points[idx])
        best_points[idx][0] = max(0.0, min(1.0, best_points[idx][0]))
        best_points[idx][1] = max(0.0, min(sqrt3_half, best_points[idx][1]))
    
    return best_points