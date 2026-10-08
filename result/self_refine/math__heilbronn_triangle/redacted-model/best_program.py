import numpy as np
from itertools import combinations


def heilbronn_triangle11() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 11.

    Returns:
        points: np.ndarray of shape (11,2) with the x,y coordinates of the points.
    """
    n = 11
    rng = np.random.RandomState(42)
    
    sqrt3 = np.sqrt(3)
    # Start with an improved grid-based initial configuration from top exemplar
    points = np.array([
        [0.0, 0.0],                      # Corner
        [1.0, 0.0],                      # Corner
        [0.5, sqrt3 / 2],                # Corner
        [0.25, 0.0],                     # Bottom edge mid-left
        [0.75, 0.0],                     # Bottom edge mid-right
        [0.125, sqrt3 / 8],              # Interior lower left
        [0.875, sqrt3 / 8],              # Interior lower right
        [0.5, sqrt3 / 4],                # Interior lower center
        [0.25, 3 * sqrt3 / 8],           # Interior upper left
        [0.75, 3 * sqrt3 / 8],           # Interior upper right
        [0.5, 5 * sqrt3 / 16],           # Interior upper center
    ], dtype=np.float64)
    
    # Precompute combination indices once for efficiency
    combo_indices = list(combinations(range(n), 3))
    
    def min_triangle_area_2x_fast(pts):
        min_a = float('inf')
        pts_arr = np.asarray(pts)
        for i, j, k in combo_indices:
            cross = (pts_arr[j, 0] - pts_arr[i, 0]) * (pts_arr[k, 1] - pts_arr[i, 1]) - \
                    (pts_arr[j, 1] - pts_arr[i, 1]) * (pts_arr[k, 0] - pts_arr[i, 0])
            a = abs(cross)
            if a < min_a:
                if a < 1e-12:  # Early exit for degenerate triangles
                    return 0.0
                min_a = a
        return min_a
    
    # Hill climbing optimization with improved search strategy
    best_points = points.copy()
    best_min = min_triangle_area_2x_fast(points)
    step_size = 0.09
    
    # Threshold for early exit (target significantly above incumbent)
    target_2x_area = 2.0 * 0.036529889880030156 * 1.15
    
    for iteration in range(2200):
        # Early exit if we've achieved a very good score
        if best_min >= target_2x_area:
            break
            
        # Randomize the order we try points to avoid order bias
        indices = np.arange(n)
        rng.shuffle(indices)
        
        improved = False
        for idx in indices:
            old_pt = points[idx].copy()
            best_candidate = old_pt
            candidate_best = best_min
            
            # Fewer candidates per point with better sampling
            for attempt in range(28):
                # Scale steps for boundary corners/edges appropriately
                if idx in [0, 1, 2]:  # Corner vertices - very restricted movement
                    scale = 0.02
                elif idx in [3, 4]:  # Bottom edge mid points
                    scale = 0.12
                else:
                    # Mix of Gaussian and uniform sampling
                    if attempt % 4 == 0:
                        scale = rng.normal(1.0, 0.25)
                    else:
                        scale = 1.0
                
                dx = rng.uniform(-step_size * scale, step_size * scale)
                dy = rng.uniform(-step_size * scale, step_size * scale)
                new_x = old_pt[0] + dx
                new_y = old_pt[1] + dy
                
                # Check inside triangle with small epsilon for boundary
                if (new_x >= -1e-10 and new_x <= 1 + 1e-10 and new_y >= -1e-10 and
                    new_y <= sqrt3 * new_x + 1e-10 and
                    new_y <= -sqrt3 * (new_x - 0.5) + sqrt3 / 2 + 1e-10):
                    
                    points[idx] = [new_x, new_y]
                    current_min = min_triangle_area_2x_fast(points)
                    
                    if current_min > candidate_best:
                        candidate_best = current_min
                        best_candidate = [new_x, new_y]
            
            # Accept the best candidate found for this point
            if candidate_best > best_min:
                best_min = candidate_best
                best_points = points.copy()
                improved = True
            points[idx] = best_candidate
        
        # Periodic large step to escape local minima - tuned frequency and range
        if iteration > 0 and iteration % 180 == 0 and not improved:
            for idx in range(3, n):
                points[idx] += rng.uniform(-0.07, 0.07, 2)
                # Project back inside triangle if needed
                points[idx, 0] = np.clip(points[idx, 0], 0, 1)
                y_max1 = sqrt3 * points[idx, 0]
                y_max2 = -sqrt3 * (points[idx, 0] - 0.5) + sqrt3 / 2
                points[idx, 1] = np.clip(points[idx, 1], 0, min(y_max1, y_max2))
        
        # Adaptive step size reduction - faster reduction schedule
        if iteration % 500 == 499:
            step_size *= 0.42
            # Reset to best known point when reducing step
            points = best_points.copy()
    
    return best_points