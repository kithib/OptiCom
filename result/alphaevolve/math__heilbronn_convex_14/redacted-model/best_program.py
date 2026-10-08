# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    # Use deterministic random seed
    rng = np.random.default_rng(seed=42)
    
    # Hybrid initial configuration: combine best from exemplars
    # Start with hexagonal grid pattern - known effective configuration from exemplars
    points = np.array([
        # Boundary points in convex position from exemplars
        [0.0, 0.0],
        [1.0, 0.0],
        [0.0, 1.0],
        [1.0, 1.0],
        [0.5, 0.05],
        [0.05, 0.5],
        [0.95, 0.5],
        [0.5, 0.95],
        # Interior points in hexagonal pattern for spacing
        [0.25, 0.25],
        [0.75, 0.25],
        [0.25, 0.75],
        [0.75, 0.75],
        [0.5, 0.33],
        [0.5, 0.67],
    ], dtype=np.float64)
    
    # Add small controlled noise to break regularity (exemplar level)
    points += rng.normal(0, 0.02, points.shape)
    points = np.clip(points, 0.0, 1.0)
    
    # Compute min triangle area utility function with optional worst triangle tracking
    def compute_min_area(pts, return_worst=False):
        min_area = float('inf')
        worst_triangle = None
        for i, j, k in combinations(range(len(pts)), 3):
            area = 0.5 * abs(
                (pts[j,0] - pts[i,0]) * (pts[k,1] - pts[i,1]) -
                (pts[j,1] - pts[i,1]) * (pts[k,0] - pts[i,0])
            )
            if area < min_area:
                min_area = area
                worst_triangle = (i, j, k)
        if return_worst:
            return min_area, worst_triangle
        return min_area
    
    # Phase 1: Hill climbing with worst triangle targeting (extended)
    iterations = 600
    for iter_num in range(iterations):
        # Find the worst triangle
        min_area, worst_triangle = compute_min_area(points, return_worst=True)
        
        if worst_triangle is None:
            continue
            
        # Hybrid strategy: rotate through different move types
        strategy = iter_num % 6
        
        if strategy == 0:
            # Strategy 1: nudge away from center (spreading)
            step = 0.008
            center = np.mean(points, axis=0)
            for idx in worst_triangle:
                direction = points[idx] - center
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    old_point = points[idx].copy()
                    points[idx] = np.clip(points[idx] + direction * step, 0.0, 1.0)
                    new_min = compute_min_area(points)
                    if new_min < min_area:
                        points[idx] = old_point
        elif strategy == 1:
            # Strategy 2: nudge toward convex hull boundary
            step = 0.01
            for idx in worst_triangle:
                old_point = points[idx].copy()
                # Try pushing toward boundaries
                for target in [[0.0, old_point[1]], [1.0, old_point[1]], 
                               [old_point[0], 0.0], [old_point[0], 1.0]]:
                    direction = np.array(target) - old_point
                    norm = np.linalg.norm(direction)
                    if norm > 1e-10:
                        points[idx] = np.clip(old_point + direction/norm * step, 0.0, 1.0)
                        new_min = compute_min_area(points)
                        if new_min >= min_area:
                            min_area = new_min
                        else:
                            points[idx] = old_point
        elif strategy == 2:
            # Strategy 3: pairwise vertex repulsion - spread points in worst triangle
            step = 0.006
            a, b, c = worst_triangle
            for i, j in [(a, b), (a, c), (b, c)]:
                direction = points[i] - points[j]
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    direction = direction / norm
                    old_i, old_j = points[i].copy(), points[j].copy()
                    points[i] = np.clip(points[i] + direction * step, 0.0, 1.0)
                    points[j] = np.clip(points[j] - direction * step, 0.0, 1.0)
                    new_min = compute_min_area(points)
                    if new_min < min_area:
                        points[i], points[j] = old_i, old_j
        else:
            # Strategy 4-5: controlled perturbations with adaptive step
            pt_idx = worst_triangle[rng.integers(0, 3)]
            old_point = points[pt_idx].copy()
            # Adaptive step size - gradual reduction with decay
            step_size = 0.04 * (1 - iter_num / iterations) * 0.95 + 0.01
            points[pt_idx] += rng.normal(0, step_size, 2)
            points[pt_idx] = np.clip(points[pt_idx], 0.0, 1.0)
            new_min = compute_min_area(points)
            if new_min < min_area:
                points[pt_idx] = old_point
    
    # Phase 1.5: Additional intermediate refinement with moderate moves
    for move_scale in [0.02, 0.015, 0.01, 0.008]:
        for iter_num in range(90):
            min_area = compute_min_area(points)
            pt_order = np.arange(n)
            rng.shuffle(pt_order)
            for pt_idx in pt_order:
                old_point = points[pt_idx].copy()
                points[pt_idx] += rng.normal(0, move_scale, 2)
                points[pt_idx] = np.clip(points[pt_idx], 0.0, 1.0)
                new_min = compute_min_area(points)
                if new_min < min_area:
                    points[pt_idx] = old_point

    # Phase 2: Enhanced fine-grained refinement with gradient tracking
    # Multi-scale refinement with greedy acceptance threshold
    for move_scale in [0.008, 0.006, 0.005, 0.004, 0.003, 0.002, 0.0015, 0.001]:
        for iter_num in range(120):
            min_area, (a_idx, b_idx, c_idx) = compute_min_area(points, return_worst=True)
            # Process worst triangle points FIRST, then all points in random order
            worst_set = {a_idx, b_idx, c_idx}
            other_pts = [i for i in range(n) if i not in worst_set]
            rng.shuffle(other_pts)
            pt_order = list(worst_set) + other_pts
            for pt_idx in pt_order:
                old_point = points[pt_idx].copy()
                improved = False
                # Enhanced multi-directional moves: combine 8-directional with diagonal first
                # First diagonal moves (using sqrt(2)/2 for equal diagonal length)
                for dx, dy in [(-move_scale*0.707, -move_scale*0.707), 
                               (move_scale*0.707, -move_scale*0.707),
                               (-move_scale*0.707, move_scale*0.707), 
                               (move_scale*0.707, move_scale*0.707)]:
                    points[pt_idx] = np.clip(old_point + np.array([dx, dy]), 0.0, 1.0)
                    new_min = compute_min_area(points)
                    if new_min >= min_area:
                        min_area = new_min
                        improved = True
                    else:
                        points[pt_idx] = old_point
                # Then axis-aligned moves
                for dx, dy in [(-move_scale, 0), (move_scale, 0), 
                               (0, -move_scale), (0, move_scale)]:
                    points[pt_idx] = np.clip(old_point + np.array([dx, dy]), 0.0, 1.0)
                    new_min = compute_min_area(points)
                    if new_min >= min_area:
                        min_area = new_min
                        improved = True
                    else:
                        points[pt_idx] = old_point
                # Finer intermediate angles for very precise tuning (now at all scales <= 0.008)
                if move_scale <= 0.008:
                    for angle in np.linspace(np.pi/12, 23*np.pi/12, 16, endpoint=False):
                        dx = move_scale * np.cos(angle)
                        dy = move_scale * np.sin(angle)
                        points[pt_idx] = np.clip(old_point + np.array([dx, dy]), 0.0, 1.0)
                        new_min = compute_min_area(points)
                        if new_min >= min_area:
                            min_area = new_min
                            improved = True
                        else:
                            points[pt_idx] = old_point
                if not improved:
                    points[pt_idx] = old_point

    return points


# EVOLVE-BLOCK-END