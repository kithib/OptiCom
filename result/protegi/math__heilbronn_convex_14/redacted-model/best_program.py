# EVOLVE-BLOCK-START
import numpy as np
from itertools import combinations


def _compute_min_triangle_area_x2_and_triplet(points: np.ndarray) -> tuple[float, tuple]:
    """Compute TWICE the area of the smallest triangle (faster) and its indices."""
    min_area_x2 = float('inf')
    min_triplet = None
    for trio in combinations(range(len(points)), 3):
        a, b, c = points[trio[0]], points[trio[1]], points[trio[2]]
        area_x2 = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area_x2 < min_area_x2:
            min_area_x2 = area_x2
            min_triplet = trio
    return min_area_x2, min_triplet


def _compute_min_triangle_area_x2_given_point(points: np.ndarray, idx: int) -> float:
    """Fast computation of min twice-area for triangles involving point at idx."""
    min_area_x2 = float('inf')
    n = len(points)
    pt = points[idx]
    for j in range(n):
        if j == idx:
            continue
        b = points[j]
        bx_a = b[0] - pt[0]
        by_a = b[1] - pt[1]
        for k in range(j+1, n):
            if k == idx:
                continue
            c = points[k]
            area_x2 = abs(bx_a * (c[1] - pt[1]) - by_a * (c[0] - pt[0]))
            if area_x2 < min_area_x2:
                min_area_x2 = area_x2
    return min_area_x2


def _compute_convex_hull_area(points: np.ndarray) -> float:
    """Compute convex hull area using Graham scan."""
    if len(points) < 3:
        return 0.0
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    pts = points[np.lexsort(points.T[::-1])]
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    hull = np.array(lower[:-1] + upper[:-1])
    
    if len(hull) < 3:
        return 0.0
    x, y = hull[:, 0], hull[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    rng = np.random.default_rng(seed=42)
    
    # Try multiple perturbed 4x4 grid configurations, select best starting point
    best_initial_score = -1.0
    best_initial_points = None
    
    # Increased initial search count for better starting configuration
    for _ in range(20):
        # Generate a 4x4 grid with small random perturbation
        grid_coords = np.linspace(0.06, 0.94, 4)
        xv, yv = np.meshgrid(grid_coords, grid_coords)
        all_grid = np.column_stack([xv.ravel(), yv.ravel()])  # 16 points
        all_grid += rng.normal(0, 0.036, all_grid.shape)
        all_grid = np.clip(all_grid, 0.02, 0.98)
        
        # Select best 14 out of 16 grid points
        for subset_idx in combinations(range(16), 14):
            subset = all_grid[list(subset_idx)]
            min_x2, _ = _compute_min_triangle_area_x2_and_triplet(subset)
            hull = _compute_convex_hull_area(subset)
            if hull > 1e-9:
                score = min_x2 / hull
                if score > best_initial_score:
                    best_initial_score = score
                    best_initial_points = subset.copy()
    
    points = best_initial_points
    
    # Hill climbing with priority moves and efficient evaluation
    current_min_x2, min_triplet = _compute_min_triangle_area_x2_and_triplet(points)
    hull_area = _compute_convex_hull_area(points)
    
    # Multi-stage hill climbing with decreasing step sizes, enhanced move set, and increased iterations
    # Added an additional stage with finer step sizes for late-stage polishing
    for stage, step_sizes in enumerate([[0.045, 0.035, 0.025], [0.030, 0.022, 0.016], [0.018, 0.012, 0.008], [0.010, 0.006, 0.004], [0.005, 0.003, 0.0015]]):
        # Increased iterations per stage for more thorough optimization
        iterations_per_stage = [700, 550, 420, 300, 200]
        for iteration in range(iterations_per_stage[stage]):
            improved = False
            
            # Priority: first move points involved in smallest triangle, then others
            point_order = list(range(n))
            if min_triplet:
                for idx in min_triplet:
                    if idx in point_order:
                        point_order.remove(idx)
                point_order = list(min_triplet) + point_order
            
            for i in point_order:
                orig_point = points[i].copy()
                # Fast check: if this point isn't in min_triplet, the current min
                # can only change if we create a smaller triangle with this point
                current_min_for_point = _compute_min_triangle_area_x2_given_point(points, i)
                if current_min_for_point > current_min_x2 + 1e-15 and min_triplet and i not in min_triplet:
                    continue  # Skip if this point can't improve the global minimum
                
                for step in step_sizes:
                    # Consolidated and deduplicated move set: all relevant direction ratios, intermediate angle moves, and diagonals
                    # Removed redundant patterns, added 1:10 ratio extremes, and ensured full coverage without duplication
                    for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                                   (-step, -step), (-step, step), (step, -step), (step, step),
                                   # 0.1:1 ratio (extreme slopes for broad boundary pushing)
                                   (-step*0.1, -step), (-step*0.1, step), (step*0.1, -step), (step*0.1, step),
                                   (-step, -step*0.1), (-step, step*0.1), (step, -step*0.1), (step, step*0.1),
                                   # 0.2:1 ratio
                                   (-step*0.2, -step), (-step*0.2, step), (step*0.2, -step), (step*0.2, step),
                                   (-step, -step*0.2), (-step, step*0.2), (step, -step*0.2), (step, step*0.2),
                                   # 0.3:1 ratio
                                   (-step*0.3, -step), (-step*0.3, step), (step*0.3, -step), (step*0.3, step),
                                   (-step, -step*0.3), (-step, step*0.3), (step, -step*0.3), (step, step*0.3),
                                   # 0.4:1 ratio
                                   (-step*0.4, -step), (-step*0.4, step), (step*0.4, -step), (step*0.4, step),
                                   (-step, -step*0.4), (-step, step*0.4), (step, -step*0.4), (step, step*0.4),
                                   # 0.5:1 ratio
                                   (-step*0.5, -step), (-step*0.5, step), (step*0.5, -step), (step*0.5, step),
                                   (-step, -step*0.5), (-step, step*0.5), (step, -step*0.5), (step, step*0.5),
                                   # 0.6:1 ratio
                                   (-step*0.6, -step), (-step*0.6, step), (step*0.6, -step), (step*0.6, step),
                                   (-step, -step*0.6), (-step, step*0.6), (step, -step*0.6), (step, step*0.6),
                                   # 0.7:1 ratio
                                   (-step*0.7, -step), (-step*0.7, step), (step*0.7, -step), (step*0.7, step),
                                   (-step, -step*0.7), (-step, step*0.7), (step, -step*0.7), (step, step*0.7),
                                   # 0.8:1 ratio
                                   (-step*0.8, -step), (-step*0.8, step), (step*0.8, -step), (step*0.8, step),
                                   (-step, -step*0.8), (-step, step*0.8), (step, -step*0.8), (step, step*0.8),
                                   # 0.9:1 ratio
                                   (-step*0.9, -step), (-step*0.9, step), (step*0.9, -step), (step*0.9, step),
                                   (-step, -step*0.9), (-step, step*0.9), (step, -step*0.9), (step, step*0.9),
                                   # Intermediate angle moves (e.g., 3:4, 1:1 slopes for plateau navigation)
                                   (-step*0.6, -step*0.8), (-step*0.8, -step*0.6),
                                   (step*0.6, -step*0.8), (step*0.8, -step*0.6),
                                   (step*0.8, step*0.6), (step*0.6, step*0.8),
                                   (-step*0.6, step*0.8), (-step*0.8, step*0.6),
                                   (-step*0.7, -step*0.7), (-step*0.7, step*0.7),
                                   (step*0.7, -step*0.7), (step*0.7, step*0.7)]:
                        new_point = np.clip(orig_point + [dx, dy], 0.012, 0.988)
                        points[i] = new_point
                        new_min_x2, new_trip = _compute_min_triangle_area_x2_and_triplet(points)
                        new_hull = _compute_convex_hull_area(points)
                        
                        if new_hull > 1e-9:
                            norm_new = new_min_x2 / new_hull
                            norm_current = current_min_x2 / hull_area if hull_area > 1e-9 else 0
                            # Tuned acceptance criteria: slightly increased thresholds for true improvement
                            # and broader plateau exploration in all stages to escape local maxima
                            stage_allowance = 2e-11 if stage < 2 else 5e-12 if stage == 2 else 1e-14
                            if norm_new >= norm_current - stage_allowance:
                                if norm_new > norm_current + 8e-13:
                                    # True improvement: accept and track
                                    current_min_x2 = new_min_x2
                                    hull_area = new_hull
                                    min_triplet = new_trip
                                    improved = True
                                    break
                                else:
                                    # Equal or near-equal (plateau/gentle downhill): accept but don't count as improved
                                    # This allows better exploration without signaling the outer loop to restart
                                    current_min_x2 = new_min_x2
                                    hull_area = new_hull
                                    min_triplet = new_trip
                                    orig_point = new_point.copy()
                            else:
                                points[i] = orig_point
                        else:
                            points[i] = orig_point
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break
    
    return points


# EVOLVE-BLOCK-END