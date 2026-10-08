import numpy as np
from itertools import combinations


def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _convex_hull_area(points):
    if len(points) < 3:
        return 0.0
    pts = points[np.lexsort(points.T[::-1])]
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    hull = lower[:-1] + upper[:-1]
    if len(hull) < 3:
        return 0.0
    area = 0.0
    for i in range(len(hull)):
        j = (i + 1) % len(hull)
        area += hull[i][0] * hull[j][1] - hull[j][0] * hull[i][1]
    return abs(area) / 2.0


def _normalize_to_unit_hull(points):
    hull_area = _convex_hull_area(points)
    if hull_area > 0:
        return points / np.sqrt(hull_area)
    return points


def _compute_min_triangle_area_and_triplet(points):
    n = len(points)
    min_area = float('inf')
    min_triplet = None
    for i, j, k in combinations(range(n), 3):
        area = 0.5 * abs(
            (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
            (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
        )
        if area < min_area:
            min_area = area
            min_triplet = (i, j, k)
            if min_area < 1e-15:
                return min_area, min_triplet
    return min_area, min_triplet


def _compute_min_triangle_area(points):
    n = len(points)
    min_area = float('inf')
    for i, j, k in combinations(range(n), 3):
        area = 0.5 * abs(
            (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
            (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
        )
        if area < min_area:
            min_area = area
            if min_area < 1e-15:
                return min_area
    return min_area


def _halton_sequence(num_points, base):
    seq = np.zeros(num_points)
    for i in range(num_points):
        f = 1.0
        r = 0.0
        idx = i
        while idx > 0:
            f /= base
            r += f * (idx % base)
            idx //= base
        seq[i] = r
    return seq


def heilbronn_convex14() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 14.

    Returns:
        points: np.ndarray of shape (14,2) with the x,y coordinates of the points.
    """
    n = 14
    rng = np.random.default_rng(seed=42)
    
    # Multi-start with diverse initial configurations
    num_starts = 4
    best_overall_min = 0.0
    best_overall_points = None
    
    for start_idx in range(num_starts):
        # Configuration 0: Regular hexagon boundary with optimized interior
        # Configuration 1: 8-point near-circular boundary + Halton interior
        # Configuration 2: Low-discrepancy full Halton sequence
        # Configuration 3: Grid-jittered initialization
        if start_idx == 0:
            # Hexagon (6 boundary) + 8 interior
            boundary_count = 6
            theta = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False)
            boundary = np.column_stack([
                np.cos(theta) * 0.47 + 0.5, 
                np.sin(theta) * 0.47 + 0.5
            ])
            interior_count = n - boundary_count
            interior = np.column_stack([
                _halton_sequence(interior_count, 2) * 0.7 + 0.15, 
                _halton_sequence(interior_count, 3) * 0.7 + 0.15
            ])
            points = np.vstack([boundary, interior])
        elif start_idx == 1:
            # 8-point boundary, slight ellipse for better spacing
            boundary_count = 8
            theta = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False)
            boundary = np.column_stack([
                np.cos(theta) * 0.465 + 0.5, 
                np.sin(theta) * 0.445 + 0.5
            ])
            interior_count = n - boundary_count
            interior = np.column_stack([
                _halton_sequence(interior_count, 2) * 0.65 + 0.175, 
                _halton_sequence(interior_count, 3) * 0.65 + 0.175
            ])
            points = np.vstack([boundary, interior])
        elif start_idx == 2:
            # Pure low-discrepancy Halton sequence
            points = np.column_stack([
                _halton_sequence(n, 2) * 0.96 + 0.02, 
                _halton_sequence(n, 3) * 0.96 + 0.02
            ])
        else:
            # Jittered grid initialization
            grid_size = int(np.ceil(np.sqrt(n)))
            x = np.linspace(0.12, 0.88, grid_size)
            y = np.linspace(0.12, 0.88, grid_size)
            xv, yv = np.meshgrid(x, y)
            grid_points = np.column_stack([xv.ravel(), yv.ravel()])
            selected_idx = rng.choice(len(grid_points), size=n, replace=False)
            points = grid_points[selected_idx] + rng.normal(0, 0.025, (n, 2))
        
        points = np.clip(points, 0.01, 0.99)
        best_min_area = _compute_min_triangle_area(points)
        best_points = points.copy()
        
        # Phase 1: Coordinate ascent with shrinking grid
        noise_scale = 0.075
        for iter_idx in range(120):
            improved = False
            for i in range(n):
                for dx in [-noise_scale, -noise_scale/2, noise_scale/2, noise_scale]:
                    for dy in [-noise_scale, -noise_scale/2, noise_scale/2, noise_scale]:
                        new_points = best_points.copy()
                        new_points[i] = np.clip(new_points[i] + [dx, dy], 0.01, 0.99)
                        current_min = _compute_min_triangle_area(new_points)
                        if current_min > best_min_area + 1e-12:
                            best_min_area = current_min
                            best_points = new_points.copy()
                            improved = True
            if not improved:
                noise_scale *= 0.7
                if noise_scale < 0.01:
                    break
        
        # Phase 2: Longer simulated annealing with occasional double perturbations
        T = 0.07
        alpha = 0.994
        iterations_phase2 = 4000
        current_points = best_points.copy()
        current_min = best_min_area
        
        for it in range(iterations_phase2):
            if rng.random() < 0.18:
                # Perturb two points occasionally for better escape from local optima
                idx1, idx2 = rng.choice(n, size=2, replace=False)
                delta1 = rng.normal(0, 0.028, 2)
                delta2 = rng.normal(0, 0.028, 2)
                new_points = current_points.copy()
                new_points[idx1] = np.clip(new_points[idx1] + delta1, 0.01, 0.99)
                new_points[idx2] = np.clip(new_points[idx2] + delta2, 0.01, 0.99)
            else:
                idx = rng.integers(0, n)
                delta = rng.normal(0, 0.032, 2)
                new_points = current_points.copy()
                new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
            
            new_min = _compute_min_triangle_area(new_points)
            
            if new_min > current_min:
                current_points = new_points
                current_min = new_min
                if new_min > best_min_area:
                    best_min_area = new_min
                    best_points = new_points.copy()
            else:
                diff = new_min - current_min
                prob = np.exp(diff / T) if T > 1e-6 else 0.0
                if rng.random() < prob:
                    current_points = new_points
                    current_min = new_min
            
            T *= alpha
        
        # Phase 3: Enhanced triplet-aware fine-grained hill climbing with more step sizes
        step_size = 0.025
        for fine_iter in range(150):
            improved = False
            _, min_triplet = _compute_min_triangle_area_and_triplet(best_points)
            
            # Focus on triplet forming the smallest triangle, occasionally check all
            indices_to_check = range(n) if fine_iter % 4 == 0 or min_triplet is None else min_triplet
            
            for i in indices_to_check:
                # Expanded set of step sizes for finer granularity
                for dx in [-step_size, -step_size*0.7, -step_size*0.4, -step_size*0.2, step_size*0.2, step_size*0.4, step_size*0.7, step_size]:
                    for dy in [-step_size, -step_size*0.7, -step_size*0.4, -step_size*0.2, step_size*0.2, step_size*0.4, step_size*0.7, step_size]:
                        if dx == 0 and dy == 0:
                            continue
                        new_points = best_points.copy()
                        new_points[i] = np.clip(new_points[i] + [dx, dy], 0.01, 0.99)
                        new_min = _compute_min_triangle_area(new_points)
                        if new_min > best_min_area + 1e-12:
                            best_min_area = new_min
                            best_points = new_points.copy()
                            improved = True
            if not improved:
                step_size *= 0.63
                if step_size < 0.0007:
                    break
        
        # Update multi-start best
        if best_min_area > best_overall_min:
            best_overall_min = best_min_area
            best_overall_points = best_points.copy()
    
    # Normalize to unit convex hull area
    return _normalize_to_unit_hull(best_overall_points)