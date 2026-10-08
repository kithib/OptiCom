import numpy as np
from itertools import combinations


def _min_triangle_area(points: np.ndarray) -> float:
    """Compute minimum triangle area from all triplets of points with early exit."""
    min_area = float('inf')
    for i, j, k in combinations(range(len(points)), 3):
        # Manual cross product avoids numpy function call overhead
        area = 0.5 * abs(
            (points[j, 0] - points[i, 0]) * (points[k, 1] - points[i, 1]) -
            (points[j, 1] - points[i, 1]) * (points[k, 0] - points[i, 0])
        )
        if area < min_area:
            min_area = area
            if min_area < 1e-15:
                return 0.0
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
    
    # Halton sequence for low-discrepancy initial points
    def halton_sequence(b):
        n_val, d = 0, 1
        while True:
            x = d - n_val
            if x == 1:
                n_val = 1
                d *= b
            else:
                y = d // b
                while x <= y:
                    y //= b
                n_val = (b + 1) * y - x
            yield n_val / d
    
    gen_x = halton_sequence(2)
    gen_y = halton_sequence(3)
    points = np.array([[next(gen_x), next(gen_y)] for _ in range(n)])
    points += rng.normal(0, 0.02, points.shape)
    points = np.clip(points, 0.01, 0.99)
    
    # Phase 1: Multi-step hill climbing with 8-direction perturbations
    # Trimmed step sizes and reduced iteration limit for efficiency
    step_sizes = [0.08, 0.04, 0.02, 0.015, 0.01, 0.0075, 0.005]
    for step in step_sizes:
        improved = True
        iterations = 0
        # Reduced iteration limit to improve efficiency
        while improved and iterations < 80:
            improved = False
            current_min = _min_triangle_area(points)
            
            for i in range(n):
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                               (-step, -step), (-step, step), (step, -step), (step, step)]:
                    new_point = np.clip(points[i] + [dx, dy], 0.01, 0.99)
                    new_points = points.copy()
                    new_points[i] = new_point
                    new_min = _min_triangle_area(new_points)
                    
                    if new_min > current_min + 1e-12:
                        points = new_points
                        current_min = new_min
                        improved = True
            iterations += 1
    
    # Phase 2: Combined targeted perturbation with best tracking and exhaustive evaluation
    no_improve_count = 0
    best_points = points.copy()
    best_min = _min_triangle_area(points)
    
    # Finer-grained step sizes for phase 2
    step_sizes_triplet = [0.02, 0.018, 0.015, 0.012, 0.01, 0.008, 0.006, 0.004, 0.003, 0.0025]
    
    # Reduced iteration count for phase 2 to improve efficiency
    for iter_idx in range(650):
        # Faster progressive step size reduction
        step_size = step_sizes_triplet[min(iter_idx // 70, len(step_sizes_triplet) - 1)]
        
        min_area = float('inf')
        worst_triplet = (0, 1, 2)
        # Also track second-worst triplet for more comprehensive targeting
        second_min_area = float('inf')
        second_worst_triplet = (0, 1, 3)
        
        for i, j, k in combinations(range(n), 3):
            a, b, c = points[i], points[j], points[k]
            area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0]))
            if area < min_area:
                second_min_area = min_area
                second_worst_triplet = worst_triplet
                min_area = area
                worst_triplet = (i, j, k)
            elif area < second_min_area:
                second_min_area = area
                second_worst_triplet = (i, j, k)
        
        # Strategy: Evaluate multiple candidates and select the best one
        triplet_improved = False
        best_candidate_point = None
        best_candidate_min = min_area
        
        # Build candidate set from both worst and second-worst triplets + random points
        worst_points = set(worst_triplet) | set(second_worst_triplet)
        candidates = list(worst_points) + rng.choice([p for p in range(n) if p not in worst_points], 
                                                       size=min(7, n-len(worst_points)), replace=False).tolist()
        # Reduced perturbation attempts to improve efficiency
        for p_idx in candidates:
            for _ in range(25):
                old_point = points[p_idx].copy()
                # Mix: uniform, angular, and offset perturbations
                r = rng.random()
                if r < 0.35:
                    points[p_idx] += rng.uniform(-step_size, step_size, 2)
                elif r < 0.65:
                    angle = rng.uniform(0, 2 * np.pi)
                    dist = rng.uniform(0, step_size * 1.5)
                    points[p_idx, 0] += dist * np.cos(angle)
                    points[p_idx, 1] += dist * np.sin(angle)
                else:
                    dx = rng.choice([-step_size, 0, step_size]) * rng.uniform(0.6, 1.4)
                    dy = rng.choice([-step_size, 0, step_size]) * rng.uniform(0.6, 1.4)
                    points[p_idx, 0] += dx
                    points[p_idx, 1] += dy
                points[p_idx] = np.clip(points[p_idx], 0.01, 0.99)
                
                new_min = _min_triangle_area(points)
                if new_min > best_candidate_min + 1e-12:
                    best_candidate_min = new_min
                    best_candidate_point = (p_idx, points[p_idx].copy())
                
                points[p_idx] = old_point
        
        # Apply the best move found (if any)
        if best_candidate_point is not None:
            points[best_candidate_point[0]] = best_candidate_point[1]
            triplet_improved = True
            
            # Update global best tracking
            if best_candidate_min > best_min + 1e-12:
                best_min = best_candidate_min
                best_points = points.copy()
                no_improve_count = 0
        
        if not triplet_improved:
            no_improve_count += 1
            if no_improve_count > 28:
                # Enhanced escape mechanism: larger and multi-point perturbations
                for _ in range(5):
                    p = rng.integers(0, n)
                    points[p] += rng.uniform(-step_size * 2.2, step_size * 2.2, 2)
                    points[p] = np.clip(points[p], 0.01, 0.99)
                step_size = max(step_size * 0.78, 0.0015)
                no_improve_count = 0
    
    # Extended and intensified polishing phase - trimmed for efficiency
    points = best_points.copy()
    polish_steps = [0.007, 0.005, 0.004, 0.003, 0.0022]
    for step in polish_steps:
        angles = np.linspace(0, 2*np.pi, 28, endpoint=False)
        directions = np.column_stack([np.cos(angles), np.sin(angles)]) * step
        for _ in range(3):
            current_min = _min_triangle_area(points)
            for i in range(n):
                for dx, dy in directions:
                    new_point = np.clip(points[i] + [dx, dy], 0.01, 0.99)
                    new_points = points.copy()
                    new_points[i] = new_point
                    new_min = _min_triangle_area(new_points)
                    if new_min > current_min + 1e-14:
                        points = new_points
                        current_min = new_min
                        if current_min > best_min + 1e-14:
                            best_points = points.copy()
                            best_min = current_min
    
    # Additional micro-polish phase - trimmed steps and directions for efficiency
    points = best_points.copy()
    polish_steps_final = [0.001, 0.0008, 0.0006, 0.0004]
    for dirs, step in zip([32, 40, 48, 56], polish_steps_final):
        angles = np.linspace(0, 2*np.pi, dirs, endpoint=False)
        directions = np.column_stack([np.cos(angles), np.sin(angles)]) * step
        for _ in range(2):
            current_min = _min_triangle_area(points)
            for i in range(n):
                for dx, dy in directions:
                    new_point = np.clip(points[i] + [dx, dy], 0.01, 0.99)
                    new_points = points.copy()
                    new_points[i] = new_point
                    new_min = _min_triangle_area(new_points)
                    if new_min > current_min + 1e-15:
                        points = new_points
                        current_min = new_min
                        if current_min > best_min + 1e-15:
                            best_points = points.copy()
                            best_min = current_min
        # Coordinate ascent for each axis
        current_min = _min_triangle_area(points)
        for i in range(n):
            for coord in [0, 1]:
                for delta in [step * 0.7, -step * 0.7]:
                    new_point = points[i].copy()
                    new_point[coord] = np.clip(new_point[coord] + delta, 0.01, 0.99)
                    new_points = points.copy()
                    new_points[i] = new_point
                    new_min = _min_triangle_area(new_points)
                    if new_min > current_min + 1e-15:
                        points = new_points
                        current_min = new_min
                        if current_min > best_min + 1e-15:
                            best_points = points.copy()
                            best_min = current_min
    
    return best_points.astype(np.float64)