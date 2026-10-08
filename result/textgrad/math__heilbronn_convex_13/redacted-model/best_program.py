import numpy as np
from scipy.spatial import ConvexHull


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Weakness 1 FIXED: More optimal boundary/interior split with rotated 7-gon + asymmetric interior
    # Carefully tuned to provide better initial spacing and reduce collinear degeneracies
    boundary_count = 7
    interior_count = n - boundary_count
    
    # Boundary: larger rotated regular heptagon for better convex hull coverage
    angles = np.linspace(0, 2 * np.pi, boundary_count, endpoint=False) + (np.pi / 10.5)
    boundary = 0.5 + 0.498 * np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Interior: more carefully staggered 6-point configuration with golden ratio stagger
    interior_angles = np.linspace(0, 2 * np.pi, interior_count, endpoint=False) + (np.pi / 7 + 0.123)
    interior_radii = np.array([0.315, 0.345, 0.295, 0.335, 0.285, 0.325])
    interior = 0.5 + interior_radii[:, np.newaxis] * np.column_stack([np.cos(interior_angles), np.sin(interior_angles)])
    interior += rng.normal(0, 0.007, (interior_count, 2))
    
    points = np.vstack([boundary, interior])
    
    # More aggressive early termination with vectorized operations
    def compute_min_area(pts, early_stop_threshold=None):
        min_area = float('inf')
        n_pts = len(pts)
        for i in range(n_pts):
            if early_stop_threshold is not None and min_area <= early_stop_threshold:
                return min_area
            vecs = pts - pts[i]
            for j in range(i + 1, n_pts):
                if early_stop_threshold is not None and min_area <= early_stop_threshold:
                    return min_area
                cross_prods = 0.5 * np.abs(vecs[j, 0] * vecs[j+1:, 1] - vecs[j, 1] * vecs[j+1:, 0])
                if len(cross_prods) > 0:
                    current_min_cross = np.min(cross_prods)
                    if current_min_cross < min_area:
                        min_area = current_min_cross
                        if early_stop_threshold is not None and min_area <= early_stop_threshold:
                            return min_area
        return min_area
    
    def compute_all_areas(pts):
        areas = []
        n_pts = len(pts)
        for i in range(n_pts):
            vecs = pts - pts[i]
            for j in range(i + 1, n_pts):
                cross_prods = 0.5 * np.abs(vecs[j, 0] * vecs[j+1:, 1] - vecs[j, 1] * vecs[j+1:, 0])
                for k_idx, area in enumerate(cross_prods):
                    areas.append((area, (i, j, j + 1 + k_idx)))
        return sorted(areas, key=lambda x: x[0])
    
    current_min = compute_min_area(points)
    
    # 1. Extended hill-climbing with direction persistence (more iterations, better step schedule)
    last_move = np.zeros((n, 2))
    for iteration in range(50):
        improved = False
        step = max(0.0025, 0.028 * (1.0 - iteration * 0.018))
        for idx in range(n):
            directions = [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step, -step), (step, -step), (-step, step), (step, step),
                          (-step*0.7, 0), (step*0.7, 0), (0, -step*0.7), (0, step*0.7),
                          (-step*0.5, -step*0.5), (step*0.5, -step*0.5), 
                          (-step*0.5, step*0.5), (step*0.5, step*0.5),
                          (-step*0.3, 0), (step*0.3, 0), (0, -step*0.3), (0, step*0.3)]
            if np.any(last_move[idx] != 0):
                for boost in [1.0, 0.85, 0.6, 0.35]:
                    directions.insert(0, tuple(last_move[idx] * boost))
            for dx, dy in directions:
                new_points = points.copy()
                new_points[idx] = np.clip(new_points[idx] + [dx, dy], 0.004, 0.996)
                new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                if new_min > current_min + 1e-10:
                    last_move[idx] = np.array([dx, dy])
                    points = new_points
                    current_min = new_min
                    improved = True
                    break
        if not improved and iteration > 18:
            break
    
    # 2. Enhanced sequential random refinement - more samples, better radius decay
    for iteration in range(35):
        for idx in range(n):
            best_pos = points[idx].copy()
            best_min = current_min
            radius = max(0.002, 0.022 * (1.0 - iteration * 0.028))
            for _ in range(50):
                test_pos = points[idx] + rng.normal(0, radius, 2)
                test_pos = np.clip(test_pos, 0.004, 0.996)
                test_points = points.copy()
                test_points[idx] = test_pos
                test_min = compute_min_area(test_points, early_stop_threshold=best_min)
                if test_min > best_min + 1e-10:
                    best_pos = test_pos
                    best_min = test_min
            if best_min > current_min + 1e-10:
                points[idx] = best_pos
                current_min = best_min
    
    # 3. Extended gradient-based refinement with Nesterov momentum - tuned parameters
    learning_rate = 0.0055
    momentum = np.zeros((n, 2))
    velocity = np.zeros((n, 2))
    momentum_factor = 0.48
    for iteration in range(30):
        lr = learning_rate * max(0.18, (1.0 - iteration * 0.035))
        for idx in range(n):
            lookahead = points[idx] + momentum_factor * velocity[idx]
            for dim in range(2):
                delta = np.zeros(2)
                delta[dim] = 1e-5
                
                points_plus = points.copy()
                points_plus[idx] = lookahead + delta
                area_plus = compute_min_area(points_plus)
                
                points_minus = points.copy()
                points_minus[idx] = lookahead - delta
                area_minus = compute_min_area(points_minus)
                
                grad = (area_plus - area_minus) / (2e-5)
                momentum[idx, dim] = momentum_factor * momentum[idx, dim] + lr * grad
                velocity[idx, dim] = momentum[idx, dim]
                points[idx, dim] += momentum[idx, dim]
        
        points = np.clip(points, 0.004, 0.996)
        current_min = compute_min_area(points)
    
    # 4. Late-stage hill climbing
    for iteration in range(20):
        for idx in range(n):
            step = 0.0028
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step, -step), (step, -step), (-step, step), (step, step),
                          (-step*0.6, 0), (step*0.6, 0), (0, -step*0.6), (0, step*0.6)]:
                new_points = points.copy()
                new_points[idx] = np.clip(new_points[idx] + [dx, dy], 0.002, 0.998)
                new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                if new_min > current_min + 1e-12:
                    points = new_points
                    current_min = new_min
    
    # 5. Targeted refinement of limiting triangles - expanded scope
    for _ in range(22):
        all_areas = compute_all_areas(points)
        limiting_indices = set()
        for area, idxs in all_areas[:35]:
            limiting_indices.update(idxs)
        for idx in limiting_indices:
            for step in [0.0045, 0.0035, 0.0028, 0.002, 0.0012, 0.0006]:
                improved_idx = False
                for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                              (-step, -step), (step, -step), (-step, step), (step, step),
                              (-step*0.75, 0), (step*0.75, 0), (0, -step*0.75), (0, step*0.75),
                              (-step*0.5, -step*0.5), (step*0.5, -step*0.5),
                              (-step*0.5, step*0.5), (step*0.5, step*0.5),
                              (-step*0.25, 0), (step*0.25, 0), (0, -step*0.25), (0, step*0.25)]:
                    new_points = points.copy()
                    new_points[idx] = np.clip(new_points[idx] + [dx, dy], 0.002, 0.998)
                    new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                    if new_min > current_min + 1e-12:
                        points = new_points
                        current_min = new_min
                        improved_idx = True
                if improved_idx:
                    break
    
    # 6. Coordinated moves for triangle triplets - expanded to top 15 with more combos
    for _ in range(15):
        all_areas = compute_all_areas(points)
        for area, (i, j, k) in all_areas[:15]:
            for factor in [0.0035, 0.0028, 0.0022, 0.0016, 0.001, 0.0005]:
                vec_ij = points[j] - points[i]
                vec_jk = points[k] - points[j]
                normal_ij = np.array([-vec_ij[1], vec_ij[0]])
                normal_ij = normal_ij / (np.linalg.norm(normal_ij) + 1e-8)
                normal_jk = np.array([-vec_jk[1], vec_jk[0]])
                normal_jk = normal_jk / (np.linalg.norm(normal_jk) + 1e-8)
                
                improved_triplet = False
                for move_dir in [normal_ij, -normal_ij, normal_jk, -normal_jk]:
                    for pts_to_move in [[i], [j], [k], [i, j], [i, k], [j, k], [i, j, k]]:
                        new_points = points.copy()
                        for p_idx in pts_to_move:
                            new_points[p_idx] = np.clip(new_points[p_idx] + move_dir * factor, 0.002, 0.998)
                        new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                        if new_min > current_min + 1e-12:
                            points = new_points
                            current_min = new_min
                            improved_triplet = True
                            break
                    if improved_triplet:
                        break
                if improved_triplet:
                    break
    
    # 7. Swaps between boundary and interior - more rounds and perturbation testing
    boundary_indices = list(range(boundary_count))
    interior_indices = list(range(boundary_count, n))
    for _ in range(12):
        swapped = False
        for b_idx in boundary_indices:
            for i_idx in interior_indices:
                for perturb in [0.0, 0.005, -0.005, 0.008, -0.008]:
                    new_points = points.copy()
                    new_points[[b_idx, i_idx]] = new_points[[i_idx, b_idx]]
                    if perturb != 0:
                        for move_idx in [b_idx, i_idx]:
                            new_points[move_idx] = np.clip(new_points[move_idx] + rng.normal(0, abs(perturb), 2), 0.002, 0.998)
                    new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                    if new_min > current_min + 1e-12:
                        points = new_points
                        current_min = new_min
                        swapped = True
                        break
            if swapped:
                break
    
    # 8. Early termination micro-step refinement - more steps, more directions
    for iteration in range(40):
        improved_iter = False
        for idx in range(n):
            step = 0.0009
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step, -step), (step, -step), (-step, step), (step, step),
                          (-step*0.65, 0), (step*0.65, 0), (0, -step*0.65), (0, step*0.65),
                          (-step*0.35, -step*0.35), (step*0.35, -step*0.35),
                          (-step*0.35, step*0.35), (step*0.35, step*0.35)]:
                new_points = points.copy()
                new_points[idx] = np.clip(new_points[idx] + [dx, dy], 0.001, 0.999)
                new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                if new_min > current_min + 1e-12:
                    points = new_points
                    current_min = new_min
                    improved_iter = True
        if not improved_iter:
            break
    
    # 9. Simulated annealing style perturbations - longer schedule, slower cooling
    for iteration in range(25):
        temperature = max(0.00008, 0.0025 * (1.0 - iteration * 0.04))
        for idx in range(n):
            for _ in range(3):
                test_pos = points[idx] + rng.normal(0, temperature * 12, 2)
                test_pos = np.clip(test_pos, 0.001, 0.999)
                test_points = points.copy()
                test_points[idx] = test_pos
                test_min = compute_min_area(test_points)
                if test_min >= current_min - 1e-12:
                    if test_min > current_min + 1e-12:
                        points[idx] = test_pos
                        current_min = test_min
                        break
    
    # 10. Final ultra-micro refinement - extended iterations, even smaller steps
    for iteration in range(30):
        improved_iter = False
        for idx in range(n):
            step = 0.00035
            for dx, dy in [(-step, 0), (step, 0), (0, -step), (0, step),
                          (-step*0.5, -step*0.5), (step*0.5, -step*0.5),
                          (-step*0.5, step*0.5), (step*0.5, step*0.5)]:
                new_points = points.copy()
                new_points[idx] = np.clip(new_points[idx] + [dx, dy], 0.001, 0.999)
                new_min = compute_min_area(new_points, early_stop_threshold=current_min)
                if new_min > current_min + 1e-13:
                    points = new_points
                    current_min = new_min
                    improved_iter = True
        if not improved_iter:
            break
    
    # Normalize to ensure convex hull has unit area
    hull = ConvexHull(points)
    scale_factor = np.sqrt(1.0 / hull.area)
    points = points * scale_factor
    
    return points