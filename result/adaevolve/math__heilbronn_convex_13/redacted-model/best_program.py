import numpy as np
from itertools import combinations


def heilbronn_convex13() -> np.ndarray:
    """
    Construct an arrangement of n points on or inside a convex region in order to maximize the area of the
    smallest triangle formed by these points. Here n = 13.

    Returns:
        points: np.ndarray of shape (13,2) with the x,y coordinates of the points.
    """
    n = 13
    rng = np.random.default_rng(seed=42)
    
    # Precompute triangle indices ONCE for efficiency
    triangle_indices = list(combinations(range(n), 3))
    
    # Extract indices arrays once for vectorized computation
    idx_i = np.array([i for i, j, k in triangle_indices])
    idx_j = np.array([j for i, j, k in triangle_indices])
    idx_k = np.array([k for i, j, k in triangle_indices])
    
    # Vectorized area computation
    def compute_all_areas(pts):
        a = pts[idx_i]
        b = pts[idx_j]
        c = pts[idx_k]
        crosses = np.cross(b - a, c - a)
        return 0.5 * np.abs(crosses)
    
    def compute_min_area(pts):
        return np.min(compute_all_areas(pts))
    
    # Enhanced initialization: 7 boundary points + TWO interior rings pattern + explicit center
    points = []
    
    # Add 7 boundary points with 7-fold symmetry (slightly larger radius for larger hull area)
    num_boundary = 7
    for i in range(num_boundary):
        angle = 2 * np.pi * i / num_boundary + np.pi / (2 * num_boundary)
        x = 0.5 + 0.49 * np.cos(angle)
        y = 0.5 + 0.49 * np.sin(angle)
        points.append([x, y])
    
    # Explicit center point (helps break interior degeneracies)
    points.append([0.5, 0.5])
    
    # First inner ring: 2 points - tuned radius and offset angles
    remaining = n - len(points)
    ring1_radius = 0.225
    ring1_count = 2
    for i in range(ring1_count):
        angle = 2 * np.pi * i / ring1_count + np.pi / 4
        x = 0.5 + ring1_radius * np.cos(angle)
        y = 0.5 + ring1_radius * np.sin(angle)
        points.append([x, y])
    
    # Second outer ring: remaining points with phasing for maximum angular separation
    ring2_count = remaining - ring1_count
    ring2_radius = 0.335
    for i in range(ring2_count):
        angle = 2 * np.pi * i / ring2_count + np.pi / (2 * ring2_count)
        x = 0.5 + ring2_radius * np.cos(angle)
        y = 0.5 + ring2_radius * np.sin(angle)
        points.append([x, y])
    
    points = np.array(points[:n])
    
    # Normalize with slightly larger controlled perturbation
    points = 0.025 + 0.95 * (points - points.min(axis=0)) / (points.max(axis=0) - points.min(axis=0) + 1e-8)
    points = np.clip(points + rng.normal(0, 0.013, points.shape), 0.015, 0.985)
    
    # Enhanced repulsion refinement (three-stage with increasing thresholds and more iterations)
    for stage, thresh in enumerate([0.24, 0.27, 0.30]):
        for _ in range(14):
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    dist = np.linalg.norm(diff)
                    if dist < thresh:
                        repulsion_strength = 0.07 / (dist + 0.035)
                        disp = diff * repulsion_strength
                        points[i] += disp
                        points[j] -= disp
            points = np.clip(points, 0.01, 0.99)
    
    # Nesterov-accelerated gradient ascent - more iterations and fine-tuned parameters
    num_iterations = 480
    initial_step = 0.028
    momentum = 0.81
    eps = 1e-10
    
    best_points = points.copy()
    best_min_area = compute_min_area(points)
    
    velocity = np.zeros_like(points)
    
    for step in range(num_iterations):
        # Cosine annealing learning rate - slightly adjusted frequency
        step_size = initial_step * (1 - step / num_iterations) * (0.52 + 0.48 * np.cos(step * 0.05))
        
        # Nesterov look-ahead
        lookahead = np.clip(points + momentum * velocity, 0, 1)
        
        # Vectorized area computation
        areas = compute_all_areas(lookahead)
        min_area = areas.min()
        
        # Handle degeneracy with stronger perturbation
        if min_area < eps:
            jitter = rng.normal(0, 9e-4, points.shape)
            points = np.clip(points + jitter, 0, 1)
            velocity *= 0.48
            continue
        
        # Track best configuration found so far
        if min_area > best_min_area + 1e-12:
            best_min_area = min_area
            best_points = lookahead.copy()
        
        # Adaptive percentile with cosine annealing - wider range and higher lower bound
        pct = 17 + 14 * np.cos(step * 0.05)
        threshold = min(3.6 * min_area, np.percentile(areas, pct))
        small_tri_mask = areas < threshold
        
        grad = np.zeros_like(points)
        
        # Gradient computation with fine-tuned weight and area threshold inclusion
        for (i, j, k), area, is_small in zip(triangle_indices, areas, small_tri_mask):
            if is_small:
                a, b, c = lookahead[i], lookahead[j], lookahead[k]
                cross = np.cross(b - a, c - a)
                if abs(cross) < 1e-15:
                    continue
                sign = np.sign(cross)
                
                # Fine-tuned weighting: higher exponent for very small triangles, tighter regularization
                area_exponent = 2.65 if area < 1.6 * min_area else 2.42
                weight = 1.0 / ((area + 3.8e-5) ** area_exponent)
                
                grad[i] += sign * weight * np.array([c[1] - b[1], b[0] - c[0]])
                grad[j] += sign * weight * np.array([a[1] - c[1], c[0] - a[0]])
                grad[k] += sign * weight * np.array([b[1] - a[1], a[0] - b[0]])
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm > 1e-12:
            grad = grad / grad_norm
        
        # Nesterov momentum update with staged ramp-up (fine-tuned intervals and factors)
        if step < 30:
            momentum_factor = (step / 30.0) * momentum * 0.65
        elif step < 70:
            momentum_factor = 0.65 * momentum + ((step - 30) / 40.0) * 0.35 * momentum
        else:
            momentum_factor = momentum
        
        velocity = momentum_factor * velocity + step_size * grad
        points = np.clip(points + velocity, 0, 1)
        
        # Periodic repulsion (fine-tuned intervals and stronger repulsion to prevent overcrowding/collinearity)
        rep_interval = 5 if step < 180 else 11
        if step % rep_interval == 0:
            for i in range(n):
                for j in range(i + 1, n):
                    diff = points[i] - points[j]
                    dist_sq = np.sum(diff ** 2)
                    # Higher threshold and stronger repulsion with distance-dependent falloff
                    if dist_sq < 0.014:
                        repulsion = diff * (0.105 / (dist_sq + 0.0007))
                        points[i] += repulsion
                        points[j] -= repulsion
            points = np.clip(points, 0, 1)
    
    # Fine-grained local refinement of top candidates (enhanced final polishing step)
    candidate_points = [best_points.copy(), points.copy()]
    candidate_areas = [best_min_area, compute_min_area(points)]
    
    # Generate additional candidates with small perturbations at multiple scales
    for perturb_scale in [0.004, 0.007, 0.011, 0.016]:
        for _ in range(3):
            perturbed = best_points + rng.normal(0, perturb_scale, best_points.shape)
            perturbed = np.clip(perturbed, 0, 1)
            candidate_points.append(perturbed)
            candidate_areas.append(compute_min_area(perturbed))
    
    # Apply focused gradient refinement to top candidates with extended iterations
    sorted_idx = np.argsort(candidate_areas)[::-1][:5]
    for idx in sorted_idx:
        cand = candidate_points[idx].copy()
        cand_vel = np.zeros_like(cand)
        for fine_step in range(70):
            fine_step_size = 0.008 * (1 - fine_step / 70)
            fine_lookahead = np.clip(cand + 0.68 * cand_vel, 0, 1)
            fine_areas = compute_all_areas(fine_lookahead)
            fine_min = fine_areas.min()
            fine_thresh = min(3.3 * fine_min, np.percentile(fine_areas, 26))
            fine_mask = fine_areas < fine_thresh
            
            fine_grad = np.zeros_like(cand)
            for (i, j, k), area, is_small in zip(triangle_indices, fine_areas, fine_mask):
                if is_small:
                    a, b, c = fine_lookahead[i], fine_lookahead[j], fine_lookahead[k]
                    cross = np.cross(b - a, c - a)
                    if abs(cross) < 1e-15:
                        continue
                    sign = np.sign(cross)
                    fine_weight = 1.0 / ((area + 3.5e-5) ** 2.55)
                    fine_grad[i] += sign * fine_weight * np.array([c[1] - b[1], b[0] - c[0]])
                    fine_grad[j] += sign * fine_weight * np.array([a[1] - c[1], c[0] - a[0]])
                    fine_grad[k] += sign * fine_weight * np.array([b[1] - a[1], a[0] - b[0]])
            
            fine_grad_norm = np.linalg.norm(fine_grad)
            if fine_grad_norm > 1e-12:
                fine_grad = fine_grad / fine_grad_norm
            
            cand_vel = 0.62 * cand_vel + fine_step_size * fine_grad
            cand = np.clip(cand + cand_vel, 0, 1)
            
            cand_min = compute_min_area(cand)
            if cand_min > best_min_area + 1e-12:
                best_min_area = cand_min
                best_points = cand.copy()
    
    # Multi-point local hill climbing: systematically test coordinated moves for points in smallest triangles
    for _ in range(3):
        current_min = compute_min_area(best_points)
        areas = compute_all_areas(best_points)
        small_indices = np.where(areas < 1.5 * current_min)[0]
        involved_points = set()
        for idx in small_indices:
            involved_points.update(triangle_indices[idx])
        
        improved = False
        for p1 in involved_points:
            for p2 in list(involved_points) + [None]:
                for dx1 in [-0.012, -0.006, 0, 0.006, 0.012]:
                    for dy1 in [-0.012, -0.006, 0, 0.006, 0.012]:
                        if dx1 == 0 and dy1 == 0 and p2 is None:
                            continue
                        for dx2 in [-0.009, 0, 0.009] if p2 is not None else [0]:
                            for dy2 in [-0.009, 0, 0.009] if p2 is not None else [0]:
                                test_points = best_points.copy()
                                test_points[p1] = np.clip(test_points[p1] + [dx1, dy1], 0.01, 0.99)
                                if p2 is not None and p2 != p1:
                                    test_points[p2] = np.clip(test_points[p2] + [dx2, dy2], 0.01, 0.99)
                                test_min = compute_min_area(test_points)
                                if test_min > best_min_area + 1e-12:
                                    best_min_area = test_min
                                    best_points = test_points.copy()
                                    improved = True
        if not improved:
            break
    
    # Final verification and return
    final_current_min = compute_min_area(points)
    if final_current_min > best_min_area:
        return points
    return best_points