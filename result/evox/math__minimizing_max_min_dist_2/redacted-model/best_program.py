# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    # Enhanced initial staggered hexagonal grid with optimal spacing ratio
    n = 4
    points = []
    hex_spacing = np.sqrt(3) / 2  # Optimal hexagonal packing vertical spacing
    for i in range(n):
        for j in range(n):
            x = j / (n - 1)
            y = i / (n - 1) * hex_spacing
            if i % 2 == 1:
                x += 0.5 / (n - 1)
            points.append([x, y])
    points = np.array(points)
    # Normalize initial spacing to fill [0,1] range optimally
    points[:, 1] = points[:, 1] / np.max(points[:, 1])
    points = np.clip(points, 0.0, 1.0)
    
    # Hyper-enhanced gradient ascent with full meta-evolved improvements:
    # Multi-phase cosine annealing, adaptive top-K pairs, nested momentum, adaptive repulsion,
    # multi-component boundary forces, diversified restart strategy with orthogonal/skew transforms,
    # gradient clipping with adaptive norms, aggressive fine-tuning with adaptive dithering,
    # multi-phase post-optimization, and direct ratio-maximization objective gradient.
    np.random.seed(42)
    num_restarts = 12
    base_iters = 3600
    best_ratio = -1.0
    best_points = points.copy()
    
    for restart in range(num_restarts + 1):
        if restart > 0:
            # Full restart diversity: graduated perturbation with multi-transformations
            if restart % 8 == 0:
                # Large perturbation with rotation/reflection/shear for escaping deep local optima
                theta = np.random.uniform(-np.pi/4.2, np.pi/4.2)
                rot_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                       [np.sin(theta), np.cos(theta)]])
                # Random reflection
                if np.random.rand() > 0.5:
                    rot_matrix[:, 0] *= -1
                # Add shear component
                shear = np.random.uniform(-0.3, 0.3)
                shear_matrix = np.array([[1.0, shear], [0.0, 1.0]])
                combined = rot_matrix @ shear_matrix
                points = (best_points - 0.5) @ combined + 0.5
                points += np.random.normal(0, 0.072, best_points.shape)
            elif restart % 7 == 0:
                # Skew transform perturbation
                a, b = np.random.uniform(-0.32, 0.32, 2)
                skew_matrix = np.array([[1.0, a], [b, 1.0]])
                points = (best_points - 0.5) @ skew_matrix + 0.5
                points += np.random.normal(0, 0.055, best_points.shape)
            elif restart % 6 == 0:
                points = best_points + np.random.normal(0, 0.038, best_points.shape)
            elif restart % 5 == 0:
                points = best_points + np.random.normal(0, 0.022, best_points.shape)
            elif restart % 4 == 0:
                points = best_points + np.random.normal(0, 0.013, best_points.shape)
            elif restart % 3 == 0:
                points = best_points + np.random.normal(0, 0.0075, best_points.shape)
            elif restart % 2 == 0:
                points = best_points + np.random.normal(0, 0.0042, best_points.shape)
            else:
                # Finest adaptive jitter with directional bias
                grad_jitter = np.random.normal(0, 0.0018, best_points.shape)
                points = best_points + grad_jitter
            points = np.clip(points, 0.0, 1.0)
        
        num_iters = base_iters + restart * 1200
        initial_step = 0.016
        repulsion_weight = 0.38
        boundary_weight = 0.0065
        ratio_weight = 9.2
        momentum_beta = 0.85
        second_momentum_beta = 0.92  # Nesterov-style second momentum
        velocity = np.zeros_like(points)
        velocity2 = np.zeros_like(points)
        
        for iter_num in range(num_iters):
            # Enhanced cosine annealing schedule with warm restart characteristics
            progress = iter_num / num_iters
            cycle_len = num_iters // 4
            cycle_progress = (iter_num % cycle_len) / cycle_len
            if iter_num < num_iters * 0.7:
                step_size = initial_step * (0.5 * (1.0 + np.cos(progress * np.pi * 0.92)))
            else:
                # Finer-grained annealing in late phase
                step_size = initial_step * 0.08 * (0.5 * (1.0 + np.cos(progress * np.pi * 3)))
            
            # Compute distance matrix
            diffs = points[:, np.newaxis] - points
            dists = np.linalg.norm(diffs, axis=2)
            np.fill_diagonal(dists, np.inf)
            
            # Find min and max distances
            dmin = np.min(dists)
            dmax = np.max(dists)
            current_ratio = dmin / dmax
            
            # Adaptive top-K closest pairs - increases focus as optimization progresses
            k_min = max(8, min(16, int(15 - progress * 9)))
            min_indices = []
            dists_copy = dists.copy()
            for _ in range(k_min):
                min_idx = np.unravel_index(np.argmin(dists_copy), dists_copy.shape)
                min_indices.append(min_idx)
                dists_copy[min_idx] = np.inf
                dists_copy[min_idx[::-1]] = np.inf
            
            # Adaptive top-K farthest pairs
            k_max = max(5, min(12, int(11 - progress * 7)))
            max_indices = []
            dists_copy_max = dists.copy()
            for _ in range(k_max):
                max_idx = np.unravel_index(np.argmax(dists_copy_max), dists_copy_max.shape)
                max_indices.append(max_idx)
                dists_copy_max[max_idx] = -np.inf
                dists_copy_max[max_idx[::-1]] = -np.inf
            
            gradient = np.zeros_like(points)
            
            # Push closest pairs apart with enhanced adaptive weighting and direct ratio gradient
            for i, j in min_indices:
                dir_vec = points[j] - points[i]
                dir_vec_norm = np.linalg.norm(dir_vec)
                if dir_vec_norm > 1e-8:
                    dir_vec = dir_vec / dir_vec_norm
                    ratio_gap = (1.0 - current_ratio) ** 1.7
                    weight = (1.0 + ratio_weight * ratio_gap * 2.4)
                    gradient[i] -= dir_vec * weight
                    gradient[j] += dir_vec * weight
            
            # Pull farthest pairs together with stronger decaying pull strength
            for k, l in max_indices:
                dir_vec2 = points[l] - points[k]
                dir_vec2_norm = np.linalg.norm(dir_vec2)
                if dir_vec2_norm > 1e-8:
                    dir_vec2 = dir_vec2 / dir_vec2_norm
                    pull_weight = 1.85 + 0.95 * (1.0 - progress)
                    gradient[k] += dir_vec2 * pull_weight
                    gradient[l] -= dir_vec2 * pull_weight
            
            # Adaptive all-pair repulsion with tuned distance exponent (2.46)
            adaptive_repulsion = repulsion_weight * (0.41 / (dmin + 0.058))
            for p in range(16):
                for q in range(p + 1, 16):
                    diff = points[q] - points[p]
                    dist = np.linalg.norm(diff)
                    if dist > 1e-6:
                        dist_factor = 1.0 / (dist ** 2.46 + 1e-6)
                        force = adaptive_repulsion * dist_factor
                        unit_vec = diff / dist
                        gradient[p] -= force * unit_vec
                        gradient[q] += force * unit_vec
            
            # Enhanced boundary repulsion with corner avoidance
            for p in range(16):
                x, y = points[p]
                # Multi-scale boundary repulsion
                force_x = boundary_weight * (1.0/(x + 0.008)**2.2 - 1.0/(1.0 - x + 0.008)**2.2)
                force_y = boundary_weight * (1.0/(y + 0.008)**2.2 - 1.0/(1.0 - y + 0.008)**2.2)
                gradient[p, 0] -= force_x
                gradient[p, 1] -= force_y
                # Corner repulsion - avoid tight corner packing
                corner_dist1 = np.sqrt(x**2 + y**2)
                corner_dist2 = np.sqrt((1-x)**2 + y**2)
                corner_dist3 = np.sqrt(x**2 + (1-y)**2)
                corner_dist4 = np.sqrt((1-x)**2 + (1-y)**2)
                corner_force = boundary_weight * 0.4
                if corner_dist1 < 0.3:
                    gradient[p, 0] += corner_force / (corner_dist1**2 + 0.01)
                    gradient[p, 1] += corner_force / (corner_dist1**2 + 0.01)
                if corner_dist2 < 0.3:
                    gradient[p, 0] -= corner_force / (corner_dist2**2 + 0.01)
                    gradient[p, 1] += corner_force / (corner_dist2**2 + 0.01)
                if corner_dist3 < 0.3:
                    gradient[p, 0] += corner_force / (corner_dist3**2 + 0.01)
                    gradient[p, 1] -= corner_force / (corner_dist3**2 + 0.01)
                if corner_dist4 < 0.3:
                    gradient[p, 0] -= corner_force / (corner_dist4**2 + 0.01)
                    gradient[p, 1] -= corner_force / (corner_dist4**2 + 0.01)
            
            # Enhanced centering with variance balancing
            center = np.mean(points, axis=0)
            center_offset = center - 0.5
            gradient[:, 0] -= center_offset[0] * 0.018
            gradient[:, 1] -= center_offset[1] * 0.018
            # Spread balancing - encourage equal variance in both axes
            var_x = np.var(points[:, 0])
            var_y = np.var(points[:, 1])
            var_diff = var_x - var_y
            gradient[:, 0] -= (points[:, 0] - center[0]) * var_diff * 0.008
            gradient[:, 1] += (points[:, 1] - center[1]) * var_diff * 0.008
            
            # Enhanced gradient clipping with adaptive norm
            clip_val = 0.42 + 1.1 * (1.0 - progress)
            grad_norm = np.linalg.norm(gradient)
            if grad_norm > clip_val * 16:
                gradient = gradient * (clip_val * 16 / grad_norm)
            gradient = np.clip(gradient, -clip_val, clip_val)
            
            # Apply Nesterov-style dual momentum with acceleration
            velocity = momentum_beta * velocity + (1.0 - momentum_beta) * gradient
            velocity2 = second_momentum_beta * velocity2 + (1.0 - second_momentum_beta) * velocity
            points += step_size * (0.4 * velocity + 0.6 * velocity2)
            points = np.clip(points, 0.0, 1.0)
        
        # Track best configuration across restarts
        final_diffs = points[:, np.newaxis] - points
        final_dists = np.linalg.norm(final_diffs, axis=2)
        np.fill_diagonal(final_dists, np.inf)
        final_ratio = np.min(final_dists) / np.max(final_dists)
        if final_ratio > best_ratio:
            best_ratio = final_ratio
            best_points = points.copy()
    
    # Enhanced secondary fine-tuning with plateau detection and escape
    points = best_points.copy()
    fine_tune_iters = 3800
    velocity = np.zeros_like(points)
    velocity2 = np.zeros_like(points)
    prev_ratio = -1
    plateau_count = 0
    velocity_decay = 0.99978
    
    for iter_num in range(fine_tune_iters):
        progress = iter_num / fine_tune_iters
        step_size = 0.0062 * (0.5 * (1.0 + np.cos(progress * np.pi)))
        
        diffs = points[:, np.newaxis] - points
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dmin = np.min(dists)
        dmax = np.max(dists)
        current_ratio = dmin / dmax
        
        # Sensitive plateau detection
        if abs(current_ratio - prev_ratio) < 7e-8:
            plateau_count += 1
        else:
            plateau_count = 0
        prev_ratio = current_ratio
        
        if plateau_count > 14:
            # Enhanced adaptive dithering with multi-scale
            dither_mag = min(0.0019 + plateau_count * 0.00009, 0.0048)
            points += np.random.normal(0, dither_mag, points.shape)
            # Also apply slight rotation
            if plateau_count > 40:
                theta = np.random.uniform(-0.12, 0.12)
                rot = np.array([[np.cos(theta), -np.sin(theta)], 
                               [np.sin(theta), np.cos(theta)]])
                center = np.mean(points, axis=0)
                points = (points - center) @ rot + center
            points = np.clip(points, 0.0, 1.0)
            velocity *= 0.78
            velocity2 *= 0.78
            if plateau_count > 90:
                plateau_count = 0
            continue
        
        # Adaptive close pairs during fine-tune
        dmin_tight = np.min(dists)
        close_pairs = np.where((dists < dmin_tight * 1.12) & (dists > 1e-8))
        min_indices = list(zip(close_pairs[0], close_pairs[1]))
        
        # Focus on top-7 farthest pairs
        k_max_ft = 7
        max_indices = []
        dists_copy_max = dists.copy()
        for _ in range(k_max_ft):
            max_idx = np.unravel_index(np.argmax(dists_copy_max), dists_copy_max.shape)
            max_indices.append(max_idx)
            dists_copy_max[max_idx] = -np.inf
            dists_copy_max[max_idx[::-1]] = -np.inf
        
        gradient = np.zeros_like(points)
        
        # Extremely aggressive push for minimum distance pairs
        for i, j in min_indices:
            dir_vec = points[j] - points[i]
            dir_vec_norm = np.linalg.norm(dir_vec)
            if dir_vec_norm > 1e-8:
                dir_vec = dir_vec / dir_vec_norm
                ratio_gap = (1.0 - current_ratio) ** 2.8
                weight = (1.0 + 12.5 * ratio_gap * 2.9)
                gradient[i] -= dir_vec * weight
                gradient[j] += dir_vec * weight
        
        for k, l in max_indices:
            dir_vec2 = points[l] - points[k]
            dir_vec2_norm = np.linalg.norm(dir_vec2)
            if dir_vec2_norm > 1e-8:
                dir_vec2 = dir_vec2 / dir_vec2_norm
                pull_weight = 2.05 + 0.75 * (1.0 - progress)
                gradient[k] += dir_vec2 * pull_weight
                gradient[l] -= dir_vec2 * pull_weight
        
        # Tuned all-pair repulsion for fine-tuning with optimized exponent
        adaptive_repulsion = 0.29 * (0.46 / (dmin + 0.052))
        for p in range(16):
            for q in range(p + 1, 16):
                diff = points[q] - points[p]
                dist = np.linalg.norm(diff)
                if dist > 1e-6:
                    force = adaptive_repulsion / (dist ** 2.53 + 1e-6)
                    unit_vec = diff / dist
                    gradient[p] -= force * unit_vec
                    gradient[q] += force * unit_vec
        
        velocity = velocity * velocity_decay * 0.52 + 0.48 * gradient
        velocity2 = velocity2 * 0.62 + 0.38 * velocity
        points += step_size * (0.35 * velocity + 0.65 * velocity2)
        points = np.clip(points, 0.0, 1.0)
    
    # Check if fine-tuned version is better
    final_diffs = points[:, np.newaxis] - points
    final_dists = np.linalg.norm(final_diffs, axis=2)
    np.fill_diagonal(final_dists, np.inf)
    final_ratio = np.min(final_dists) / np.max(final_dists)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        best_points = points.copy()
    
    # Aggressive tertiary dmin-boosting phase
    points = best_points.copy()
    tertiary_iters = 2800
    velocity = np.zeros_like(points)
    np.random.seed(44)
    
    for iter_num in range(tertiary_iters):
        progress = iter_num / tertiary_iters
        step_size = 0.0042 * (0.5 * (1.0 + np.cos(progress * np.pi)))
        
        diffs = points[:, np.newaxis] - points
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dmin = np.min(dists)
        dmax = np.max(dists)
        current_ratio = dmin / dmax
        
        # Focus on all minimum-like pairs - very tight threshold
        close_pairs = np.where((dists < dmin * 1.06) & (dists > 1e-8))
        min_indices = list(zip(close_pairs[0], close_pairs[1]))
        
        # Only top-4 farthest pairs for gentle dmax compression
        far_pairs = np.where((dists > dmax * 0.96) & (dists > 1e-8))
        max_indices = list(zip(far_pairs[0], far_pairs[1]))[:4]
        
        gradient = np.zeros_like(points)
        
        # Maximum push for closest pairs - primary focus
        for i, j in min_indices:
            dir_vec = points[j] - points[i]
            dir_vec_norm = np.linalg.norm(dir_vec)
            if dir_vec_norm > 1e-8:
                dir_vec = dir_vec / dir_vec_norm
                dist_factor = (dmin / (dir_vec_norm + 1e-8)) ** 4.2
                weight = 5.2 * dist_factor * (1.0 + 3.5 * (1.0 - current_ratio))
                gradient[i] -= dir_vec * weight
                gradient[j] += dir_vec * weight
        
        # Gentle pull for far pairs
        for k, l in max_indices:
            dir_vec2 = points[l] - points[k]
            dir_vec2_norm = np.linalg.norm(dir_vec2)
            if dir_vec2_norm > 1e-8:
                dir_vec2 = dir_vec2 / dir_vec2_norm
                gradient[k] += dir_vec2 * 1.2
                gradient[l] -= dir_vec2 * 1.2
        
        # Light stabilization repulsion
        for p in range(16):
            for q in range(p + 1, 16):
                diff = points[q] - points[p]
                dist = np.linalg.norm(diff)
                if dist > 1e-6 and dist < dmin * 1.7:
                    force = 0.048 / (dist ** 2.42 + 1e-6)
                    unit_vec = diff / dist
                    gradient[p] -= force * unit_vec
                    gradient[q] += force * unit_vec
        
        velocity = 0.44 * velocity + 0.56 * gradient
        points += step_size * velocity
        points = np.clip(points, 0.0, 1.0)
    
    # Check if tertiary improved
    final_diffs = points[:, np.newaxis] - points
    final_dists = np.linalg.norm(final_diffs, axis=2)
    np.fill_diagonal(final_dists, np.inf)
    final_ratio = np.min(final_dists) / np.max(final_dists)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        best_points = points.copy()
    
    # Quaternary phase: extreme targeted dmin maximization
    points = best_points.copy()
    quaternary_iters = 2200
    velocity = np.zeros_like(points)
    np.random.seed(45)
    
    for iter_num in range(quaternary_iters):
        progress = iter_num / quaternary_iters
        step_size = 0.0028 * (0.5 * (1.0 + np.cos(progress * np.pi)))
        
        diffs = points[:, np.newaxis] - points
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        dmin = np.min(dists)
        current_ratio = dmin / np.max(dists)
        
        # Only the actual minimum distance pairs - hyper focus
        min_pairs = np.where((dists < dmin * 1.025) & (dists > 1e-8))
        min_indices = list(zip(min_pairs[0], min_pairs[1]))
        
        gradient = np.zeros_like(points)
        
        # Max push for absolute minimum pairs
        for i, j in min_indices:
            dir_vec = points[j] - points[i]
            dir_vec_norm = np.linalg.norm(dir_vec)
            if dir_vec_norm > 1e-8:
                dir_vec = dir_vec / dir_vec_norm
                weight = 6.8 + 5.2 * (1.0 - current_ratio) ** 1.3
                gradient[i] -= dir_vec * weight
                gradient[j] += dir_vec * weight
        
        # Very light neighbor stabilization
        for p in range(16):
            for q in range(p + 1, 16):
                diff = points[q] - points[p]
                dist = np.linalg.norm(diff)
                if dist > 1e-6 and dist < dmin * 1.5:
                    force = 0.022 / (dist ** 2.18 + 1e-6)
                    unit_vec = diff / dist
                    gradient[p] -= force * unit_vec
                    gradient[q] += force * unit_vec
        
        velocity = 0.38 * velocity + 0.62 * gradient
        points += step_size * velocity
        points = np.clip(points, 0.0, 1.0)
    
    # Check if quaternary improved
    final_diffs = points[:, np.newaxis] - points
    final_dists = np.linalg.norm(final_diffs, axis=2)
    np.fill_diagonal(final_dists, np.inf)
    final_ratio = np.min(final_dists) / np.max(final_dists)
    if final_ratio > best_ratio:
        best_ratio = final_ratio
        best_points = points.copy()
    
    points = best_points
    # Final optimal normalization
    min_coords = points.min(axis=0, keepdims=True)
    max_coords = points.max(axis=0, keepdims=True)
    points = (points - min_coords) / (max_coords - min_coords + 1e-8)
    
    return points


# EVOLVE-BLOCK-END