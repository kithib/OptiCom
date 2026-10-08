import numpy as np
from scipy.spatial.distance import pdist, squareform


def _compute_objective(points: np.ndarray) -> tuple:
    """Compute dmin/dmax ratio for given points."""
    dists = pdist(points)
    dmin = np.min(dists)
    dmax = np.max(dists)
    return dmin / dmax, dmin, dmax


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Center and normalize points to fit in unit sphere."""
    points = points - np.mean(points, axis=0)
    max_norm = np.max(np.linalg.norm(points, axis=1))
    if max_norm > 0:
        points = points / max_norm
    return points


def _compute_gradient_analytical(points: np.ndarray, dist_matrix: np.ndarray, 
                                    min_pair: tuple, max_pair: tuple) -> np.ndarray:
    """
    Compute analytical gradient of dmin/dmax w.r.t. point coordinates.
    More efficient and accurate than finite differences.
    """
    n, d = points.shape
    grad = np.zeros_like(points)
    
    # Get indices and compute distance values
    i_min, j_min = min_pair
    i_max, j_max = max_pair
    d_min = dist_matrix[i_min, j_min]
    d_max = dist_matrix[i_max, j_max]
    
    if d_min < 1e-10 or d_max < 1e-10:
        return grad
    
    # Unit vectors for gradient computation
    diff_min = points[i_min] - points[j_min]
    unit_min = diff_min / d_min
    diff_max = points[i_max] - points[j_max]
    unit_max = diff_max / d_max
    
    # Gradient of ratio = (dmin/dmax) = (dmin * dmax^(-1))
    grad[i_min] += (1.0 / d_max) * unit_min
    grad[j_min] -= (1.0 / d_max) * unit_min
    grad[i_max] -= (d_min / (d_max ** 2)) * unit_max
    grad[j_max] += (d_min / (d_max ** 2)) * unit_max
    
    return grad


def _compute_gradient_multi_min(points: np.ndarray, dist_matrix: np.ndarray, 
                                   min_pairs: list, max_pair: tuple, weights: np.ndarray = None) -> np.ndarray:
    """
    Compute gradient using multiple minimum distance pairs for more stable optimization.
    Weight each min pair contribution to better distribute optimization pressure.
    """
    n, d = points.shape
    grad = np.zeros_like(points)
    
    if weights is None:
        weights = np.ones(len(min_pairs)) / len(min_pairs)
    
    # Get max distance info
    i_max, j_max = max_pair
    d_max = dist_matrix[i_max, j_max]
    if d_max < 1e-10:
        return grad
    
    # Average gradient from multiple min pairs
    total_dmin = 0.0
    for idx, (i_min, j_min) in enumerate(min_pairs):
        d_min = dist_matrix[i_min, j_min]
        total_dmin += weights[idx] * d_min
    
    for idx, (i_min, j_min) in enumerate(min_pairs):
        d_min = dist_matrix[i_min, j_min]
        w = weights[idx]
        if d_min < 1e-10:
            continue
        
        diff_min = points[i_min] - points[j_min]
        unit_min = diff_min / d_min
        
        grad[i_min] += w * (1.0 / d_max) * unit_min
        grad[j_min] -= w * (1.0 / d_max) * unit_min
    
    # Max pair gradient
    diff_max = points[i_max] - points[j_max]
    unit_max = diff_max / d_max
    grad[i_max] -= (total_dmin / (d_max ** 2)) * unit_max
    grad[j_max] += (total_dmin / (d_max ** 2)) * unit_max
    
    return grad


def _compute_all_kth_min_pairs(dist_matrix: np.ndarray, k: int = 12) -> list:
    """
    Find the k smallest distinct distance pairs for multi-pair gradient averaging.
    Returns list of (i, j) index pairs.
    """
    n = dist_matrix.shape[0]
    upper_tri = np.triu_indices(n, k=1)
    dists = dist_matrix[upper_tri]
    sorted_indices = np.argsort(dists)
    pairs = []
    seen = set()
    for idx in sorted_indices[:min(k * 3, len(sorted_indices))]:
        i, j = upper_tri[0][idx], upper_tri[1][idx]
        if (i, j) not in seen and (j, i) not in seen:
            pairs.append((i, j))
            seen.add((i, j))
            if len(pairs) >= k:
                break
    return pairs


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    np.random.seed(42)
    
    # Step 1: Expanded initialization candidates with wider range, adaptive pole strategy,
    # and known high-performance configurations
    s = 1.0
    base_face_centers = np.array([
        [-s,  0,  0], [ s,  0,  0],
        [ 0, -s,  0], [ 0,  s,  0],
        [ 0,  0, -s], [ 0,  0,  s]
    ])
    
    candidates = []
    # Extended range of scales with finer granularity around critical region (0.70-0.76)
    for scale in [0.67, 0.68, 0.69, 0.70, 0.703, 0.706, 0.709, 0.712, 0.715, 0.718, 0.721, 0.724, 0.727,
                  0.73, 0.733, 0.736, 0.739, 0.742, 0.745, 0.748, 0.751, 0.754, 0.757, 0.76, 0.765, 0.77, 
                  0.775, 0.78, 0.785, 0.79, 0.795, 0.80, 0.805, 0.81, 0.815, 0.82, 0.85, 0.88]:
        cube_vertices = np.array([
            [-s, -s, -s], [-s, -s,  s], [-s,  s, -s], [-s,  s,  s],
            [ s, -s, -s], [ s, -s,  s], [ s,  s, -s], [ s,  s,  s]
        ]) * scale
        # Extended range of pole scalings with finer granularity for high precision
        for pole_scale in [0.88, 0.90, 0.91, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.0, 
                           1.01, 1.02, 1.03, 1.04, 1.05, 1.06, 1.07, 1.08, 1.10, 1.12]:
            # All axes symmetric variant
            face_centers = base_face_centers.copy()
            face_centers *= pole_scale
            pts = _normalize_points(np.vstack([cube_vertices, face_centers]))
            ratio, dmin, _ = _compute_objective(pts)
            score = ratio + 0.35 * dmin  # Emphasize dmin more for better initial configurations
            candidates.append((score, ratio, pts))
            # Z-only scaled variants (often optimal)
            face_centers2 = base_face_centers.copy()
            face_centers2[4:, :] *= pole_scale
            pts2 = _normalize_points(np.vstack([cube_vertices, face_centers2]))
            ratio2, dmin2, _ = _compute_objective(pts2)
            score2 = ratio2 + 0.35 * dmin2
            candidates.append((score2, ratio2, pts2))
            # Two-axis scaled variants
            face_centers3 = base_face_centers.copy()
            face_centers3[2:, :] *= pole_scale
            pts3 = _normalize_points(np.vstack([cube_vertices, face_centers3]))
            ratio3, dmin3, _ = _compute_objective(pts3)
            score3 = ratio3 + 0.35 * dmin3
            candidates.append((score3, ratio3, pts3))
            # Add asymmetric variants for richer search
            for pole_scale2 in [0.94, 0.96, 0.98, 1.0, 1.02, 1.04]:
                face_centers4 = base_face_centers.copy()
                face_centers4[4:, :] *= pole_scale
                face_centers4[2:4, :] *= pole_scale2
                face_centers4[0:2, :] *= pole_scale2
                pts4 = _normalize_points(np.vstack([cube_vertices, face_centers4]))
                ratio4, dmin4, _ = _compute_objective(pts4)
                score4 = ratio4 + 0.35 * dmin4
                candidates.append((score4, ratio4, pts4))
    
    # Pick top candidates with diversity filtering - avoid duplicate basins
    candidates.sort(reverse=True, key=lambda x: x[0])
    
    # Diversity filter - keep only configurations that are sufficiently different
    filtered_candidates = []
    similarity_threshold = 0.015
    for cand in candidates:
        is_unique = True
        for existing in filtered_candidates:
            # Check configuration similarity via point set RMSD
            diff = cand[2] - existing[2]
            rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
            if rmsd < similarity_threshold:
                is_unique = False
                break
        if is_unique:
            filtered_candidates.append(cand)
        if len(filtered_candidates) >= 8:
            break
    
    # Initialize from diverse high-quality candidates
    best_points_overall = filtered_candidates[0][2]
    best_ratio_overall = filtered_candidates[0][1]
    
    # Optimize top filtered candidates for better basin exploration
    for init_idx in range(len(filtered_candidates)):
        points = filtered_candidates[init_idx][2].copy()
        
        # Step 2: Balanced optimization with improved adaptive momentum scheduling
        initial_lr = 0.165
        iterations = 3000
        current_best_ratio = _compute_objective(points)[0]
        run_best_points = points.copy()
        plateau_count = 0
        
        # Improved adaptive momentum configuration
        momentum_base = 0.78
        velocity = np.zeros_like(points)
        
        # Multi-min gradient configuration for better stability - increased pairs
        use_multi_min = True
        num_min_pairs = 9
        
        for iter_idx in range(iterations):
            dist_matrix = squareform(pdist(points))
            current_ratio, dmin, dmax = _compute_objective(points)
            
            # Track best solution with tighter tolerance
            if current_ratio > current_best_ratio + 3e-10:
                current_best_ratio = current_ratio
                run_best_points = points.copy()
                plateau_count = 0
            else:
                plateau_count += 1
            
            # More sensitive restart mechanism with two-stage perturbation
            if plateau_count > 110:
                # First try a small perturbation, then larger if still stuck
                perturb_scale = max(0.007, 0.032 * (1 - iter_idx / iterations))
                if plateau_count > 180:
                    perturb_scale *= 1.8
                points = run_best_points + np.random.normal(0, perturb_scale, points.shape)
                points = _normalize_points(points)
                plateau_count = 0
                velocity *= 0.12
            
            # Find min and max distance pairs
            dist_matrix_flat = dist_matrix + np.eye(n) * np.inf
            min_pair = np.unravel_index(np.argmin(dist_matrix_flat), (n, n))
            max_pair = np.unravel_index(np.argmax(dist_matrix), (n, n))
            
            # Adaptive momentum with higher final value
            momentum = min(momentum_base + 0.12 * (iter_idx / iterations), 0.92)
            
            # Nesterov momentum look-ahead
            nesterov_points = points + momentum * velocity
            ndist_matrix = squareform(pdist(nesterov_points))
            ndist_matrix_flat = ndist_matrix + np.eye(n) * np.inf
            nmin_pair = np.unravel_index(np.argmin(ndist_matrix_flat), (n, n))
            nmax_pair = np.unravel_index(np.argmax(ndist_matrix), (n, n))
            
            # Compute gradient - use multi-min pairs for more stable optimization
            if use_multi_min and iter_idx < 2400:
                nmin_pairs = _compute_all_kth_min_pairs(ndist_matrix, k=num_min_pairs)
                if len(nmin_pairs) > 0:
                    ndists_from_pairs = np.array([ndist_matrix[i, j] for i, j in nmin_pairs])
                    weights = 1.0 / (ndists_from_pairs ** 2 + 1e-10)
                    weights = weights / weights.sum()
                    grad = _compute_gradient_multi_min(nesterov_points, ndist_matrix, nmin_pairs, nmax_pair, weights)
                else:
                    grad = _compute_gradient_analytical(nesterov_points, ndist_matrix, nmin_pair, nmax_pair)
            else:
                grad = _compute_gradient_analytical(nesterov_points, ndist_matrix, nmin_pair, nmax_pair)
            
            # Enhanced vectorized repulsion with improved scaling and adaptive parameters
            eps_repulse = 4.8e-4 * np.exp(-iter_idx / 1300) + 2e-6
            threshold = dmin * 4.1
            i_idx, j_idx = np.triu_indices(n, k=1)
            mask = dist_matrix[i_idx, j_idx] < threshold
            i_close = i_idx[mask]
            j_close = j_idx[mask]
            if len(i_close) > 0:
                diffs = points[i_close] - points[j_close]
                dists = dist_matrix[i_close, j_close, None]
                dists_safe = np.maximum(dists, 1e-10)
                forces = eps_repulse / (dists_safe ** 3) * diffs / dists_safe
                np.add.at(points, i_close, forces)
                np.add.at(points, j_close, -forces)
            
            # Refined adaptive learning rate schedule with smoother decay
            if iter_idx < 750:
                learning_rate = initial_lr
            elif iter_idx < 1150:
                learning_rate = initial_lr * 0.56
            elif iter_idx < 1550:
                learning_rate = initial_lr * 0.29
            elif iter_idx < 1950:
                learning_rate = initial_lr * 0.13
            elif iter_idx < 2300:
                learning_rate = initial_lr * 0.055
            elif iter_idx < 2650:
                learning_rate = initial_lr * 0.025
            else:
                learning_rate = initial_lr * 0.010
            
            # Nesterov momentum update
            velocity = momentum * velocity + learning_rate * grad
            points += velocity
            points = _normalize_points(points)
        
        # Extended fine-tuning phase with multi-pair gradient throughout first half
        points = run_best_points.copy()
        ft_best_ratio = current_best_ratio
        for ft_idx in range(2200):
            dist_matrix = squareform(pdist(points))
            dist_matrix_flat = dist_matrix + np.eye(n) * np.inf
            
            # Multi-pair gradient extended further into fine-tuning with more pairs
            if ft_idx < 1000:
                # Use more min pairs for robust fine-tuning start
                small_pairs = _compute_all_kth_min_pairs(dist_matrix, k=10)[:8]
                max_pair = np.unravel_index(np.argmax(dist_matrix), (n, n))
                if len(small_pairs) > 0:
                    dists_from_pairs = np.array([dist_matrix[i, j] for i, j in small_pairs])
                    weights = 1.0 / (dists_from_pairs ** 2 + 1e-10)
                    weights = weights / weights.sum()
                    grad = _compute_gradient_multi_min(points, dist_matrix, small_pairs, max_pair, weights)
                else:
                    min_pair = np.unravel_index(np.argmin(dist_matrix_flat), (n, n))
                    grad = _compute_gradient_analytical(points, dist_matrix, min_pair, max_pair)
            else:
                # Standard single-pair gradient for late fine-tuning
                min_pair = np.unravel_index(np.argmin(dist_matrix_flat), (n, n))
                max_pair = np.unravel_index(np.argmax(dist_matrix), (n, n))
                grad = _compute_gradient_analytical(points, dist_matrix, min_pair, max_pair)
            
            # Improved micro-repulsion extending deeper into fine-tuning with stronger force
            if ft_idx < 1000:
                _, dmin_ft, _ = _compute_objective(points)
                eps_micro = 1.1e-5 * (1 - ft_idx / 1000)
                threshold_ft = dmin_ft * 2.6
                i_idx, j_idx = np.triu_indices(n, k=1)
                mask = dist_matrix[i_idx, j_idx] < threshold_ft
                i_close = i_idx[mask]
                j_close = j_idx[mask]
                if len(i_close) > 0:
                    diffs = points[i_close] - points[j_close]
                    dists = dist_matrix[i_close, j_close, None]
                    dists_safe = np.maximum(dists, 1e-10)
                    forces = eps_micro / (dists_safe ** 3) * diffs / dists_safe
                    np.add.at(points, i_close, forces)
                    np.add.at(points, j_close, -forces)
            
            # More granular learning rate schedule for extended fine-tuning
            if ft_idx < 350:
                ft_lr = 0.011
            elif ft_idx < 650:
                ft_lr = 0.0058
            elif ft_idx < 950:
                ft_lr = 0.0030
            elif ft_idx < 1250:
                ft_lr = 0.0015
            elif ft_idx < 1500:
                ft_lr = 0.00070
            elif ft_idx < 1700:
                ft_lr = 0.00032
            elif ft_idx < 1900:
                ft_lr = 0.00016
            elif ft_idx < 2050:
                ft_lr = 0.00008
            else:
                ft_lr = 0.00004
            
            points += ft_lr * grad
            points = _normalize_points(points)
            current_ratio, _, _ = _compute_objective(points)
            if current_ratio > ft_best_ratio:
                ft_best_ratio = current_ratio
                run_best_points = points.copy()
        
        # Update current_best_ratio with fine-tuning result
        current_best_ratio = ft_best_ratio
        
        # Track overall best across basin explorations
        if current_best_ratio > best_ratio_overall:
            best_ratio_overall = current_best_ratio
            best_points_overall = run_best_points.copy()
    
    return best_points_overall