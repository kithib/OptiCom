import numpy as np


def _compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute dmin/dmax ratio for a set of points."""
    diff = points[:, np.newaxis] - points[np.newaxis, :]
    dists = np.linalg.norm(diff, axis=2)
    upper_tri = dists[np.triu_indices_from(dists, k=1)]
    return np.min(upper_tri) / np.max(upper_tri)


def _normalize_points(points: np.ndarray) -> np.ndarray:
    """Center and normalize points to fit in unit sphere for consistent scaling."""
    centered = points - np.mean(points, axis=0)
    norms = np.linalg.norm(centered, axis=1, keepdims=True)
    max_norm = np.max(norms)
    if max_norm > 0:
        return centered / max_norm
    return centered


def _apply_local_force_steps(best_points: np.ndarray, best_ratio: float, num_steps: int = 7) -> tuple:
    """Apply local force-based gradient ascent for fine-grained improvement."""
    local_points = best_points.copy()
    local_vel = np.zeros_like(local_points)
    for fi in range(num_steps):
        diff = local_points[:, np.newaxis] - local_points[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        np.fill_diagonal(dists, np.inf)
        min_dist = np.min(dists)
        force_w = np.where(dists < min_dist * 3.6, 1.0 / (dists ** 2 + 1e-6), 0)
        force_w *= np.exp(-(dists / (min_dist * 2.9)) ** 2)
        force_w = np.minimum(force_w, 55.0)
        np.fill_diagonal(force_w, 0)
        forces = np.einsum('ijk,ij->ik', diff, force_w)
        local_vel = 0.26 * local_vel + 0.032 * forces
        local_points = _normalize_points(local_points + local_vel)
        local_ratio = _compute_min_max_ratio(local_points)
        if local_ratio > best_ratio:
            best_ratio = local_ratio
            best_points = local_points.copy()
    return best_points, best_ratio


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3

    # Reproducible deterministic approach: start with excellent symmetric configuration
    np.random.seed(42)
    
    # Initialize with a face-centered cubic lattice subset (14 points):
    # 8 vertices of scaled cube + 6 face centers (symmetric, well-balanced starting point)
    # Scale factor: 1/sqrt(3) ≈ 0.57735 for equal face diagonals and better initial spacing
    s = 0.5773502691896257
    vertices = np.array([[-s, -s, -s], [-s, -s, s], [-s, s, -s], [-s, s, s],
                         [s, -s, -s], [s, -s, s], [s, s, -s], [s, s, s]])
    face_centers = np.array([[-s, 0, 0], [s, 0, 0], [0, -s, 0], 
                              [0, s, 0], [0, 0, -s], [0, 0, s]])
    points = np.vstack([vertices, face_centers])
    points = _normalize_points(points)
    
    # Track best configuration found
    best_ratio = _compute_min_max_ratio(points)
    best_points = points.copy()
    
    # Phase 1: Force-based gradient ascent - Enhanced with momentum + Nesterov acceleration
    base_rate = 0.0575
    momentum = 0.27
    velocity = np.zeros_like(points)
    for i in range(160):
        # Anneal learning rate for finer convergence
        learning_rate = base_rate * (1 - i / 160) * 0.72 + base_rate * 0.28
        
        # Nesterov acceleration: look ahead with momentum before computing forces
        points_lookahead = _normalize_points(points + momentum * velocity)
        
        # Compute pairwise distance vectors and distances on lookahead
        diff = points_lookahead[:, np.newaxis] - points_lookahead[np.newaxis, :]
        dists = np.linalg.norm(diff, axis=2)
        
        # Avoid division by zero on diagonal
        np.fill_diagonal(dists, np.inf)
        
        # Find minimum distance pairs and create repulsive forces
        min_dist = np.min(dists)
        # Broader force radius with inverse-square falloff and gaussian soft cutoff + capped forces
        force_weights = np.where(dists < min_dist * 3.6, 1.0 / (dists ** 2 + 1e-6), 0)
        force_weights *= np.exp(-(dists / (min_dist * 2.9)) ** 2)
        force_weights = np.minimum(force_weights, 55.0)
        np.fill_diagonal(force_weights, 0)
        
        # Sum forces with momentum and update positions
        forces = np.einsum('ijk,ij->ik', diff, force_weights)
        velocity = momentum * velocity + learning_rate * forces
        points += velocity
        
        # Normalize and check for improvement
        points = _normalize_points(points)
        current_ratio = _compute_min_max_ratio(points)
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
    
    # Phase 2: Stochastic perturbation search with Metropolis-style acceptance + gradient ascent interleaving
    for restart in range(7):
        # Slight jitter for restart from current best (balanced exploration/exploitation)
        if restart == 0:
            points = best_points.copy()
        else:
            points = best_points + np.random.normal(0, 0.023, best_points.shape)
            points = _normalize_points(points)
            
        for iteration in range(700):
            step_size = 0.044 * (1 - iteration / 700)
            perturbations = np.random.normal(0, step_size, (n, d))
            new_points = points + perturbations
            new_points = _normalize_points(new_points)
            
            new_ratio = _compute_min_max_ratio(new_points)
            current_points_ratio = _compute_min_max_ratio(points)
            # Metropolis-style acceptance: accept if better OR with probability based on how much worse
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
                points = new_points
            else:
                acceptance_prob = min(1.0, np.exp(41.0 * (new_ratio - current_points_ratio)))
                if new_ratio > current_points_ratio or np.random.rand() < acceptance_prob:
                    points = new_points
            
            # Periodically run local force steps for fine-grained improvement (now extracted as helper)
            if iteration % 45 == 44 and iteration > 0:
                best_points, best_ratio = _apply_local_force_steps(best_points, best_ratio, num_steps=7)
    
    # Phase 3: Diminishing random walk with occasional restarts to best point (fine polish)
    points = best_points.copy()
    for iteration in range(500):
        if iteration % 35 == 0:
            points = best_points.copy()
        step_size = 0.012 * (1 - iteration / 500)
        perturbations = np.random.normal(0, step_size, (n, d))
        new_points = points + perturbations
        new_points = _normalize_points(new_points)
        
        new_ratio = _compute_min_max_ratio(new_points)
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_points = new_points.copy()
            points = new_points
    
    # Final force-based polish to maximize spacing
    best_points, best_ratio = _apply_local_force_steps(best_points, best_ratio, num_steps=10)
    
    return best_points