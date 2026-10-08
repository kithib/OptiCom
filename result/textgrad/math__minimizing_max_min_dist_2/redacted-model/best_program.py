import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    np.random.seed(42)

    # Start with staggered hexagonal grid instead of square grid - better theoretical spacing
    # Hexagonal packing provides ~15% higher packing density than square packing
    grid_size = int(np.sqrt(n))
    x_regular = np.linspace(0, 1, grid_size)
    y_regular = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x_regular, y_regular)
    # Improved hexagonal stagger: 0.5 of grid spacing for better hexagonal approximation
    hex_offset = (x_regular[1] - x_regular[0]) * 0.5
    for i in range(grid_size):
        if i % 2 == 1:
            xv[i] += hex_offset
    points = np.column_stack([xv.ravel(), yv.ravel()]).astype(float)

    # Balanced iteration count with refined annealing schedule
    num_iters = 5500
    initial_step_size = 0.025
    best_ratio = 0.0
    best_points = points.copy()
    
    # Precompute mask for efficiency (upper triangle only to avoid double processing)
    mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    full_mask = ~np.eye(n, dtype=bool)
    
    # Pre-allocate arrays for efficiency
    dists = np.zeros((n, n))
    diffs = np.zeros((n, n, d))

    for iter_idx in range(num_iters):
        # Compute distances efficiently
        np.subtract(points[:, np.newaxis], points, out=diffs)
        np.square(diffs, out=diffs)
        np.sum(diffs, axis=2, out=dists)
        np.sqrt(dists, out=dists)
        
        dmin = dists[full_mask].min()
        dmax = dists[full_mask].max()
        ratio = dmin / dmax

        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()

        # Extended cosine annealing with longer cycle for late refinement
        cycle_length = 600 if iter_idx < num_iters * 0.7 else 250
        cycle_pos = iter_idx % cycle_length
        step_size = initial_step_size * 0.5 * (1 + np.cos(np.pi * cycle_pos / cycle_length))
        # Gradually reduce step size in late iterations for fine tuning
        if iter_idx > num_iters * 0.85:
            step_size *= 0.15
        
        # Perturb to improve dmin for closest pairs
        # Include pairs within 2% of dmin for more robust improvement
        min_dist_val = dists[mask].min()
        min_dist_threshold = min_dist_val * 1.02
        min_dist_pairs = np.where((dists <= min_dist_threshold) & mask)
        for i, j in zip(*min_dist_pairs):
            direction = points[j] - points[i]
            dir_norm = np.linalg.norm(direction)
            if dir_norm > 1e-8:
                direction /= dir_norm
                # Stronger perturbation for pairs at exact minimum
                strength = 0.65 if dists[i, j] == min_dist_val else 0.45
                points[i] -= direction * step_size * strength
                points[j] += direction * step_size * strength
        
        # Contract far pairs to reduce dmax - start earlier with dual threshold
        if iter_idx > int(num_iters * 0.12):
            far_dist_val = dists[mask].max()
            # Handle absolute maximum pairs with higher priority
            max_pairs = np.where((dists == far_dist_val) & mask)
            for i, j in zip(*max_pairs):
                direction = points[j] - points[i]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-8:
                    direction /= dir_norm
                    contract_strength = step_size * 0.5
                    points[i] += direction * contract_strength
                    points[j] -= direction * contract_strength
            
            # Also handle near-max pairs (top 12%) with moderate contraction
            if iter_idx > int(num_iters * 0.30):
                threshold = far_dist_val * 0.88
                far_pairs = np.where((dists >= threshold) & (dists < far_dist_val) & mask)
                for i, j in zip(*far_pairs):
                    direction = points[j] - points[i]
                    dir_norm = np.linalg.norm(direction)
                    if dir_norm > 1e-8:
                        direction /= dir_norm
                        contract_strength = step_size * 0.22
                        points[i] += direction * contract_strength
                        points[j] -= direction * contract_strength
        
        # Late-phase: additional contraction on pairs that define dmax (at corners)
        if iter_idx > int(num_iters * 0.55):
            max_dist_val = dists[mask].max()
            max_pairs = np.where((dists == max_dist_val) & mask)
            for i, j in zip(*max_pairs):
                direction = points[j] - points[i]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-8:
                    direction /= dir_norm
                    # Strong pull to reduce maximum distance without disturbing dmin
                    points[i] += direction * step_size * 0.3
                    points[j] -= direction * step_size * 0.3

        # Keep points in [0,1] bounds
        points = np.clip(points, 0, 1)

    # Compute current ratio for best_points before any modification
    np.subtract(best_points[:, np.newaxis], best_points, out=diffs)
    np.square(diffs, out=diffs)
    np.sum(diffs, axis=2, out=dists)
    np.sqrt(dists, out=dists)
    current_min = dists[full_mask].min()
    
    # Independent axis normalization for better [0,1] filling
    min_coords = best_points.min(axis=0, keepdims=True)
    max_coords = best_points.max(axis=0, keepdims=True)
    span = max_coords - min_coords
    
    # Identify maximum span axis
    max_span_idx = np.argmax(span)
    
    # Scale both axes by the same factor to maintain aspect ratio
    scale_factor = 1.0 / (span[:, max_span_idx] + 1e-8)
    
    # Apply uniform scaling
    best_points = (best_points - min_coords) * scale_factor
    
    # Center the points on both axes
    new_max_coords = best_points.max(axis=0, keepdims=True)
    best_points += (1.0 - new_max_coords) / 2.0
    
    # Final refinement phase: targeted dmax reduction with dmin protection
    refinement_iters = 800
    for iter_idx in range(refinement_iters):
        np.subtract(best_points[:, np.newaxis], best_points, out=diffs)
        np.square(diffs, out=diffs)
        np.sum(diffs, axis=2, out=dists)
        np.sqrt(dists, out=dists)
        
        current_max = dists[full_mask].max()
        current_min_ref = dists[full_mask].min()
        
        # Identify points with maximum coordinate values
        corner_mask = ((best_points[:, 0] == best_points[:, 0].max()) | 
                       (best_points[:, 0] == best_points[:, 0].min()) |
                       (best_points[:, 1] == best_points[:, 1].max()) | 
                       (best_points[:, 1] == best_points[:, 1].min()))
        
        max_pairs = np.where((dists == current_max) & mask)
        for i, j in zip(*max_pairs):
            if corner_mask[i] and corner_mask[j]:
                direction = best_points[j] - best_points[i]
                dir_norm = np.linalg.norm(direction)
                if dir_norm > 1e-8:
                    direction /= dir_norm
                    contract_strength = 0.0018 * (1 - iter_idx / refinement_iters)
                    # Test move
                    temp_i = best_points[i] + direction * contract_strength
                    temp_j = best_points[j] - direction * contract_strength
                    # Calculate min distance involving moved points
                    test_points = best_points.copy()
                    test_points[i] = temp_i
                    test_points[j] = temp_j
                    test_diffs = test_points[:, np.newaxis] - test_points
                    test_dists = np.sqrt(np.sum(test_diffs**2, axis=2))
                    test_min = test_dists[full_mask].min()
                    # Only accept if dmin doesn't drop too much - fixed reference variable
                    if test_min >= current_min * 0.995:
                        best_points[i] = temp_i
                        best_points[j] = temp_j
        
        best_points = np.clip(best_points, 0, 1)

    return best_points