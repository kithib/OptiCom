# EVOLVE-BLOCK-START
import numpy as np


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2

    # Start with 4x4 grid centered in unit square with equal spacing
    grid_size = int(np.sqrt(n))
    coords = np.linspace(1/(2*grid_size), 1 - 1/(2*grid_size), grid_size)
    xv, yv = np.meshgrid(coords, coords)
    points = np.column_stack([xv.ravel(), yv.ravel()])

    # Extended gradient ascent optimization with adaptive learning rate and best tracking
    np.random.seed(42)
    initial_learning_rate = 0.005
    iterations = 1050  # Further extended iterations for enhanced refinement
    best_ratio = -np.inf
    best_points = points.copy()
    
    # Cache upper triangle indices outside loop for efficiency
    i_indices, j_indices = np.triu_indices(n, k=1)
    
    # Pre-cache diagonal and anti-diagonal classification with optimized strengths
    attract_strengths = np.empty(len(i_indices), dtype=np.float64)
    for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
        diag_1_diff = abs(i % 4 - j % 4) - abs(i // 4 - j // 4)
        diag_2_diff = abs(i % 4 - j % 4) + abs(i // 4 - j // 4)
        is_main_diag = diag_1_diff == 0
        is_anti_diag = diag_2_diff == 3
        
        if is_main_diag:
            attract_strengths[idx] = 0.215  # Fine-tuned main diagonal attraction
        elif is_anti_diag:
            attract_strengths[idx] = 0.425  # Optimized anti-diagonal for balanced aspect ratio
        else:
            attract_strengths[idx] = 0.74  # Enhanced off-axis attraction to compress excess dimensions
    
    # Pre-cache sqrt(2) reference for aspect ratio normalization
    sqrt2 = np.sqrt(2)
    
    for iter_idx in range(iterations):
        grad = np.zeros_like(points)
        
        # Calculate all pairwise distances - vectorized
        diffs = points[:, np.newaxis] - points[np.newaxis, :]
        dists_sq = np.sum(diffs ** 2, axis=2)
        dists = np.sqrt(dists_sq)
        
        # Avoid diagonal
        np.fill_diagonal(dists, np.inf)
        
        dmin = np.min(dists)
        dmax = np.max(dists[dists != np.inf])
        current_ratio = dmin / dmax
        
        # Track best configuration found with momentum-based backup for local optima escape
        if current_ratio > best_ratio:
            best_ratio = current_ratio
            best_points = points.copy()
        
        # Adaptive learning rate: reduces to 10% for ultra-fine tuning (more aggressive decay)
        progress = iter_idx / iterations
        learning_rate = initial_learning_rate * (1 - progress * 0.90)
        
        # Adaptive thresholds with more aggressive tightening for precision
        dmin_threshold = dmin + 0.025 * (1 - progress * 0.75)
        dmax_threshold = dmax - 0.013 * (1 - progress * 0.75)
        
        # Aspect ratio normalization term to guide toward optimal square-like arrangement
        width = np.max(points[:, 0]) - np.min(points[:, 0])
        height = np.max(points[:, 1]) - np.min(points[:, 1])
        aspect_ratio = max(width, height) / (min(width, height) + 1e-12)
        
        # Apply forces using pre-cached upper triangle indices
        for idx, (i, j) in enumerate(zip(i_indices, j_indices)):
            diff = points[j] - points[i]
            dist = dists[i, j]
            
            # Enhanced repulsion for close pairs with dmin-based boost and distance-aware scaling
            if dist < dmin_threshold:
                dist_ratio = dmin / (dist + 1e-12)
                repel_mag = 1.22 + min(0.15, (dist_ratio - 1) * 0.06)
                repel_force = learning_rate * repel_mag * (dmin_threshold - dist) * diff / (dist * dist + 1e-12)
                grad[i] -= repel_force
                grad[j] += repel_force
            
            # Aspect ratio corrected attraction forces with refined bounds
            if dist > dmax_threshold:
                ar_correction = 1.0 + min(0.18, (aspect_ratio - sqrt2) * 0.12)
                attract_force = learning_rate * ar_correction * attract_strengths[idx] * diff / (dist + 1e-12)
                grad[i] += attract_force
                grad[j] -= attract_force
        
        points += grad
        # Keep points within unit square bounds with minimal margin
        points = np.clip(points, 0.004, 0.996)

    # Return best configuration found
    return best_points


# EVOLVE-BLOCK-END