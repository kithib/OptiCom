# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    np.random.seed(42)
    
    # Start with 4x4 grid inset from boundaries for better optimization
    grid_size = int(np.sqrt(n))
    x = np.linspace(0.05, 0.95, grid_size)
    y = np.linspace(0.05, 0.95, grid_size)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Add small controlled perturbation to break perfect grid symmetry
    perturbation = 0.02 * np.random.randn(n, 2)
    points += perturbation
    points = np.clip(points, 0, 1)
    
    # Apply gradient ascent optimization with annealing
    base_learning_rate = 0.01
    iterations = 500
    
    for i in range(iterations):
        # Learning rate annealing to stabilize later iterations
        learning_rate = base_learning_rate * (1 - i / iterations)
        
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, np.inf)
        
        # Handle multiple close pairs for better minimum distance improvement
        # Get top 2 closest pairs instead of just 1
        flat_dist = dist_matrix.ravel()
        closest_indices = np.argsort(flat_dist)[:4]  # Get 2 pairs (each pair appears twice in symmetric matrix)
        unique_pairs = []
        seen = set()
        for idx in closest_indices:
            pair = np.unravel_index(idx, dist_matrix.shape)
            if pair[0] != pair[1] and (pair[1], pair[0]) not in seen:
                unique_pairs.append(pair)
                seen.add(pair)
            if len(unique_pairs) >= 2:
                break
        
        # Apply forces to multiple min pairs for better dispersion
        for idx, min_dist_idx in enumerate(unique_pairs):
            weight = 1.0 - (idx * 0.3)  # Decreasing weight for secondary pairs
            direction_min = points[min_dist_idx[0]] - points[min_dist_idx[1]]
            direction_min /= np.linalg.norm(direction_min) + 1e-8
            points[min_dist_idx[0]] += learning_rate * direction_min * weight
            points[min_dist_idx[1]] -= learning_rate * direction_min * weight
        
        # Move max pair closer with gentle force
        max_dist_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        direction_max = points[max_dist_idx[0]] - points[max_dist_idx[1]]
        direction_max /= np.linalg.norm(direction_max) + 1e-8
        points[max_dist_idx[0]] -= learning_rate * direction_max * 0.2
        points[max_dist_idx[1]] += learning_rate * direction_max * 0.2
        
        # Keep points within unit square
        points = np.clip(points, 0, 1)
    
    return points


# EVOLVE-BLOCK-END