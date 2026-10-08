import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    
    # Initial configuration: cube vertices (8) + face centers (6) = 14 points
    # This is known to have good dispersion properties
    points = np.zeros((14, 3))
    
    # First 8: cube vertices at (±1, ±1, ±1)
    idx = 0
    for x in [-1.0, 1.0]:
        for y in [-1.0, 1.0]:
            for z in [-1.0, 1.0]:
                points[idx] = [x, y, z]
                idx += 1
    
    # Next 6: face centers (±1,0,0), (0,±1,0), (0,0,±1)
    points[8] = [1.3, 0.0, 0.0]
    points[9] = [-1.3, 0.0, 0.0]
    points[10] = [0.0, 1.3, 0.0]
    points[11] = [0.0, -1.3, 0.0]
    points[12] = [0.0, 0.0, 1.3]
    points[13] = [0.0, 0.0, -1.3]
    
    # Add small controlled perturbation
    points += 0.03 * np.random.randn(14, 3)
    
    # Normalize to unit sphere initially
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Refined gradient ascent with better scaling and multi-pair handling
    for iteration in range(300):
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        # Learning rate scheduling
        lr_repel = 0.015 * (1 - iteration / 300) + 0.002
        lr_attract = 0.0075 * (1 - iteration / 300) + 0.001
        
        # Handle MULTIPLE closest pairs for more robust improvement
        flat_dists = dists.flatten()
        closest_indices = np.argsort(flat_dists)[:6]  # Top 3 unique closest pairs
        for idx in closest_indices:
            i, j = idx // 14, idx % 14
            if i < j:  # Avoid double counting
                grad_repel = (points[i] - points[j]) * lr_repel
                points[i] += grad_repel
                points[j] -= grad_repel
        
        # Attract MULTIPLE farthest pairs
        farthest_indices = np.argsort(flat_dists)[-4:]  # Top 2 unique farthest pairs
        for idx in farthest_indices:
            k, l = idx // 14, idx % 14
            if k < l:  # Avoid double counting
                grad_attract = (points[k] - points[l]) * lr_attract
                points[k] -= grad_attract
                points[l] += grad_attract
        
        # Re-normalize to unit sphere
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    return points