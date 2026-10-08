import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    n = 14
    d = 3
    
    # Start with high-quality 14-point spherical code (cube vertices + 6 phi-based points)
    phi = (1 + np.sqrt(5)) / 2
    points = np.array([
        [ 1,  1,  1],
        [ 1,  1, -1],
        [ 1, -1,  1],
        [ 1, -1, -1],
        [-1,  1,  1],
        [-1,  1, -1],
        [-1, -1,  1],
        [-1, -1, -1],
        [ phi,  1/phi,  0],
        [-phi,  1/phi,  0],
        [ phi, -1/phi,  0],
        [-phi, -1/phi,  0],
        [ 1/phi,  0,  phi],
        [ 1/phi,  0, -phi]
    ])
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Enhanced gradient ascent with explicit ratio optimization
    lr = 0.07
    lr_min_factor = 2.5  # Give more weight to min distance improvement
    for iter_idx in range(800):
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        # Find all min distance pairs (handle ties)
        dmin = np.min(dists)
        min_pairs = np.argwhere(dists == dmin)
        
        # Find all max distance pairs (handle ties)
        dmax = np.max(dists)
        max_pairs = np.argwhere(dists == dmax)
        
        grad = np.zeros_like(points)
        
        # Push all min distance pairs apart (more important for ratio)
        for i, j in min_pairs:
            if i < j:
                grad[i] += lr_min_factor * (points[i] - points[j]) / dmin
                grad[j] += lr_min_factor * (points[j] - points[i]) / dmin
        
        # Pull all max distance pairs together
        for k, l in max_pairs:
            if k < l:
                grad[k] -= (points[k] - points[l]) / dmax
                grad[l] -= (points[l] - points[k]) / dmax
        
        # Add repulsion from all close neighbors to improve overall spacing
        # Gradually reduce neighbor repulsion as iterations progress
        neighbor_weight = 0.3 * (1 - iter_idx / 800)
        for m in range(n):
            close_mask = (dists[m] < dmin * 2.5) & (np.arange(n) != m)
            if np.any(close_mask):
                for neighbor in np.where(close_mask)[0]:
                    grad[m] += neighbor_weight * (points[m] - points[neighbor]) / dists[m, neighbor]
        
        points += lr * grad
        # Project to unit sphere
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        # Slower learning rate decay for more fine-grained optimization
        lr *= 0.9993
    
    # Extended fine-tuning phase for marginal ratio gains
    lr = 0.005
    for iter_idx in range(300):
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        dmin = np.min(dists)
        min_pairs = np.argwhere(dists == dmin)
        
        dmax = np.max(dists)
        max_pairs = np.argwhere(dists == dmax)
        
        grad = np.zeros_like(points)
        
        # Increased min factor for fine-tuning
        for i, j in min_pairs:
            if i < j:
                grad[i] += 2.0 * (points[i] - points[j]) / dmin
                grad[j] += 2.0 * (points[j] - points[i]) / dmin
        
        for k, l in max_pairs:
            if k < l:
                grad[k] -= (points[k] - points[l]) / dmax
                grad[l] -= (points[l] - points[k]) / dmax
        
        points += lr * grad
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        lr *= 0.998
    
    return points