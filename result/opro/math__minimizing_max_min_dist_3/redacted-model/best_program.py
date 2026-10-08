# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a perturbed regular hexagonal dipyramid configuration with gradient-based refinement.
    
    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.
    """
    n = 14
    d = 3

    # Reproducibility: fixed random seed
    np.random.seed(42)
    
    # Initialize with a structured configuration: hexagonal dipyramid (12 points) + 2 extra points
    # Hexagon in xy-plane (6 points)
    theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    hex_points = np.column_stack([np.cos(theta), np.sin(theta), np.zeros(6)])
    
    # Dipyramid apexes (2 points)
    apexes = np.array([[0, 0, 1.0], [0, 0, -1.0]])
    
    # Extra points (6 more to reach 14) - placed symmetrically
    extra_theta = np.linspace(0, 2*np.pi, 6, endpoint=False)
    extra_points = np.column_stack([0.5*np.cos(extra_theta), 0.5*np.sin(extra_theta), 0.5*np.ones(6)])
    
    points = np.vstack([hex_points, apexes, extra_points])
    
    # Add small random perturbation
    points += 0.05 * np.random.randn(n, d)
    
    # Normalize points to unit sphere for initial constraint
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Simple gradient ascent to refine min/max ratio
    learning_rate = 0.01
    iterations = 500
    
    for _ in range(iterations):
        dists = squareform(pdist(points))
        dmin_idx = np.unravel_index(np.argmin(dists + np.eye(n)*1e10), dists.shape)
        dmax_idx = np.unravel_index(np.argmax(dists), dists.shape)
        
        dmin_vec = points[dmin_idx[0]] - points[dmin_idx[1]]
        dmax_vec = points[dmax_idx[0]] - points[dmax_idx[1]]
        
        # Push min distance points apart, pull max distance points together
        grad = np.zeros_like(points)
        grad[dmin_idx[0]] += dmin_vec / (np.linalg.norm(dmin_vec) + 1e-10)
        grad[dmin_idx[1]] -= dmin_vec / (np.linalg.norm(dmin_vec) + 1e-10)
        grad[dmax_idx[0]] -= dmax_vec / (np.linalg.norm(dmax_vec) + 1e-10)
        grad[dmax_idx[1]] += dmax_vec / (np.linalg.norm(dmax_vec) + 1e-10)
        
        points += learning_rate * grad
        
        # Renormalize to unit cube
        points = np.clip(points, 0, 1)
    
    return points
# EVOLVE-BLOCK-END