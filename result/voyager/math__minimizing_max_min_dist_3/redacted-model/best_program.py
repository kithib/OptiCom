# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    
    # Start with perturbed cube face centers + vertices (14 total known configuration)
    # Cube has 6 face centers + 8 vertices, but we need exactly 14
    # Use: 8 vertices + 6 face centers of a unit cube
    vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    face_centers = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1],
                              [0.5, 0, 0.5], [0.5, 1, 0.5],
                              [0, 0.5, 0.5], [1, 0.5, 0.5]])
    points = np.vstack([vertices, face_centers])
    
    # Simple gradient ascent to improve min/max ratio
    learning_rate = 0.01
    iterations = 500
    
    for _ in range(iterations):
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, np.inf)
        
        d_min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        d_max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        
        # Pull close points apart, push far points together
        min_pair = points[list(d_min_idx)]
        max_pair = points[list(d_max_idx)]
        
        # Update min distance pair
        direction_min = min_pair[0] - min_pair[1]
        direction_min /= (np.linalg.norm(direction_min) + 1e-8)
        points[d_min_idx[0]] += learning_rate * direction_min
        points[d_min_idx[1]] -= learning_rate * direction_min
        
        # Update max distance pair
        direction_max = max_pair[0] - max_pair[1]
        direction_max /= (np.linalg.norm(direction_max) + 1e-8)
        points[d_max_idx[0]] -= learning_rate * direction_max * 0.5
        points[d_max_idx[1]] += learning_rate * direction_max * 0.5
        
        # Normalize to unit cube
        points = np.clip(points, 0, 1)
    
    return points


# EVOLVE-BLOCK-END