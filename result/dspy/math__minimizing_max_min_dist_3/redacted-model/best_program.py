import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    
    # Start with vertices of a cube (8 points) and a regular octahedron (6 points)
    # This gives us 14 known symmetric points with good spacing properties
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
    
    # Center octahedron within the cube
    octahedron = np.array([[0.5, 0.5, 0.0], [0.5, 0.5, 1.0],
                           [0.5, 0.0, 0.5], [0.5, 1.0, 0.5],
                           [0.0, 0.5, 0.5], [1.0, 0.5, 0.5]])
    
    points = np.vstack([cube, octahedron])
    
    # Perform gradient ascent to maximize min/max ratio
    learning_rate = 0.01
    iterations = 500
    
    for _ in range(iterations):
        dist_matrix = squareform(pdist(points))
        np.fill_diagonal(dist_matrix, np.inf)
        
        d_min = np.min(dist_matrix)
        d_max = np.max(dist_matrix)
        ratio = d_min / d_max
        
        # Find which pair gives min distance and max distance
        min_pair = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
        max_pair = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
        
        # Push points away for the minimum distance pair
        i, j = min_pair
        direction = points[i] - points[j]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        points[i] += learning_rate * direction
        points[j] -= learning_rate * direction
        
        # Pull points together for the maximum distance pair
        i, j = max_pair
        direction = points[i] - points[j]
        direction = direction / (np.linalg.norm(direction) + 1e-8)
        points[i] -= learning_rate * direction
        points[j] += learning_rate * direction
        
        # Keep points within unit cube bounds
        points = np.clip(points, 0, 1)
    
    return points