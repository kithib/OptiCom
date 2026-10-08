# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import pdist, squareform


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses a hybrid structured initial configuration with gradient-based optimization featuring momentum,
    cyclic learning rate adjustment, plateau detection for periodic restarts, and per-point nearest 
    neighbor gradient computation for improved dispersion.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    n = 14
    d = 3
    np.random.seed(42)  # Ensure reproducibility
    
    # Start with structured symmetric configuration
    # Cube vertices (8 points) + octahedron vertices (6 points) = 14 points
    cube = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                     [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
    octahedron = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1],
                           [0.5, 0, 0.5], [0.5, 1, 0.5],
                           [0, 0.5, 0.5], [1, 0.5, 0.5]], dtype=float)
    points = np.vstack([cube, octahedron])
    # Center configuration
    points -= points.mean(axis=0)
    # Normalize to unit sphere before adding noise (improves initial symmetry)
    points /= np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
    # Add small controlled noise to break symmetry
    points += np.random.normal(0, 0.03, size=points.shape)
    
    # Gradient ascent optimization with momentum, cyclic learning rate, and periodic restart
    learning_rate = 0.05
    momentum = 0.9
    velocity = np.zeros_like(points)
    best_ratio = 0.0
    best_points = points.copy()
    plateau_count = 0
    
    for epoch in range(2000):
        # Evaluate current configuration
        distances = pdist(points)
        dmin_idx = distances.argmin()
        dmax_idx = distances.argmax()
        
        # Convert linear indices to pair indices
        triu_i, triu_j = np.triu_indices(n, k=1)
        min_pair = (triu_i[dmin_idx], triu_j[dmin_idx])
        max_pair = (triu_i[dmax_idx], triu_j[dmax_idx])
        
        dmin = distances[dmin_idx]
        dmax = distances[dmax_idx]
        current_ratio = dmin / dmax
        
        # Track best configuration and detect plateau with epsilon for numerical stability
        if current_ratio > best_ratio + 1e-6:
            best_ratio = current_ratio
            best_points = points.copy()
            plateau_count = 0
        else:
            plateau_count += 1
        
        # Periodic restart: if no improvement for 300 epochs, reset with noise around best
        if plateau_count >= 300 and epoch > 500:
            points = best_points.copy()
            points += np.random.normal(0, 0.03, size=points.shape)
            learning_rate *= 0.9
            velocity.fill(0)
            plateau_count = 0
        
        # Cyclic learning rate adjustment to escape local maxima
        if epoch % 200 == 0 and epoch > 0:
            learning_rate *= 0.8
            velocity.fill(0)
        
        # Compute gradients using per-point nearest neighbor approach (for min distance improvement)
        grad = np.zeros_like(points)
        dist_matrix = squareform(distances)
        # Move points away from their nearest neighbors to increase min distance
        for i in range(n):
            nearest_idx = np.argmin(dist_matrix[i] + np.eye(n)[i] * 1e6)
            diff = points[i] - points[nearest_idx]
            norm = np.linalg.norm(diff)
            if norm > 0:
                grad[i] += diff / norm
                grad[nearest_idx] -= diff / norm
        # Gradient for max pair (decrease dmax, multiplied by ratio factor from proper gradient formulation)
        diff_max = points[max_pair[0]] - points[max_pair[1]]
        grad_max = diff_max / (dmax + 1e-8)
        grad[max_pair[0]] -= grad_max * current_ratio
        grad[max_pair[1]] += grad_max * current_ratio
        
        # Update with momentum
        velocity = momentum * velocity + (1 - momentum) * grad
        points += learning_rate * velocity
        
        # Normalize to unit sphere to bound dmax
        points /= np.linalg.norm(points, axis=1, keepdims=True) + 1e-8
    
    # Normalize to unit cube [0, 1]^3 while preserving relative distances
    min_coords = np.min(best_points, axis=0)
    max_coords = np.max(best_points, axis=0)
    best_points = (best_points - min_coords) / (max_coords - min_coords)
    
    return best_points


# EVOLVE-BLOCK-END