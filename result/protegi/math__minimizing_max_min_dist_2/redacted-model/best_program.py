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
    d = 2

    # Start with nearly uniform grid (4x4 grid with small perturbation for potential improvement)
    np.random.seed(42)
    grid_size = int(np.sqrt(n))
    x = np.linspace(0, 1, grid_size, endpoint=True)
    y = np.linspace(0, 1, grid_size, endpoint=True)
    xv, yv = np.meshgrid(x, y)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Simple gradient ascent to improve min/max ratio
    learning_rate = 0.005
    iterations = 200
    
    for _ in range(iterations):
        distances = pdist(points)
        d_min = np.min(distances)
        d_max = np.max(distances)
        
        # Compute pairwise distance vectors
        grad = np.zeros_like(points)
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                diff = points[i] - points[j]
                dist = distances[pair_idx]
                
                # Push points apart if they contribute to d_min
                if dist == d_min and dist > 0:
                    force = diff / dist
                    grad[i] += force
                    grad[j] -= force
                
                # Pull points together if they contribute to d_max
                if dist == d_max and dist > 0:
                    force = diff / dist
                    grad[i] -= force
                    grad[j] += force
                
                pair_idx += 1
        
        # Update points and clamp to [0,1]
        points += learning_rate * grad
        points = np.clip(points, 0, 1)

    # Fine-tuning phase with simulated annealing
    iterations_sa = 500
    initial_temp = 0.01
    final_temp = 0.0001
    
    dist_matrix = squareform(pdist(points))
    best_dmin = dist_matrix[dist_matrix > 0].min()
    best_dmax = dist_matrix.max()
    best_ratio = best_dmin / best_dmax
    best_points = points.copy()
    
    for i in range(iterations_sa):
        temp = initial_temp * (final_temp / initial_temp) ** (i / iterations_sa)
        point_idx = np.random.randint(0, n)
        perturbation = np.random.normal(0, temp, 2)
        
        new_points = points.copy()
        new_points[point_idx] = np.clip(new_points[point_idx] + perturbation, 0.0, 1.0)
        
        new_dist_matrix = squareform(pdist(new_points))
        new_dmin = new_dist_matrix[new_dist_matrix > 0].min()
        new_dmax = new_dist_matrix.max()
        new_ratio = new_dmin / new_dmax
        
        if new_ratio > best_ratio:
            points = new_points
            best_points = new_points
            best_ratio = new_ratio
    
    return best_points


# EVOLVE-BLOCK-END