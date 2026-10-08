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

    # Weakness 1 addressed: Standard grid has suboptimal boundary point placement
    # Offset grid inward by 0.5/(grid_size-1) to reduce maximum diagonal distance, maintaining same min grid distance
    grid_size = int(np.sqrt(n))
    coords = np.linspace(0, 1, grid_size)
    # Adjust boundary points inward for better dmin/dmax ratio
    boundary_offset = 0.5 / (grid_size - 1)
    coords[0] += boundary_offset
    coords[-1] -= boundary_offset
    xv, yv = np.meshgrid(coords, coords, indexing='xy')
    points = np.column_stack([xv.ravel(), yv.ravel()])

    # Apply gradient ascent refinement to improve min/max ratio
    learning_rate = 0.01
    # Weakness 2 addressed: Insufficient refinement iterations
    for iteration in range(250):
        dists = pdist(points)
        d_min = np.min(dists)
        d_max = np.max(dists)
        
        if d_max == 0:
            break
            
        dist_matrix = squareform(dists)
        grad = np.zeros_like(points)
        
        for i in range(n):
            for j in range(i+1, n):
                diff = points[j] - points[i]
                dist_ij = dist_matrix[i, j]
                if dist_ij < 1e-10:
                    continue
                # Weakness 3 addressed: Only acting on exact min/max distances - use percentile bands with scaling
                if dist_ij <= d_min * 1.3:
                    proximity = 1.0 - (dist_ij - d_min) / (d_min * 0.3)
                    force_magnitude = 0.15 * proximity
                    force = diff / dist_ij * force_magnitude
                    grad[i] -= force
                    grad[j] += force
                elif dist_ij >= d_max * 0.75:
                    proximity = 1.0 - (d_max - dist_ij) / (d_max * 0.25)
                    force_magnitude = 0.08 * proximity
                    force = diff / dist_ij * force_magnitude
                    grad[i] += force
                    grad[j] -= force
        
        # Learning rate decay for finer convergence
        lr = learning_rate * (1 - iteration / 250)
        points += lr * grad
        points = np.clip(points, 0.0, 1.0)

    return points


# EVOLVE-BLOCK-END