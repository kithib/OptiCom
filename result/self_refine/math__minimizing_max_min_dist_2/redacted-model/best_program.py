import numpy as np
from scipy.spatial.distance import pdist


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    d = 2
    grid_size = int(np.sqrt(n))  # 4x4 grid for 16 points
    
    # Start with regular grid centered in unit square (better symmetric initialization)
    cell_size = 1.0 / grid_size
    offset = cell_size / 2.0
    x_coords = np.linspace(offset, 1.0 - offset, grid_size)
    y_coords = np.linspace(offset, 1.0 - offset, grid_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.column_stack((xv.ravel(), yv.ravel()))
    
    # Gradient ascent with simulated annealing to improve min/max ratio
    np.random.seed(42)
    learning_rate = 0.023
    iterations = 2600  # Balanced iterations for better convergence without excessive runtime
    best_ratio = 0.0
    best_points = points.copy()
    
    for iteration in range(iterations):
        # Learning rate schedule: reduce over time for finer adjustments
        current_lr = learning_rate * (1 - iteration / iterations)
        # Temperature for simulated annealing: accept worse moves occasionally early on
        temperature = max(0.004, 0.115 * (1 - iteration / iterations))
        
        distances = pdist(points)
        dmin = distances.min()
        dmax = distances.max()
        ratio = dmin / dmax
        
        if ratio > best_ratio:
            best_ratio = ratio
            best_points = points.copy()
        
        # Compute pairwise distance vectors - improved force calculation
        grad = np.zeros_like(points)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = points[i] - points[j]
                dist = distances[idx]
                if dist < dmin + 1e-6:
                    # Push apart points that are too close (stronger, distance-dependent force)
                    force_mag = 2.7 - (dist / dmin) if dmin > 1e-9 else 2.0
                    force = (diff / dist) * force_mag if dist > 1e-9 else np.random.randn(d) * 0.1
                    grad[i] += force
                    grad[j] -= force
                elif dist > dmax - 1e-6:
                    # Pull together points that are too far (balanced distance-dependent force)
                    force_mag = 1.35 + 0.85 * ((dmax - dist) / dmax) if dmax > 1e-9 else 1.45
                    force = (diff / dist) * force_mag if dist > 1e-9 else np.random.randn(d) * 0.1
                    grad[i] -= force
                    grad[j] += force
                idx += 1
        
        # Apply gradient with adaptive noise for exploration (reduces over time)
        noise_scale = current_lr * 0.23 * (1 - iteration / iterations)
        noise = np.random.randn(n, d) * noise_scale
        new_points = points + current_lr * grad + noise
        new_points = np.clip(new_points, 0.017, 0.983)  # Optimized bounds to balance dmin/dmax
        
        # Evaluate new points
        new_distances = pdist(new_points)
        new_dmin = new_distances.min()
        new_dmax = new_distances.max()
        new_ratio = new_dmin / new_dmax
        
        # Acceptance: always accept better, sometimes accept worse (early iterations)
        if new_ratio > ratio or np.random.rand() < np.exp((new_ratio - ratio) / temperature):
            points = new_points
    
    # Enhanced three-stage local refinement around best points (fine tuning)
    # First stage: moderate adjustments
    refine_lr = 0.0028
    for _ in range(350):
        distances = pdist(best_points)
        dmin = distances.min()
        dmax = distances.max()
        grad = np.zeros_like(best_points)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = best_points[i] - best_points[j]
                dist = distances[idx]
                if dist < dmin + 1e-6:
                    force_mag = 2.15 - 1.15 * (dist / dmin) if dmin > 1e-9 else 1.2
                    force = (diff / dist) * force_mag if dist > 1e-9 else 0
                    grad[i] += force
                    grad[j] -= force
                elif dist > dmax - 1e-6:
                    force_mag = 1.15 + 0.55 * ((dmax - dist) / dmax) if dmax > 1e-9 else 1.15
                    force = (diff / dist) * force_mag if dist > 1e-9 else 0
                    grad[i] -= force
                    grad[j] += force
                idx += 1
        best_points = np.clip(best_points + refine_lr * grad, 0.017, 0.983)
    
    # Second stage: fine adjustments
    refine_lr = 0.0009
    for _ in range(300):
        distances = pdist(best_points)
        dmin = distances.min()
        dmax = distances.max()
        grad = np.zeros_like(best_points)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = best_points[i] - best_points[j]
                dist = distances[idx]
                if dist < dmin + 1e-6:
                    force_mag = 1.65 - 0.65 * (dist / dmin) if dmin > 1e-9 else 1.05
                    force = (diff / dist) * force_mag if dist > 1e-9 else 0
                    grad[i] += force
                    grad[j] -= force
                elif dist > dmax - 1e-6:
                    force = (diff / dist) * 0.85 if dist > 1e-9 else 0
                    grad[i] -= force
                    grad[j] += force
                idx += 1
        best_points = np.clip(best_points + refine_lr * grad, 0.017, 0.983)
    
    # Third stage: very fine micro-adjustments
    refine_lr = 0.0003
    for _ in range(250):
        distances = pdist(best_points)
        dmin = distances.min()
        dmax = distances.max()
        grad = np.zeros_like(best_points)
        idx = 0
        for i in range(n):
            for j in range(i+1, n):
                diff = best_points[i] - best_points[j]
                dist = distances[idx]
                if dist < dmin + 1e-6:
                    force_mag = 1.25 - 0.25 * (dist / dmin) if dmin > 1e-9 else 1.0
                    force = (diff / dist) * force_mag if dist > 1e-9 else 0
                    grad[i] += force
                    grad[j] -= force
                elif dist > dmax - 1e-6:
                    force = (diff / dist) * 0.45 if dist > 1e-9 else 0
                    grad[i] -= force
                    grad[j] += force
                idx += 1
        best_points = np.clip(best_points + refine_lr * grad, 0.017, 0.983)
    
    return best_points