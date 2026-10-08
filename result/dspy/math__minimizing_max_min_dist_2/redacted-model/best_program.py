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
    np.random.seed(42)
    
    # Start with a centered 4x4 grid (known good configuration for dispersion)
    grid_size = int(np.sqrt(n))
    cell_size = 1.0 / grid_size
    x_coords = np.linspace(cell_size/2, 1 - cell_size/2, grid_size)
    y_coords = np.linspace(cell_size/2, 1 - cell_size/2, grid_size)
    xv, yv = np.meshgrid(x_coords, y_coords)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Small initial perturbation to break perfect grid symmetry
    points += np.random.normal(0, 0.01, points.shape)
    points = np.clip(points, 0.01, 0.99)
    
    # Helper function to compute the ratio
    def compute_ratio(pts):
        dists = pdist(pts)
        dmin = dists.min()
        dmax = dists.max()
        return dmin / dmax if dmax > 0 else 0, dmin, dmax
    
    # Phase 1: Enhanced gradient ascent optimization with more iterations and better annealing
    learning_rate = 0.006
    num_iterations = 12000
    
    for iter_num in range(num_iterations):
        # Anneal learning rate gradually with improved schedule
        current_lr = learning_rate * (1 - iter_num / num_iterations) * 0.7 + learning_rate * 0.3
        
        # Compute distance matrix using vectorized operations for efficiency
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        
        dmin = dist.min()
        dmax = dist.max()
        
        # Find indices of min and max distance pairs
        min_idx = np.unravel_index(np.argmin(dist_sq), (n, n))
        max_idx = np.unravel_index(np.argmax(dist_sq), (n, n))
        
        # dmin gradient: push min pair apart
        i_min, j_min = min_idx
        grad_min = np.zeros_like(points)
        if dmin > 1e-8:
            direction_min = (points[i_min] - points[j_min]) / dmin
            grad_min[i_min] = direction_min
            grad_min[j_min] = -direction_min
        
        # dmax gradient: pull max pair together
        i_max, j_max = max_idx
        grad_max = np.zeros_like(points)
        if dmax > 1e-8:
            direction_max = (points[i_max] - points[j_max]) / dmax
            grad_max[i_max] = -direction_max
            grad_max[j_max] = direction_max
        
        # Combined gradient for ratio = dmin/dmax
        grad = (grad_min / dmax) - (dmin / (dmax**2)) * grad_max
        
        points += current_lr * grad
        points = np.clip(points, 0.01, 0.99)
        points = points - points.mean(axis=0) + 0.5
    
    # Phase 2: Enhanced hill climbing refinement with annealing and frequent multi-point moves
    best_ratio, _, _ = compute_ratio(points)
    step_size = 0.035
    hill_climb_iterations = 18000
    
    for iter_num in range(hill_climb_iterations):
        # Anneal step size gradually with improved schedule
        current_step = step_size * (1 - iter_num / hill_climb_iterations) * 0.65 + step_size * 0.35
        
        # Try moving multiple points more frequently
        if iter_num % 2 == 0:
            # Move 2-5 points
            num_moves = np.random.randint(2, 6)
            idxs = np.random.choice(n, num_moves, replace=False)
            new_points = points.copy()
            for idx in idxs:
                delta = np.random.uniform(-current_step, current_step, 2)
                new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        else:
            # Move single point
            idx = np.random.randint(n)
            delta = np.random.uniform(-current_step, current_step, 2)
            new_points = points.copy()
            new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        
        new_points = new_points - new_points.mean(axis=0) + 0.5
        
        new_ratio, _, _ = compute_ratio(new_points)
        if new_ratio > best_ratio:
            points = new_points
            best_ratio = new_ratio
    
    # Phase 3: Pairwise gradient polish with balanced forces and more iterations
    learning_rate = 0.0028
    polish_iterations = 6000
    
    for _ in range(polish_iterations):
        dists = pdist(points)
        dmin = dists.min()
        dmax = dists.max()
        
        dist_matrix = squareform(dists)
        grad = np.zeros_like(points)
        
        for i in range(n):
            for j in range(i + 1, n):
                d = dist_matrix[i, j]
                if d < 1e-10:
                    continue
                diff = points[j] - points[i]
                unit = diff / d
                
                if abs(d - dmin) < 1e-10:
                    force = 0.25 * unit
                    grad[i] -= force
                    grad[j] += force
                elif abs(d - dmax) < 1e-10:
                    force = 0.07 * unit
                    grad[i] += force
                    grad[j] -= force
        
        points += learning_rate * grad
        points = np.clip(points, 0.01, 0.99)
        points = points - points.mean(axis=0) + 0.5
    
    # Phase 4: Fine-grained hill climbing with smaller steps, more iterations and multi-point moves
    best_ratio, _, _ = compute_ratio(points)
    step_size = 0.012
    fine_hill_climb_iterations = 40000
    
    for iter_num in range(fine_hill_climb_iterations):
        # Gradually reduce step size for finer adjustments
        current_step = step_size * (1 - iter_num / fine_hill_climb_iterations) * 0.65 + step_size * 0.35
        
        # Frequently move 2-4 points for better exploration
        if iter_num % 2 == 0:
            num_moves = np.random.randint(2, 5)
            idxs = np.random.choice(n, num_moves, replace=False)
            new_points = points.copy()
            for idx in idxs:
                delta = np.random.uniform(-current_step, current_step, 2)
                new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        else:
            idx = np.random.randint(n)
            delta = np.random.uniform(-current_step, current_step, 2)
            new_points = points.copy()
            new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        
        new_points = new_points - new_points.mean(axis=0) + 0.5
        
        new_ratio, _, _ = compute_ratio(new_points)
        if new_ratio > best_ratio:
            points = new_points
            best_ratio = new_ratio
    
    # Phase 5: Super-fine micro-adjustment hill climbing for final polish
    best_ratio, _, _ = compute_ratio(points)
    step_size = 0.004
    micro_hill_climb_iterations = 30000
    
    for iter_num in range(micro_hill_climb_iterations):
        # Very gradual annealing for super-fine adjustments
        current_step = step_size * (1 - iter_num / micro_hill_climb_iterations) * 0.85 + step_size * 0.15
        
        # Try small coordinated moves on up to 3 points
        if iter_num % 5 < 2:
            num_moves = np.random.randint(1, 4)
            idxs = np.random.choice(n, num_moves, replace=False)
            new_points = points.copy()
            for idx in idxs:
                delta = np.random.uniform(-current_step, current_step, 2)
                new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        else:
            idx = np.random.randint(n)
            delta = np.random.uniform(-current_step, current_step, 2)
            new_points = points.copy()
            new_points[idx] = np.clip(new_points[idx] + delta, 0.01, 0.99)
        
        new_points = new_points - new_points.mean(axis=0) + 0.5
        
        new_ratio, _, _ = compute_ratio(new_points)
        if new_ratio > best_ratio:
            points = new_points
            best_ratio = new_ratio
    
    return points