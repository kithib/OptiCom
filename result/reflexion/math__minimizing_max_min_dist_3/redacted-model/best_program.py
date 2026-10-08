import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_score(points):
    """Helper to compute min/max ratio for monitoring"""
    dists = pdist(points)
    return np.min(dists) / np.max(dists)


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    np.random.seed(42)
    
    # Enhanced near-optimal antipodal starting arrangement with better initial symmetry
    points = np.array([
        [-0.780474, -0.320598, -0.539108],
        [-0.776928, -0.101883,  0.620849],
        [-0.702027,  0.655516, -0.275544],
        [-0.673782,  0.446158,  0.590139],
        [-0.581582, -0.782142, -0.223221],
        [-0.561987, -0.579936,  0.590139],
        [-0.158433, -0.275118, -0.948925],
        [-0.137078,  0.907224, -0.398238],
        [ 0.137078, -0.907224,  0.398238],
        [ 0.158433,  0.275118,  0.948925],
        [ 0.561987,  0.579936, -0.590139],
        [ 0.581582,  0.782142,  0.223221],
        [ 0.673782, -0.446158, -0.590139],
        [ 0.702027, -0.655516,  0.275544]
    ])
    
    # Normalize to unit sphere to bound the space
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    best_points = points.copy()
    best_score = compute_score(points)
    
    # Phase 1: Exploration - Enhanced with better weight decay for 14 closest pairs
    initial_lr = 0.0264
    num_iterations_phase1 = 765
    momentum = np.zeros_like(points)
    momentum_decay = 0.895
    
    for iteration in range(num_iterations_phase1):
        progress = iteration / num_iterations_phase1
        learning_rate = initial_lr * (1 - progress) * 0.787 + initial_lr * 0.107
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process ALL 14 closest pairs with optimized exponential weight decay
        flat_dists = dists.flatten()
        for pair_idx in range(14):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.656 * (0.878 ** pair_idx)
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer with optimized weight
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 1.000 * direction
            gradient[j_max] -= 1.000 * direction
        
        momentum = momentum_decay * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    points = best_points.copy()
    
    # Phase 2: Fine-tuning - Enhanced balanced phase
    initial_lr_fine = 0.00800
    num_iterations_phase2 = 1272
    momentum = np.zeros_like(points)
    momentum_decay_fine = 0.761
    
    for iteration in range(num_iterations_phase2):
        progress = iteration / num_iterations_phase2
        learning_rate = initial_lr_fine * (1 - progress) * 0.847 + initial_lr_fine * 0.066
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 12 closest pairs with optimized weights
        flat_dists = dists.flatten()
        for pair_idx in range(12):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.422 * (0.941 ** pair_idx)
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.878 * direction
            gradient[j_max] -= 0.878 * direction
        
        momentum = momentum_decay_fine * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    points = best_points.copy()
    
    # Phase 3: Micro-adjustments - Enhanced polishing phase
    initial_lr_micro = 0.00260
    num_iterations_phase3 = 877
    momentum = np.zeros_like(points)
    
    for iteration in range(num_iterations_phase3):
        progress = iteration / num_iterations_phase3
        learning_rate = initial_lr_micro * (1 - progress) * 0.908 + initial_lr_micro * 0.0345
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 10 closest pairs gently
        flat_dists = dists.flatten()
        for pair_idx in range(10):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.167 - pair_idx * 0.037
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.828 * direction
            gradient[j_max] -= 0.828 * direction
        
        momentum = 0.728 * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    points = best_points.copy()
    
    # Phase 4: Extra micro-polishing phase - Enhanced ultra gentle adjustments
    initial_lr_extra = 0.00110
    num_iterations_phase4 = 777
    momentum = np.zeros_like(points)
    
    for iteration in range(num_iterations_phase4):
        progress = iteration / num_iterations_phase4
        learning_rate = initial_lr_extra * (1 - progress) * 0.952 + initial_lr_extra * 0.0135
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 8 closest pairs extremely gently
        flat_dists = dists.flatten()
        for pair_idx in range(8):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.048 - pair_idx * 0.015
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer with very light optimized weight
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.728 * direction
            gradient[j_max] -= 0.728 * direction
        
        momentum = 0.628 * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    # Phase 5: Final nano-polishing for ultimate convergence - Enhanced
    points = best_points.copy()
    initial_lr_nano = 0.000186
    num_iterations_phase5 = 660
    momentum = np.zeros_like(points)
    
    for iteration in range(num_iterations_phase5):
        progress = iteration / num_iterations_phase5
        learning_rate = initial_lr_nano * (1 - progress) * 0.967 + initial_lr_nano * 0.0080
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 7 closest pairs with nano adjustments
        flat_dists = dists.flatten()
        for pair_idx in range(7):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.023 - pair_idx * 0.0075
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer with ultra light weight
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.608 * direction
            gradient[j_max] -= 0.608 * direction
        
        momentum = 0.508 * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    # Phase 6: Extra pico-polishing phase for marginal additional improvement
    points = best_points.copy()
    initial_lr_pico = 0.0000575
    num_iterations_phase6 = 440
    momentum = np.zeros_like(points)
    
    for iteration in range(num_iterations_phase6):
        progress = iteration / num_iterations_phase6
        learning_rate = initial_lr_pico * (1 - progress) * 0.984 + initial_lr_pico * 0.0035
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 5 closest pairs with pico adjustments
        flat_dists = dists.flatten()
        for pair_idx in range(5):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.010 - pair_idx * 0.003
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer with ultra light weight
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.438 * direction
            gradient[j_max] -= 0.438 * direction
        
        momentum = 0.438 * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    # Phase 7: Femto-polishing phase for ultra-marginal additional improvement
    points = best_points.copy()
    initial_lr_femto = 0.0000165
    num_iterations_phase7 = 350
    momentum = np.zeros_like(points)
    
    for iteration in range(num_iterations_phase7):
        progress = iteration / num_iterations_phase7
        learning_rate = initial_lr_femto * (1 - progress) * 0.99 + initial_lr_femto * 0.002
        
        dists = squareform(pdist(points))
        np.fill_diagonal(dists, np.inf)
        
        gradient = np.zeros_like(points)
        
        # Process top 3 closest pairs with femto adjustments
        flat_dists = dists.flatten()
        for pair_idx in range(3):
            min_idx = np.argmin(flat_dists)
            i_min, j_min = np.unravel_index(min_idx, dists.shape)
            
            direction = points[j_min] - points[i_min]
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                direction /= norm
                weight = 1.0035 - pair_idx * 0.001
                gradient[i_min] -= weight * direction
                gradient[j_min] += weight * direction
                flat_dists[min_idx] = np.inf
        
        # Pull farthest pairs closer with ultra light weight
        i_max, j_max = np.unravel_index(np.argmax(dists), dists.shape)
        direction = points[j_max] - points[i_max]
        norm = np.linalg.norm(direction)
        if norm > 1e-8:
            direction /= norm
            gradient[i_max] += 0.33 * direction
            gradient[j_max] -= 0.33 * direction
        
        momentum = 0.33 * momentum + learning_rate * gradient
        points += momentum
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
        
        current_score = compute_score(points)
        if current_score > best_score:
            best_score = current_score
            best_points = points.copy()
    
    return best_points