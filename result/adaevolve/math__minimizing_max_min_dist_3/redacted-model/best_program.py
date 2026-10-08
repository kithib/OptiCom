import numpy as np


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """

    n = 14
    d = 3

    # Phase 1: Start with structured configuration (cube vertices + face centers)
    vertices = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]], dtype=float)
    
    face_centers = np.array([[0.5, 0.5, 0], [0.5, 0.5, 1],
                              [0.5, 0, 0.5], [0.5, 1, 0.5],
                              [0, 0.5, 0.5], [1, 0.5, 0.5]], dtype=float)
    
    points = np.vstack([vertices, face_centers])
    
    # Center and project to unit sphere (bounds max distance)
    points = points - np.mean(points, axis=0)
    points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Phase 2: Pairwise repulsion optimization - fully vectorized for consistency
    np.random.seed(42)
    for _ in range(250):
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        np.fill_diagonal(dist_sq, np.inf)
        dist = np.sqrt(dist_sq)
        # Calculate force magnitude for all pairs - stronger initial force
        force_mag = 0.018 / (dist ** 3 + 1e-6)
        # Weight deltas and sum for gradient (symmetric)
        grad = np.sum(force_mag[:, :, np.newaxis] * deltas, axis=1)
        points += grad
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Phase 3: Gradient ascent on min/max ratio with cosine annealing
    base_lr = 0.018
    iterations = 750
    
    for iter_idx in range(iterations):
        # Cosine annealing for smoother, more robust learning rate decay
        learning_rate = base_lr * 0.5 * (1 + np.cos(np.pi * iter_idx / iterations))
        
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        
        # Increase minimum distance - expanded top-k pairs with exponential weighting
        np.fill_diagonal(dist_sq, np.inf)
        sorted_indices = np.dstack(np.unravel_index(np.argsort(dist_sq, axis=None), dist_sq.shape))[0]
        k = min(25, n * (n - 1) // 2)
        for idx in range(k):
            i, j = sorted_indices[idx]
            if i < j:
                delta_vec = deltas[i, j, :]
                d = np.sqrt(dist_sq[i, j])
                if d > 0:
                    grad = delta_vec / d
                    weight = 0.9 * np.exp(-0.12 * idx)
                    points[i, :] += learning_rate * weight * grad
                    points[j, :] -= learning_rate * weight * grad
        
        # Decrease maximum distance - handle top 10 farthest pairs with exponential decay
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        np.fill_diagonal(dist_sq, -np.inf)
        sorted_max_indices = np.dstack(np.unravel_index(np.argsort(-dist_sq, axis=None), dist_sq.shape))[0]
        k_max = min(12, n * (n - 1) // 2)
        for idx in range(k_max):
            i_max, j_max = sorted_max_indices[idx]
            if i_max < j_max:
                max_dist = np.sqrt(dist_sq[i_max, j_max])
                grad_max = deltas[i_max, j_max, :] / max_dist if max_dist > 0 else 0
                weight = np.exp(-0.15 * idx)
                points[i_max, :] -= learning_rate * weight * grad_max
                points[j_max, :] += learning_rate * weight * grad_max
        
        # Keep points bounded to unit sphere during optimization
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
    
    # Phase 4: Enhanced gradient ascent phase with expanded pairs and balanced weighting
    ext_base_lr = 0.014
    ext_iterations = 400
    
    for iter_idx in range(ext_iterations):
        # Smoother learning rate with cosine annealing
        learning_rate = ext_base_lr * 0.5 * (1 + np.cos(np.pi * iter_idx / ext_iterations))
        
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        
        # Increase minimum distance - target top 25 closest pairs
        np.fill_diagonal(dist_sq, np.inf)
        sorted_indices = np.dstack(np.unravel_index(np.argsort(dist_sq, axis=None), dist_sq.shape))[0]
        k = min(25, n * (n - 1) // 2)
        for idx in range(k):
            i, j = sorted_indices[idx]
            if i < j:
                delta_vec = deltas[i, j, :]
                d = np.sqrt(dist_sq[i, j])
                if d > 0:
                    grad = delta_vec / d
                    weight = 0.85 * np.exp(-0.1 * idx)
                    points[i, :] += learning_rate * weight * grad
                    points[j, :] -= learning_rate * weight * grad
        
        # Decrease maximum distance - handle top 12 farthest pairs
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        np.fill_diagonal(dist_sq, -np.inf)
        sorted_max_indices = np.dstack(np.unravel_index(np.argsort(-dist_sq, axis=None), dist_sq.shape))[0]
        k_max = min(12, n * (n - 1) // 2)
        for idx in range(k_max):
            i_max, j_max = sorted_max_indices[idx]
            if i_max < j_max:
                max_dist = np.sqrt(dist_sq[i_max, j_max])
                grad_max = deltas[i_max, j_max, :] / max_dist if max_dist > 0 else 0
                weight = np.exp(-0.12 * idx)
                points[i_max, :] -= learning_rate * weight * grad_max
                points[j_max, :] += learning_rate * weight * grad_max
        
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.where(norms > 0, norms, 1)
    
    # Phase 5: KNN polishing - refined to focus on improving the critical closest distances without disrupting global arrangement
    k_neighbors = 8
    polish_iterations = 350
    for polish_idx in range(polish_iterations):
        polish_lr = 0.012 * 0.5 * (1 + np.cos(np.pi * polish_idx / polish_iterations))
        diffs = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.linalg.norm(diffs, axis=2)
        np.fill_diagonal(dists, np.inf)
        
        # Get indices of k closest neighbors for each point
        closest_indices = np.argsort(dists, axis=1)[:, :k_neighbors]
        
        # Apply repulsion from k closest neighbors to each point with distance-aware force to avoid large jumps
        for i in range(n):
            for neighbor_rank, j in enumerate(closest_indices[i]):
                diff = points[i] - points[j]
                dist = dists[i, j]
                if dist > 0:
                    force = (polish_lr * 0.7 * np.exp(-0.2 * neighbor_rank)) * diff / (dist ** 2 + 1e-6)
                    points[i] += force
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Phase 6: Added explicit dmin/dmax gradient ascent phase - direct ratio optimization with expanded pairs
    ratio_lr = 0.005
    ratio_iterations = 500
    for iter_idx in range(ratio_iterations):
        learning_rate = ratio_lr * 0.5 * (1 + np.cos(np.pi * iter_idx / ratio_iterations))
        
        deltas = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist_sq = np.sum(deltas ** 2, axis=2)
        dist_matrix = np.sqrt(np.maximum(dist_sq, 1e-12))
        
        # Find min and max distances
        np.fill_diagonal(dist_matrix, np.inf)
        dmin = np.min(dist_matrix)
        np.fill_diagonal(dist_matrix, -np.inf)
        dmax = np.max(dist_matrix)
        
        if dmin < 1e-6 or dmax < 1e-6:
            continue
        
        # Compute gradient for closest pairs - expanded to top 12 closest pairs with better weighting
        np.fill_diagonal(dist_matrix, np.inf)
        min_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                min_pairs.append((dist_matrix[i, j], i, j))
        min_pairs.sort()
        for pair_idx, (d_val, i, j) in enumerate(min_pairs[:12]):
            # Increase dmin for top closest pairs with exponential weight decay
            diff = points[i] - points[j]
            grad = diff / (d_val * dmax)
            weight = np.exp(-0.1 * pair_idx)
            points[i] += learning_rate * 0.5 * weight * grad
            points[j] -= learning_rate * 0.5 * weight * grad
        
        # Compute gradient for farthest pairs - expanded to top 10 farthest pairs with better weighting
        np.fill_diagonal(dist_matrix, -np.inf)
        max_pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                max_pairs.append((dist_matrix[i, j], i, j))
        max_pairs.sort(reverse=True)
        for pair_idx, (d_val, i, j) in enumerate(max_pairs[:10]):
            # Decrease dmax for top farthest pairs with exponential weight decay
            diff = points[i] - points[j]
            grad = diff * dmin / (d_val ** 3)
            weight = np.exp(-0.12 * pair_idx)
            points[i] -= learning_rate * 0.5 * weight * grad
            points[j] += learning_rate * 0.5 * weight * grad
        
        points = points / np.linalg.norm(points, axis=1, keepdims=True)
    
    # Final normalization to [0, 1]^3 cube
    points -= points.min(axis=0)
    points /= points.max(axis=0)
    
    return points