# EVOLVE-BLOCK-START
import numpy as np


def _compute_ratio(points: np.ndarray) -> float:
    """Compute the min/max distance ratio for a set of points."""
    n = len(points)
    # Compute pairwise squared distances for efficiency
    diff = points[:, np.newaxis] - points
    sq_dist = np.sum(diff ** 2, axis=2)
    # Get upper triangular indices to avoid duplicates and i==j
    i, j = np.triu_indices(n, k=1)
    sq_distances = sq_dist[i, j]
    # Find min and max squared distances, take sqrt of ratio
    min_sq = np.min(sq_distances)
    max_sq = np.max(sq_distances)
    return np.sqrt(min_sq / max_sq)


def _initialize_14_point_arrangement() -> np.ndarray:
    """Initialize a symmetric 14-point arrangement in 3D."""
    # 1) 6 vertices of a regular octahedron (axes-aligned)
    octahedron = np.array([
        [1, 0, 0], [-1, 0, 0],
        [0, 1, 0], [0, -1, 0],
        [0, 0, 1], [0, 0, -1]
    ])
    
    # 2) 8 vertices of a cube with tuned scale factor for better initial balance
    s = 0.85  # Further refined relative scale to balance initial inter-point distances
    cube = np.array([
        [-s, -s, -s], [-s, -s,  s],
        [-s,  s, -s], [-s,  s,  s],
        [ s, -s, -s], [ s, -s,  s],
        [ s,  s, -s], [ s,  s,  s]
    ])
    
    combined = np.vstack([octahedron, cube])
    # Normalize initial points to unit sphere to bound the search space
    return combined / np.linalg.norm(combined, axis=1, keepdims=True)


def _gradient_ascent_optimize(points: np.ndarray, iterations: int = 1000, initial_step: float = 0.018, final_step: float = 0.0015) -> np.ndarray:
    """Perform gradient ascent with step annealing and structured perturbations to maximize the min/max distance ratio."""
    best_points = points.copy()
    best_ratio = _compute_ratio(best_points)
    
    np.random.seed(42)  # Fixed seed for reproducibility
    
    for iter_idx in range(iterations):
        # Cosine annealing for learning rate (smoother decay than linear)
        step = final_step + 0.5 * (initial_step - final_step) * (1 + np.cos(np.pi * iter_idx / iterations))
        
        # Compute all pairwise squared distances
        n = len(points)
        diff = points[:, np.newaxis] - points
        sq_dist = np.sum(diff ** 2, axis=2)
        i, j = np.triu_indices(n, k=1)
        sq_distances = sq_dist[i, j]
        
        # Find closest pair (to repel) and farthest pair (to attract)
        min_idx = np.argmin(sq_distances)
        max_idx = np.argmax(sq_distances)
        i_min, j_min = i[min_idx], j[min_idx]
        i_max, j_max = i[max_idx], j[max_idx]
        
        # Create perturbation vector
        perturbation = np.zeros_like(points)
        
        # Repel closest pair strongly
        diff_min = points[i_min] - points[j_min]
        norm_min = np.linalg.norm(diff_min)
        if norm_min > 1e-8:
            force_min = diff_min / norm_min * step * 1.3  # Increased repulsion force
            perturbation[i_min] += force_min
            perturbation[j_min] -= force_min
        
        # Attract farthest pair weakly
        diff_max = points[i_max] - points[j_max]
        norm_max = np.linalg.norm(diff_max)
        if norm_max > 1e-8:
            force_max = diff_max / norm_max * step * 0.5
            perturbation[i_max] -= force_max
            perturbation[j_max] += force_max
        
        # Add small noise for exploration - slightly reduced for finer tuning
        perturbation += np.random.normal(0, step * 0.12, points.shape)
        
        # Apply perturbation and normalize
        candidate = points + perturbation
        candidate = candidate / np.linalg.norm(candidate, axis=1, keepdims=True)
        
        candidate_ratio = _compute_ratio(candidate)
        if candidate_ratio > best_ratio:
            best_ratio = candidate_ratio
            best_points = candidate.copy()
            points = candidate
        elif np.random.random() < 0.10:  # Slightly reduced probabilistic acceptance
            points = candidate
    
    return best_points


def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (14,3) containing the (x,y,z) coordinates of the 14 points.

    """
    # Initialize with a symmetric 14-point arrangement (6 octahedron + 8 cube vertices)
    initial = _initialize_14_point_arrangement()
    
    # Optimize the arrangement with improved gradient ascent
    optimized = _gradient_ascent_optimize(initial)
    
    return optimized


# EVOLVE-BLOCK-END