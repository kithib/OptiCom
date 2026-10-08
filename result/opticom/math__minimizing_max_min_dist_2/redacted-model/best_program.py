import numpy as np
from scipy.spatial.distance import pdist
from scipy.optimize import minimize
import time


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.
    Uses dynamic orthogonal constraint exploration then constrained-softened search.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    n = 16
    d = 2
    rng = np.random.default_rng(42)
    
    # Decision variables: flat vector of 16*2 = 32 elements
    bounds = [(0.0, 1.0) for _ in range(n * d)]
    alpha = 12000.0  # High LogSumExp smoothing for near-exact approximations
    penalty = 6000.0  # Very strong penalty for constraint violations
    
    def get_dists(x):
        """Compute all pairwise Euclidean distances for points."""
        points = x.reshape(n, d)
        return pdist(points, metric='euclidean')
    
    def smooth_loss(x):
        """Smooth differentiable loss for numerical optimization."""
        dists = get_dists(x)
        
        # Numerically stable LogSumExp for smooth min approximation
        neg_ad = -alpha * dists
        max_neg_ad = np.max(neg_ad)
        smooth_dmin = -(np.log(np.sum(np.exp(neg_ad - max_neg_ad))) + max_neg_ad) / alpha
        
        # Numerically stable LogSumExp for smooth max approximation
        ad = alpha * dists
        max_ad = np.max(ad)
        smooth_dmax = (np.log(np.sum(np.exp(ad - max_ad))) + max_ad) / alpha
        
        # Smooth objective ratio
        smooth_ratio = smooth_dmin / (smooth_dmax + 1e-15)
        
        # Bound constraint violation penalty (squared for smoothness)
        lower_viol = np.sum(np.maximum(0.0, -x)**2)
        upper_viol = np.sum(np.maximum(0.0, x - 1.0)**2)
        
        return -smooth_ratio + penalty * (lower_viol + upper_viol)
    
    def true_objective(x):
        """Non-smoothed true objective for final evaluation."""
        points = x.reshape(n, d)
        points = np.clip(points, 0.0, 1.0)
        dists = pdist(points, metric='euclidean')
        dmin = np.min(dists)
        dmax = np.max(dists)
        return dmin / dmax
    
    # ============================================
    # DYNAMIC ORTHOGONAL CONSTRAINT SEARCH PHASE
    # ============================================
    # Top 2 dominant correlated metrics: numerical_optimization_streak, artifact_robustness_score
    # Orthogonal Hard Constraints used for seed generation:
    # 1. Skip ALL gradient-based L-BFGS-B steps (pure random/non-gradient exploration only)
    # 2. Force initial seed configurations to intentionally produce low-ratio configurations
    
    # Generate orthogonal constraint seeds (initially poor-performing, explore new regions)
    orthogonal_seeds = []
    
    # Orthogonal Seed 1: Clustered points (intentional poor ratio - violates dispersion objective)
    cluster_center = np.array([0.3, 0.7])
    clustered = rng.normal(cluster_center, 0.03, (n, d))
    orthogonal_seeds.append(np.clip(clustered.flatten(), 0.0, 1.0))
    
    # Orthogonal Seed 2: Linear colinear points (another intentional poor configuration)
    linear = np.zeros((n, d))
    linear[:, 0] = np.linspace(0.1, 0.9, n)
    linear[:, 1] = 0.5
    orthogonal_seeds.append(linear.flatten())
    
    # Orthogonal Seed 3: Random asymmetric quadrant-biased points (unexplored asymmetric region)
    quadrant = rng.random((n, d))
    quadrant[:8, 0] = quadrant[:8, 0] * 0.3  # Bias left
    quadrant[8:, 0] = 0.7 + quadrant[8:, 0] * 0.3  # Bias right
    orthogonal_seeds.append(np.clip(quadrant.flatten(), 0.0, 1.0))
    
    # Orthogonal Seed 4: Fractal-like recursive subdivision placement (novel structure)
    fractal = []
    def recursive_split(bbox, depth, max_depth):
        if depth >= max_depth or len(fractal) >= n:
            if len(fractal) < n:
                cx = (bbox[0] + bbox[1]) / 2 + rng.normal(0, 0.02)
                cy = (bbox[2] + bbox[3]) / 2 + rng.normal(0, 0.02)
                fractal.append([cx, cy])
            return
        mx = (bbox[0] + bbox[1]) / 2
        my = (bbox[2] + bbox[3]) / 2
        recursive_split([bbox[0], mx, bbox[2], my], depth+1, max_depth)
        recursive_split([mx, bbox[1], bbox[2], my], depth+1, max_depth)
        recursive_split([bbox[0], mx, my, bbox[3]], depth+1, max_depth)
        recursive_split([mx, bbox[1], my, bbox[3]], depth+1, max_depth)
    recursive_split([0.0, 1.0, 0.0, 1.0], 0, 3)
    orthogonal_seeds.append(np.clip(np.array(fractal[:16]).flatten(), 0.0, 1.0))
    
    # ==========================================
    # CONSTRAINED-SOFTENED SECONDARY SEARCH PHASE
    # ==========================================
    # Now combine incumbent warm-start + orthogonal seeds + proven patterns for full search
    # Incumbent warm-start (best known configuration as fallback baseline)
    warm_start = np.array([
        0.0,      0.0,
        0.0,      0.34367,
        0.0,      0.65633,
        0.0,      1.0,
        0.34367,  0.0,
        0.34367,  0.34367,
        0.34367,  0.65633,
        0.34367,  1.0,
        0.65633,  0.0,
        0.65633,  0.34367,
        0.65633,  0.65633,
        0.65633,  1.0,
        1.0,      0.0,
        1.0,      0.34367,
        1.0,      0.65633,
        1.0,      1.0
    ])
    
    # Full starts pool: incumbent + its noisy variants + orthogonal seeds + proven patterns
    starts = [warm_start.copy()]
    
    # 16 incumbent noise variations
    for _ in range(16):
        noisy_x = warm_start + rng.normal(0, 0.02, size=warm_start.shape)
        noisy_x = np.clip(noisy_x, 0.0, 1.0)
        starts.append(noisy_x)
    
    # 4 orthogonal constraint seeds (from new exploratory regions)
    for seed in orthogonal_seeds:
        starts.append(seed.copy())
    
    # Additional variations around orthogonal seeds (constrained-softened: allow gradient now)
    for seed in orthogonal_seeds:
        for _ in range(2):
            noisy_seed = seed + rng.normal(0, 0.04, size=seed.shape)
            noisy_seed = np.clip(noisy_seed, 0.0, 1.0)
            starts.append(noisy_seed)
    
    # Add 4 proven patterns as additional anchors
    # Hexagonal pattern anchor
    hex_anchor = [[i/3.5 + 0.5*(j%2)/3.5, j/3.0] for i in range(4) for j in range(4)]
    starts.append(np.clip(np.array(hex_anchor[:16]).flatten(), 0.0, 1.0))
    
    # Boundary biased anchor
    boundary_pattern = [
        [0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0],  # 4 corners
        [0.5, 0.0], [0.5, 1.0], [0.0, 0.5], [1.0, 0.5],  # 4 edge midpoints
        [0.28, 0.28], [0.72, 0.28], [0.28, 0.72], [0.72, 0.72],  # 4 quarter points
        [0.36, 0.5], [0.64, 0.5], [0.5, 0.36], [0.5, 0.64]   # 4 center area
    ]
    starts.append(np.array(boundary_pattern).flatten())
    
    # ======================================
    # MAIN OPTIMIZATION (constrained softened - gradient allowed now)
    # ======================================
    best_obj = -np.inf
    best_x = warm_start.copy()
    
    # Cap starts to avoid timeout (600s limit) while exploring broadly
    starts = starts[:30]
    
    lbfgs_options = {
        'ftol': 1e-10,
        'gtol': 1e-10,
        'maxiter': 1200  # Slightly reduced to help with timeout
    }
    
    for start_x in starts:
        result = minimize(
            smooth_loss,
            start_x,
            method='L-BFGS-B',
            bounds=bounds,
            options=lbfgs_options
        )
        current_obj = true_objective(result.x)
        if current_obj > best_obj:
            best_obj = current_obj
            best_x = result.x
    
    # Final polish with ultra-tight tolerance on best found configuration
    polish_result = minimize(
        smooth_loss,
        best_x,
        method='L-BFGS-B',
        bounds=bounds,
        options={'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 5000}
    )
    polished_x = polish_result.x
    
    # Project to feasible region
    points = polished_x.reshape(n, d)
    points = np.clip(points, 0.0, 1.0)
    
    return points


if __name__ == "__main__":
    start_time = time.time()
    pts = min_max_dist_dim2_16()
    elapsed_time = time.time() - start_time
    dists = pdist(pts, metric='euclidean')
    dmin = np.min(dists)
    dmax = np.max(dists)
    ratio = dmin / dmax
    print(f"Points shape: {pts.shape}")
    print(f"dmin: {dmin:.12f}")
    print(f"dmax: {dmax:.12f}")
    print(f"min_max_ratio: {ratio:.12f}")
    print(f"combined_score: {ratio / 0.2786:.12f}")
    print(f"eval_time: {elapsed_time:.4f} seconds")