import numpy as np
from numba import jit

@jit(nopython=True)
def _compute_ratio(points):
    """Numba-accelerated ratio computation for maximum efficiency."""
    n = len(points)
    dmin = np.inf
    dmax = -np.inf
    for i in range(n):
        for j in range(i + 1, n):
            dx = points[i, 0] - points[j, 0]
            dy = points[i, 1] - points[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < dmin:
                dmin = dist
            if dist > dmax:
                dmax = dist
    return dmin / dmax


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    np.random.seed(42)
    
    # Start with an optimally staggered hexagonal 4x4 grid
    grid_size = 4
    x = np.linspace(0, 1, grid_size)
    y = np.linspace(0, 1, grid_size)
    xv, yv = np.meshgrid(x, y)
    # Apply optimal hexagonal row staggering (based on geometry)
    row_height = 1.0 / (grid_size - 1)
    hex_offset = 0.5 / (grid_size - 1)
    for i in range(grid_size):
        if i % 2 == 1:
            xv[i] += hex_offset
    xv = np.clip(xv, 0, 1)
    points = np.column_stack([xv.ravel(), yv.ravel()])
    
    # Calculate initial objective with numba
    best_ratio = _compute_ratio(points)
    best_points = points.copy()
    
    # Enhanced optimization with expanded iterations and adaptive strategy
    step_size = 0.035
    iterations = 4500
    
    for iter_idx in range(iterations):
        improved = False
        
        # Try perturbing each point with comprehensive 24-directional moves + random
        for i in range(16):
            # Generate directions at 15-degree intervals (24 compass directions)
            directions = []
            for angle in np.linspace(0, 2 * np.pi, 24, endpoint=False):
                directions.append((step_size * np.cos(angle), step_size * np.sin(angle)))
            # Also add half and quarter step versions
            for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
                directions.append((step_size * 0.5 * np.cos(angle), step_size * 0.5 * np.sin(angle)))
                directions.append((step_size * 0.25 * np.cos(angle), step_size * 0.25 * np.sin(angle)))
            
            for dx, dy in directions:
                new_points = best_points.copy()
                new_points[i] = np.clip(new_points[i] + [dx, dy], 0, 1)
                ratio = _compute_ratio(new_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = new_points.copy()
                    improved = True
            
            # Additional random perturbation with varying sigma values
            for sigma in [step_size * 0.3, step_size * 0.7, step_size * 1.2]:
                new_points = best_points.copy()
                new_points[i] = np.clip(new_points[i] + np.random.randn(2) * sigma, 0, 1)
                ratio = _compute_ratio(new_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = new_points.copy()
                    improved = True
        
        # MCTS-guided rollouts - explore several moves deep
        if not improved or iter_idx % 25 == 0:
            for _ in range(5):
                candidate_points = best_points.copy()
                # 3-step lookahead rollout
                for rollout_step in range(3):
                    # Pick random point to move
                    i = np.random.randint(16)
                    # Pick random direction
                    angle = np.random.uniform(0, 2 * np.pi)
                    magnitude = step_size * np.random.uniform(0.3, 1.5)
                    candidate_points[i] = np.clip(
                        candidate_points[i] + [magnitude * np.cos(angle), magnitude * np.sin(angle)],
                        0, 1
                    )
                ratio = _compute_ratio(candidate_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = candidate_points.copy()
                    improved = True
        
        # Coordinated pair and triple perturbation for deeper exploration
        if not improved or iter_idx % 40 == 0:
            for i in range(16):
                for j in range(i + 1, 16):
                    for _ in range(2):
                        new_points = best_points.copy()
                        new_points[i] = np.clip(new_points[i] + np.random.randn(2) * step_size * 0.6, 0, 1)
                        new_points[j] = np.clip(new_points[j] + np.random.randn(2) * step_size * 0.6, 0, 1)
                        ratio = _compute_ratio(new_points)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_points = new_points.copy()
                            improved = True
        
        # Try swapping pairs then triple perturbation for escaping local optima
        if not improved and iter_idx % 70 == 0:
            for i in range(16):
                for j in range(i + 1, 16):
                    new_points = best_points.copy()
                    new_points[i], new_points[j] = new_points[j].copy(), new_points[i].copy()
                    # Multi-point perturbation after swap
                    for k in np.random.choice(16, min(4, 16), replace=False):
                        new_points[k] = np.clip(new_points[k] + np.random.randn(2) * step_size * 0.4, 0, 1)
                    ratio = _compute_ratio(new_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = new_points.copy()
                        improved = True
        
        # Collective micro-adjustment pulse with varying intensities
        if not improved and iter_idx % 35 == 0:
            for intensity in [0.15, 0.3, 0.5]:
                new_points = best_points + np.random.randn(16, 2) * step_size * intensity
                new_points = np.clip(new_points, 0, 1)
                ratio = _compute_ratio(new_points)
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_points = new_points.copy()
                    improved = True
        
        # Adaptive step annealing with refined decay rates
        if improved:
            step_size *= 0.9992
        else:
            step_size *= 0.993
        if step_size < 0.00002:
            break
    
    # Enhanced final fine-tuning pass with nested refinement
    for power in [1, 2, 3]:
        final_step = 0.002 / (10 ** (power - 1))
        for _ in range(150 // power):
            improved = False
            for i in range(16):
                # 16-directional micro-moves
                for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
                    dx = final_step * np.cos(angle)
                    dy = final_step * np.sin(angle)
                    new_points = best_points.copy()
                    new_points[i] = np.clip(new_points[i] + [dx, dy], 0, 1)
                    ratio = _compute_ratio(new_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = new_points.copy()
                        improved = True
            if not improved:
                break
    
    # Ultimate coordinate descent to polish boundaries
    for _ in range(50):
        improved = False
        for i in range(16):
            for coord in [0, 1]:  # x, then y
                for delta in [-1e-5, 1e-5]:
                    new_points = best_points.copy()
                    new_points[i, coord] = np.clip(new_points[i, coord] + delta, 0, 1)
                    ratio = _compute_ratio(new_points)
                    if ratio > best_ratio:
                        best_ratio = ratio
                        best_points = new_points.copy()
                        improved = True
        if not improved:
            break
    
    return best_points