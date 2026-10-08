import numpy as np
from scipy.spatial.distance import pdist


def compute_min_max_ratio(points: np.ndarray) -> float:
    """Compute the min/max distance ratio for a set of points."""
    distances = pdist(points)
    dmin = np.min(distances)
    dmax = np.max(distances)
    return dmin / dmax


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """

    n = 16
    grid_size = 4
    
    # Start with centered 4x4 grid (better than edge-to-edge grid)
    points = []
    for i in range(grid_size):
        for j in range(grid_size):
            x = (i + 0.5) / grid_size
            y = (j + 0.5) / grid_size
            points.append([x, y])
    
    points = np.array(points)
    best_ratio = compute_min_max_ratio(points)
    best_points = points.copy()
    
    # Enhanced hill climbing with dual perturbation strategy
    np.random.seed(42)
    step_size = 0.012
    iterations = 80000
    
    for _ in range(iterations):
        idx = np.random.randint(0, n)
        if np.random.random() < 0.7:
            delta = np.random.uniform(-step_size, step_size, size=2)
        else:
            delta = np.random.uniform(-step_size * 0.5, step_size * 0.5, size=2)
        new_points = points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, 0.045, 0.955)
        new_ratio = compute_min_max_ratio(new_points)
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_points = new_points.copy()
            points = new_points.copy()
    
    # Enhanced simulated annealing with better exploration
    np.random.seed(123)
    current_ratio = compute_min_max_ratio(points)
    temp = 0.01
    cooling_rate = 0.9997
    
    for _ in range(150000):
        idx = np.random.randint(0, n)
        if np.random.random() < 0.6:
            delta = np.random.uniform(-step_size * 0.6, step_size * 0.6, size=2)
        elif np.random.random() < 0.9:
            delta = np.random.uniform(-step_size * 0.3, step_size * 0.3, size=2)
        else:
            delta = np.random.uniform(-step_size * 0.1, step_size * 0.1, size=2)
        new_points = points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, 0.042, 0.958)
        new_ratio = compute_min_max_ratio(new_points)
        
        if new_ratio > current_ratio:
            current_ratio = new_ratio
            points = new_points.copy()
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
        else:
            prob = np.exp((new_ratio - current_ratio) / temp) if temp > 1e-12 else 0.0
            if np.random.random() < prob:
                current_ratio = new_ratio
                points = new_points.copy()
        
        temp *= cooling_rate
    
    # Multi-phase polishing with decreasing step sizes
    polish_configs = [
        (0.004, 40000, 777),
        (0.002, 60000, 999),
        (0.001, 80000, 333),
        (0.0005, 100000, 111),
        (0.00025, 120000, 555),
        (0.0001, 150000, 222),
        (0.00005, 180000, 666),
        (0.000025, 200000, 888),
        (0.00001, 250000, 444),
        (0.000005, 300000, 1212),
    ]
    
    for step, iters, seed in polish_configs:
        np.random.seed(seed)
        for _ in range(iters):
            idx = np.random.randint(0, n)
            delta = np.random.uniform(-step, step, size=2)
            new_points = best_points.copy()
            new_points[idx] += delta
            new_points = np.clip(new_points, 0.04, 0.96)
            new_ratio = compute_min_max_ratio(new_points)
            if new_ratio > best_ratio:
                best_ratio = new_ratio
                best_points = new_points.copy()
    
    # Quantum-tunneling inspired refinement with adaptive acceptance
    np.random.seed(31337)
    quantum_step = 0.0000025
    quantum_temp = 2e-15
    for _ in range(350000):
        idx = np.random.randint(0, n)
        delta = np.random.uniform(-quantum_step, quantum_step, size=2)
        new_points = best_points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, 0.04, 0.96)
        new_ratio = compute_min_max_ratio(new_points)
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_points = new_points.copy()
        elif np.random.random() < np.exp((new_ratio - best_ratio) / quantum_temp):
            best_ratio = new_ratio
            best_points = new_points.copy()
    
    # Final exploration phase accepting equal ratios
    np.random.seed(42069)
    final_step = 0.000001
    for _ in range(400000):
        idx = np.random.randint(0, n)
        delta = np.random.uniform(-final_step, final_step, size=2)
        new_points = best_points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, 0.04, 0.96)
        new_ratio = compute_min_max_ratio(new_points)
        if new_ratio >= best_ratio:
            best_ratio = new_ratio
            best_points = new_points.copy()
    
    # Ultra-final micro-adjustment phase
    np.random.seed(7355608)
    ultra_step = 0.0000005
    for _ in range(500000):
        idx = np.random.randint(0, n)
        delta = np.random.uniform(-ultra_step, ultra_step, size=2)
        new_points = best_points.copy()
        new_points[idx] += delta
        new_points = np.clip(new_points, 0.04, 0.96)
        new_ratio = compute_min_max_ratio(new_points)
        if new_ratio > best_ratio:
            best_ratio = new_ratio
            best_points = new_points.copy()
    
    return best_points