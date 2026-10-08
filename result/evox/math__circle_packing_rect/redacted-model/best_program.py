import numpy as np

def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 21
    
    # Co-optimized rectangle aspect ratio based on phase analysis
    width = 1.128
    height = 0.872  # width + height = 2 as required
    
    # Initialize with staggered hex grid - asymmetric 6-row pattern (4+3+4+3+4+3=21)
    rows = 6
    cols_per_row = [4, 3, 4, 3, 4, 3]
    
    # Optimal initial radius for dense hex packing
    base_r = 0.0985
    
    circles = np.zeros((n, 3))
    idx = 0
    
    for row in range(rows):
        cols = cols_per_row[row]
        row_width = cols * 2 * base_r
        # Dynamic centering offset per row for better space utilization
        if row % 2 == 1:
            row_start = (width - (cols - 1) * 2 * base_r * 0.97) / 2
        else:
            row_start = (width - cols * 2 * base_r * 0.97) / 2
        
        for col in range(cols):
            x = row_start + col * 2 * base_r * 0.97
            y = base_r * 0.965 + row * 2 * base_r * (np.sqrt(3) / 2) * 0.975
            circles[idx] = [x, y, base_r]
            idx += 1
    
    # Phase 1: Aggressive expansion with adaptive repulsion
    iterations = 3500
    
    for iteration in range(iterations):
        learning_rate = 0.015 * np.exp(-iteration / 2800) + 0.0012
        
        boundary_mins = np.minimum(circles[:, 0], 
                                   np.minimum(width - circles[:, 0],
                                             np.minimum(circles[:, 1], height - circles[:, 1])))
        
        min_circle_dists = np.full(n, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) - circles[i, 2] - circles[j, 2]
                min_circle_dists[i] = min(min_circle_dists[i], dist)
                min_circle_dists[j] = min(min_circle_dists[j], dist)
        
        min_clearance = np.minimum(boundary_mins, min_circle_dists)
        delta_r = learning_rate * min_clearance
        circles[:, 2] = np.maximum(circles[:, 2] + delta_r, 0.001)
        
        repulsion_strength = 0.25 * (1.0 - iteration / iterations * 0.55)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist * 1.005 and dist > 1e-8:
                    overlap = min_dist * 1.005 - dist
                    move = repulsion_strength * overlap * 0.5 / dist
                    circles[i, 0] -= dx * move
                    circles[i, 1] -= dy * move
                    circles[j, 0] += dx * move
                    circles[j, 1] += dy * move
        
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], width - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], height - circles[:, 2])
    
    # Phase 2: Equalization phase with enhanced small circle boosting
    iterations = 7000
    
    for iteration in range(iterations):
        learning_rate = 0.012 * np.exp(-iteration / 4000) + 0.0005
        
        boundary_mins = np.minimum(circles[:, 0], 
                                   np.minimum(width - circles[:, 0],
                                             np.minimum(circles[:, 1], height - circles[:, 1])))
        
        min_circle_dists = np.full(n, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) - circles[i, 2] - circles[j, 2]
                min_circle_dists[i] = min(min_circle_dists[i], dist)
                min_circle_dists[j] = min(min_circle_dists[j], dist)
        
        min_clearance = np.minimum(boundary_mins, min_circle_dists)
        
        # Enhanced boosting using quartile statistics
        sorted_r = np.sort(circles[:, 2])
        q1 = sorted_r[5]
        q2 = sorted_r[10]
        q3 = sorted_r[15]
        
        # Multi-level boosting targeting bottom quartile circles most aggressively
        boost = np.where(circles[:, 2] < q1, 1.5,
                        np.where(circles[:, 2] < q2, 1.25,
                                np.where(circles[:, 2] < q3, 1.05, 0.85)))
        
        delta_r = learning_rate * min_clearance * boost
        circles[:, 2] = np.maximum(circles[:, 2] + delta_r, 0.001)
        
        repulsion_strength = 0.185 * (1.0 - iteration / iterations * 0.6)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    # Size-dependent repulsion: larger circles move less
                    size_mod = (circles[i, 2] + circles[j, 2]) / (2 * np.mean(circles[:, 2]) + 1e-8)
                    move = repulsion_strength * overlap * 0.5 / dist / np.sqrt(size_mod + 0.5)
                    circles[i, 0] -= dx * move
                    circles[i, 1] -= dy * move
                    circles[j, 0] += dx * move
                    circles[j, 1] += dy * move
        
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], width - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], height - circles[:, 2])
    
    # Phase 3: Polish phase with adaptive momentum and clearance-aware growth
    iterations = 3200
    momentum = np.zeros((n, 2))
    prev_grad = np.zeros((n, 2))
    
    for iteration in range(iterations):
        learning_rate = 0.001 + 0.0009 * (1 - iteration / iterations)
        
        boundary_mins = np.minimum(circles[:, 0], 
                                   np.minimum(width - circles[:, 0],
                                             np.minimum(circles[:, 1], height - circles[:, 1])))
        
        min_circle_dists = np.full(n, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) - circles[i, 2] - circles[j, 2]
                min_circle_dists[i] = min(min_circle_dists[i], dist)
                min_circle_dists[j] = min(min_circle_dists[j], dist)
        
        min_clearance = np.minimum(boundary_mins, min_circle_dists)
        
        # Clearance-aware size factor with max-min normalization
        min_cl = np.min(min_clearance)
        max_cl = np.max(min_clearance)
        if max_cl - min_cl > 1e-10:
            normalized_cl = (min_clearance - min_cl) / (max_cl - min_cl)
        else:
            normalized_cl = np.zeros(n)
        size_factor = 0.6 + 1.0 * normalized_cl
        
        delta_r = learning_rate * min_clearance * size_factor
        circles[:, 2] = np.maximum(circles[:, 2] + delta_r, 0.001)
        
        repulsion_strength = 0.145
        momentum_factor = 0.42
        grad = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    move = repulsion_strength * overlap * 0.5 / dist
                    grad[i, 0] -= dx * move
                    grad[i, 1] -= dy * move
                    grad[j, 0] += dx * move
                    grad[j, 1] += dy * move
        
        # Enhanced Nesterov momentum with adaptive coefficient
        momentum = momentum_factor * momentum + grad + 0.18 * (grad - prev_grad)
        circles[:, 0] += momentum[:, 0]
        circles[:, 1] += momentum[:, 1]
        prev_grad = grad.copy()
        
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], width - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], height - circles[:, 2])
    
    # Phase 4: Final sub-gradient micro-optimization with precision
    iterations = 1500
    velocity = np.zeros((n, 2))
    
    for iteration in range(iterations):
        phase = iteration / iterations
        learning_rate = 0.00035 + 0.00025 * np.cos(phase * np.pi * 0.5)
        
        boundary_mins = np.minimum(circles[:, 0], 
                                   np.minimum(width - circles[:, 0],
                                             np.minimum(circles[:, 1], height - circles[:, 1])))
        
        min_circle_dists = np.full(n, np.inf)
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy) - circles[i, 2] - circles[j, 2]
                min_circle_dists[i] = min(min_circle_dists[i], dist)
                min_circle_dists[j] = min(min_circle_dists[j], dist)
        
        min_clearance = np.minimum(boundary_mins, min_circle_dists)
        
        # Size harmonization - reduce coefficient of variation
        r_mean = np.mean(circles[:, 2])
        r_std = np.std(circles[:, 2])
        cv = r_std / (r_mean + 1e-8)
        eq_factor = 1.0 + 1.2 * cv * (r_mean - circles[:, 2]) / (r_mean + 1e-8)
        eq_factor = np.clip(eq_factor, 0.6, 1.6)
        
        delta_r = learning_rate * min_clearance * eq_factor
        circles[:, 2] = np.maximum(circles[:, 2] + delta_r, 0.001)
        
        # Soft repulsion with damping
        repulsion_strength = 0.075
        damping = 0.88
        grad = np.zeros((n, 2))
        
        for i in range(n):
            for j in range(i + 1, n):
                dx = circles[j, 0] - circles[i, 0]
                dy = circles[j, 1] - circles[i, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                
                if dist < min_dist and dist > 1e-8:
                    overlap = min_dist - dist
                    move = repulsion_strength * overlap * 0.5 / dist
                    grad[i, 0] -= dx * move
                    grad[i, 1] -= dy * move
                    grad[j, 0] += dx * move
                    grad[j, 1] += dy * move
        
        velocity = damping * velocity + grad
        circles[:, 0] += velocity[:, 0]
        circles[:, 1] += velocity[:, 1]
        
        circles[:, 0] = np.clip(circles[:, 0], circles[:, 2], width - circles[:, 2])
        circles[:, 1] = np.clip(circles[:, 1], circles[:, 2], height - circles[:, 2])
    
    return circles


if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
    benchmark = 2.3658321334167627
    print(f"Combined score: {np.sum(circles[:,-1]) / benchmark}")