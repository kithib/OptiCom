# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    n = 21
    np.random.seed(42)  # Determinism requirement
    
    # Optimize width/height ratio first (golden ratio as starting point)
    def objective_ratio(wh):
        w = wh[0]
        h = 2 - w
        if w <= 0 or h <= 0:
            return 1e10
        # Estimate potential radius based on area efficiency
        area_efficiency = 0.85  # Typical circle packing density
        total_circle_area = area_efficiency * w * h
        # Equal area circles
        r_estimate = np.sqrt(total_circle_area / (n * np.pi))
        return -r_estimate  # Negative for minimization
    
    w_opt = minimize(objective_ratio, x0=[1.2], bounds=[(0.1, 1.9)], method='L-BFGS-B').x[0]
    w = max(0.1, min(1.9, w_opt))
    h = 2 - w
    
    # Generate initial configuration - hexagonal grid packing with equal radii
    cols = int(np.ceil(np.sqrt(n * w / h)))
    rows = int(np.ceil(n / cols))
    
    circles = np.zeros((n, 3))
    for i in range(n):
        row = i // cols
        col = i % cols
        # Hexagonal grid offset for odd rows
        offset_x = (w / (2 * cols)) if (row % 2 == 1) else 0
        circles[i, 0] = (col + 0.5) * (w / cols) + offset_x + 0.01 * np.random.randn()
        circles[i, 1] = (row + 0.5) * (h / rows) + 0.01 * np.random.randn()
    
    # Calculate initial equal radius - use more aggressive initial spacing
    spacing = min(w / cols, h / rows)
    initial_r = spacing / 2.1
    circles[:, 2] = initial_r
    
    # Optimization variables: [w, *x_coords, *y_coords, *radii]
    # Note: width + height = 2, so height = 2 - width
    def pack_objective(vars):
        current_w = vars[0]
        current_h = 2 - current_w
        xs = vars[1:1+n]
        ys = vars[1+n:1+2*n]
        rs = vars[1+2*n:1+3*n]
        return -np.sum(rs)  # Negative sum for minimization
    
    def constraints_pack(vars):
        current_w = vars[0]
        current_h = 2 - current_w
        xs = vars[1:1+n]
        ys = vars[1+n:1+2*n]
        rs = vars[1+2*n:1+3*n]
        
        cons = []
        # Boundary constraints
        for i in range(n):
            cons.append(xs[i] - rs[i])  # x >= r
            cons.append(current_w - xs[i] - rs[i])  # x <= w - r
            cons.append(ys[i] - rs[i])  # y >= r
            cons.append(current_h - ys[i] - rs[i])  # y <= h - r
            cons.append(rs[i] - 1e-6)  # Positive radius constraint with small buffer
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist_sq = dx*dx + dy*dy
                min_dist = rs[i] + rs[j]
                cons.append(dist_sq - min_dist * min_dist + 1e-12)  # Add small numeric buffer
        
        return np.array(cons)
    
    # Build initial variable vector
    x0 = np.concatenate([[w], circles[:, 0], circles[:, 1], circles[:, 2]])
    
    # Bounds
    lb = np.zeros_like(x0)
    ub = np.zeros_like(x0)
    lb[0] = 0.1
    ub[0] = 1.9
    lb[1:1+n] = 0
    ub[1:1+n] = 2.0
    lb[1+n:1+2*n] = 0
    ub[1+n:1+2*n] = 2.0
    lb[1+2*n:1+3*n] = 1e-6
    ub[1+2*n:1+3*n] = 1.0
    
    bounds = list(zip(lb, ub))
    
    # Sequential optimization for better results - more iterations
    for _ in range(4):
        result = minimize(
            pack_objective,
            x0=x0,
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints_pack},
            method='SLSQP',
            options={'maxiter': 1500, 'ftol': 1e-12}
        )
        if result.success:
            x0 = result.x.copy()
        else:
            break
    
    # Extract final configuration
    final_w = x0[0]
    circles[:, 0] = x0[1:1+n]
    circles[:, 1] = x0[1+n:1+2*n]
    circles[:, 2] = x0[1+2*n:1+3*n]
    
    # Final constraint validation and correction
    final_h = 2 - final_w
    for i in range(n):
        # Clip to boundaries (maintain positive radius)
        max_r = min(circles[i, 0], final_w - circles[i, 0], circles[i, 1], final_h - circles[i, 1])
        circles[i, 2] = min(circles[i, 2], max_r)
        circles[i, 2] = max(circles[i, 2], 1e-6)
        
        # Re-center if needed
        circles[i, 0] = max(circles[i, 0], circles[i, 2])
        circles[i, 0] = min(circles[i, 0], final_w - circles[i, 2])
        circles[i, 1] = max(circles[i, 1], circles[i, 2])
        circles[i, 1] = min(circles[i, 1], final_h - circles[i, 2])
    
    # Fix overlaps with a more aggressive approach - multiple passes
    for _ in range(5):
        overlap_found = False
        for i in range(n):
            for j in range(i+1, n):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist - 1e-8:
                    overlap_found = True
                    # Shrink radii proportionally to resolve overlap
                    ratio = dist / min_dist
                    circles[i, 2] *= ratio * 0.999
                    circles[j, 2] *= ratio * 0.999
        if not overlap_found:
            break
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
    benchmark = 2.3658321334167627
    print(f"Benchmark: {benchmark}")
    print(f"Ratio: {np.sum(circles[:,-1])/benchmark}")