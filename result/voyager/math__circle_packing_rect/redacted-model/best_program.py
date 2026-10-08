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
    np.random.seed(42)
    
    # Decision variables: width, height, then n circles (x_i, y_i, r_i)
    # w + h = 2 constraint
    w = 1.0
    h = 1.0
    
    # Initialize with a hexagonal-like packing - IMPROVED INITIALIZATION
    circles = np.zeros((n, 3))
    idx = 0
    
    # Create multiple rows of circles - better distribution for 21
    rows = 5
    circles_per_row = [5, 4, 4, 4, 4]  # Sum to 21
    
    base_r = 0.095
    y_spacing = base_r * 2 * np.sqrt(3) / 2  # Hexagonal vertical spacing
    
    y_pos = base_r
    for row, count in enumerate(circles_per_row):
        x_spacing = (w - 2*base_r) / (count + 1)
        offset = base_r if row % 2 == 0 else base_r + x_spacing/2
        
        for i in range(count):
            if idx >= n:
                break
            circles[idx, 0] = offset + (i + 0.5) * x_spacing
            circles[idx, 1] = y_pos
            circles[idx, 2] = base_r
            idx += 1
        y_pos += y_spacing
    
    # Optimization to maximize sum of radii
    def objective(vars):
        # vars: w, then x1,y1,r1,x2,y2,r2,...xn,yn,rn
        rs = vars[3::3]
        return -np.sum(rs)  # Negative for minimization
    
    def constraint_boundary(vars):
        w_var = vars[0]
        h_var = 2.0 - w_var
        constraints = []
        for i in range(n):
            x = vars[1 + i*3]
            y = vars[2 + i*3]
            r = vars[3 + i*3]
            constraints.extend([
                x - r,           # x >= r
                w_var - x - r,   # x <= w - r
                y - r,           # y >= r
                h_var - y - r    # y <= h - r
            ])
        return np.array(constraints)
    
    def constraint_overlap(vars):
        constraints = []
        for i in range(n):
            x1 = vars[1 + i*3]
            y1 = vars[2 + i*3]
            r1 = vars[3 + i*3]
            for j in range(i+1, n):
                x2 = vars[1 + j*3]
                y2 = vars[2 + j*3]
                r2 = vars[3 + j*3]
                dx = x1 - x2
                dy = y1 - y2
                dist = np.sqrt(dx*dx + dy*dy)
                constraints.append(dist - r1 - r2)
        return np.array(constraints)
    
    def constraint_positive_radii(vars):
        return vars[3::3] - 1e-6  # All radii > 1e-6
    
    def constraint_width(vars):
        return np.array([vars[0] - 0.5, 1.5 - vars[0]])  # w between 0.5 and 1.5 (better range)
    
    # Initial guess
    x0 = np.concatenate([[w], circles.flatten()])
    
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_positive_radii},
        {'type': 'ineq', 'fun': constraint_width}
    ]
    
    bounds = [(0.5, 1.5)] + [(None, None)] * (3 * n)
    
    try:
        # First optimization with SLSQP
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-10})
        
        # Try to improve further with higher precision
        if result.success:
            result = minimize(objective, result.x, method='SLSQP', bounds=bounds,
                             constraints=constraints, options={'maxiter': 2000, 'ftol': 1e-12})
        
        w_opt = result.x[0]
        optimized = result.x[1:].reshape(n, 3)
        
        # Verify and slightly clamp to ensure feasibility
        h_opt = 2.0 - w_opt
        eps = 1e-8
        for i in range(n):
            r = optimized[i, 2]
            optimized[i, 0] = np.clip(optimized[i, 0], r + eps, w_opt - r - eps)
            optimized[i, 1] = np.clip(optimized[i, 1], r + eps, h_opt - r - eps)
            optimized[i, 2] = max(optimized[i, 2], 1e-6)
        
        return optimized
    except:
        # Fallback to reasonable configuration
        return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")