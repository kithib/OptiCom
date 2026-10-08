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
    
    # Finely tuned aspect ratio: balanced width/height for hexagonal packing efficiency
    aspect = 1.29  # Intermediate value combining best from 1.28 and 1.32 variants
    w0, h0 = 2 * aspect / (1 + aspect), 2 / (1 + aspect)  # width + height = 2
    
    # Explicit row pattern matching 21 circles: 5, 4, 5, 4, 3 (optimal staggered arrangement)
    row_pattern = [5, 4, 5, 4, 3]
    actual_rows = len(row_pattern)
    
    # Improved initial radius: tighter initial packing creates better starting point for optimization
    # Explicitly accounts for hexagonal vertical spacing factor sqrt(3)/2
    max_col_count = max(row_pattern)
    dx = w0 / (max_col_count + 0.5)  # Account for horizontal stagger in odd rows
    dy = h0 / (actual_rows * np.sqrt(3) / 2)  # Vertical spacing for hexagonal packing efficiency
    r_init = min(dx, dy) * 0.492  # Balanced: tighter than 0.485, more robust than 0.495
    
    initial = np.zeros(2 + 3 * n)
    initial[0], initial[1] = w0, h0
    
    idx = 0
    for row_idx, row_len in enumerate(row_pattern):
        row_offset = r_init if row_idx % 2 else 0  # Stagger odd rows
        for col in range(row_len):
            if idx >= n:
                break
            # Carefully positioned initial hex grid with consistent 1.01x scaling factor
            initial[2 + 3*idx] = r_init * 1.01 + row_offset + col * 2 * r_init
            initial[3 + 3*idx] = r_init * 1.01 + row_idx * 2 * r_init * np.sqrt(3) / 2
            initial[4 + 3*idx] = r_init
            idx += 1
    
    def objective(vars):
        # Maximize sum of radii -> minimize negative sum
        return -np.sum(vars[4::3])  # Extract radii from position 4 onward
    
    constraints = []
    constraints.append({'type': 'eq', 'fun': lambda v: v[0] + v[1] - 2})  # Perimeter constraint
    
    # Named constraint factory functions to avoid closure issues
    def constraint_x_min(i):
        return lambda v, i=i: v[2 + 3*i] - v[4 + 3*i]
    def constraint_y_min(i):
        return lambda v, i=i: v[3 + 3*i] - v[4 + 3*i]
    def constraint_x_max(i):
        return lambda v, i=i: v[0] - v[2 + 3*i] - v[4 + 3*i]
    def constraint_y_max(i):
        return lambda v, i=i: v[1] - v[3 + 3*i] - v[4 + 3*i]
    def constraint_positive_r(i):
        return lambda v, i=i: v[4 + 3*i] - 1e-3  # Robust minimum radius constraint
    def constraint_overlap(i, j):
        return lambda v, i=i, j=j: np.sqrt((v[2+3*i]-v[2+3*j])**2 + (v[3+3*i]-v[3+3*j])**2) - v[4+3*i] - v[4+3*j]

    for i in range(n):
        # Boundary constraints
        constraints.append({'type': 'ineq', 'fun': constraint_x_min(i)})
        constraints.append({'type': 'ineq', 'fun': constraint_y_min(i)})
        constraints.append({'type': 'ineq', 'fun': constraint_x_max(i)})
        constraints.append({'type': 'ineq', 'fun': constraint_y_max(i)})
        # Positive radius constraint (enforce minimum)
        constraints.append({'type': 'ineq', 'fun': constraint_positive_r(i)})
        # Pairwise overlap constraints
        for j in range(i+1, n):
            constraints.append({'type': 'ineq', 'fun': constraint_overlap(i, j)})
    
    # Optimized bounds: tighter constraint on aspect ratio, bounded x/y/r ranges
    bounds = [(0.55, 1.45), (0.55, 1.45)]  # Narrower aspect ratio bounds for stability
    for _ in range(n):
        bounds.extend([(0, 2), (0, 2), (1e-3, 0.5)])  # Consistent bounded ranges
    
    try:
        # First optimization pass: moderately aggressive to find feasible region
        result = minimize(objective, initial, bounds=bounds, constraints=constraints,
                         method='SLSQP', options={'maxiter': 3500, 'ftol': 1e-12, 'eps': 1e-9})
        vars_opt = None
        if result.success:
            # First polish pass: refine solution with tighter tolerances
            result2 = minimize(objective, result.x, bounds=bounds, constraints=constraints,
                              method='SLSQP', options={'maxiter': 4000, 'ftol': 1e-13, 'eps': 1e-9})
            vars_opt = result2.x if result2.success else result.x
            # Second polish pass: highest precision refinement
            result3 = minimize(objective, vars_opt, bounds=bounds, constraints=constraints,
                              method='SLSQP', options={'maxiter': 4500, 'ftol': 1e-14, 'eps': 1e-10})
            vars_opt = result3.x if result3.success else vars_opt
        else:
            # Fallback 1: try with higher iteration count and looser tolerance
            result_fb1 = minimize(objective, initial, bounds=bounds, constraints=constraints,
                                  method='SLSQP', options={'maxiter': 5500, 'ftol': 1e-8, 'eps': 1e-7})
            if result_fb1.success:
                vars_opt = result_fb1.x
            else:
                # Fallback 2: alternative bounds (0.6-1.6, 0.4-1.4) if primary fails
                bounds_alt = [(0.6, 1.6), (0.4, 1.4)]
                for _ in range(n):
                    bounds_alt.extend([(0, 2), (0, 2), (1e-3, 0.5)])
                result_fb2 = minimize(objective, initial, bounds=bounds_alt, constraints=constraints,
                                      method='SLSQP', options={'maxiter': 4000, 'ftol': 1e-10, 'eps': 1e-8})
                if result_fb2.success:
                    vars_opt = result_fb2.x
        
        if vars_opt is not None:
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i] = vars_opt[2+3*i:2+3*i+3]
            return circles
    except Exception:
        pass
    
    # Fallback: return improved grid configuration if optimization fails
    circles = np.zeros((n, 3))
    grid_n = 5
    for i in range(n):
        row = i // grid_n
        col = i % grid_n
        offset = 0.1 * (row % 2)
        circles[i, 0] = 0.08 + 0.84 * (col + offset) / max(1, grid_n - 1)
        circles[i, 1] = 0.08 + 0.84 * row / max(1, (n + grid_n - 1) // grid_n - 1)
        circles[i, 2] = 0.078  # Robust fallback radius
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
    benchmark = 2.3658321334167627
    print(f"Combined score: {np.sum(circles[:,-1]) / benchmark}")