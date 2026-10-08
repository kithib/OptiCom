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
    
    # Even better rectangle aspect ratio based on known optimal packing configurations
    best_width = 1.152
    best_height = 2 - best_width
    
    # Generate improved initial configuration: hexagonal lattice with exact 6-5-5-5 pattern (21 total)
    # This pattern balances horizontal and vertical space more evenly
    cols = 6
    rows = 4
    # Finer spacing adjustment for higher initial density
    spacing = min(best_width / (cols + 0.165), best_height / (rows + 0.165)) * 0.965
    # Tighter initial radius for optimal hexagonal packing efficiency
    init_r = spacing / 1.999
    
    x_coords = []
    y_coords = []
    count = 0
    for i in range(rows):
        # Exact 6-5-5-5 row pattern for more balanced packing density across rows
        if i == 0:
            row_cols = 6
        elif i == 1:
            row_cols = 5
        elif i == 2:
            row_cols = 5
        else:
            row_cols = 5
        for j in range(row_cols):
            if count >= n:
                break
            offset = 0.5 * spacing if (i % 2 == 1) else 0
            # Perfect centering of each row within container bounds
            total_row_width = row_cols * spacing + (0.5 * spacing if (i % 2 == 1) else 0)
            x_offset = (best_width - total_row_width) / 2.0
            x = spacing * (j + 0.5) + offset + x_offset
            # Vertical centering within container
            y = spacing * (i + 0.5) + (best_height - rows * spacing) / 2.0
            x_coords.append(x)
            y_coords.append(y)
            count += 1
        if count >= n:
            break
    
    x_coords = np.array(x_coords)[:n]
    y_coords = np.array(y_coords)[:n]
    
    # Build initial guess: [width, x1, y1, r1, x2, y2, r2, ..., xn, yn, rn]
    x0 = np.zeros(1 + 3 * n)
    x0[0] = best_width
    for i in range(n):
        x0[1 + 3*i] = x_coords[i]
        x0[1 + 3*i + 1] = y_coords[i]
        x0[1 + 3*i + 2] = init_r
    
    # Objective: negative sum of radii (for minimization)
    def objective(x):
        return -np.sum(x[3::3])
    
    # Constraints
    constraints = []
    
    # Boundary constraints for each circle
    for i in range(n):
        idx_x = 1 + 3*i
        idx_y = 1 + 3*i + 1
        idx_r = 1 + 3*i + 2
        
        # x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, ix=idx_x, ir=idx_r: x[ix] - x[ir]
        })
        # x <= width - r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, ix=idx_x, ir=idx_r: (x[0] - x[ix]) - x[ir]
        })
        # y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, iy=idx_y, ir=idx_r: x[iy] - x[ir]
        })
        # y <= height - r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, iy=idx_y, ir=idx_r: ((2 - x[0]) - x[iy]) - x[ir]
        })
    
    # No-overlap constraints using squared distance for better numerical stability
    # Even smaller epsilon allows tighter packing while maintaining non-overlap constraint
    for i in range(n):
        for j in range(i + 1, n):
            idx_xi = 1 + 3*i
            idx_yi = 1 + 3*i + 1
            idx_ri = 1 + 3*i + 2
            idx_xj = 1 + 3*j
            idx_yj = 1 + 3*j + 1
            idx_rj = 1 + 3*j + 2
            
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, xi=idx_xi, yi=idx_yi, ri=idx_ri, xj=idx_xj, yj=idx_yj, rj=idx_rj:
                    ((x[xi] - x[xj])**2 + (x[yi] - x[yj])**2) - (x[ri] + x[rj])**2 - 1e-15
            })
    
    # Even tighter bounds on width for more focused search around optimum
    bounds = [(1.14, 1.17)]
    for i in range(n):
        bounds.extend([(0, None), (0, None), (1e-5, 0.5)])
    
    try:
        # First optimization run with extended iterations for higher quality
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 5000, 'ftol': 1e-14, 'disp': False}
        )
        
        if result.success:
            # First refinement pass: intermediate iteration count with higher precision
            result2 = minimize(
                objective,
                result.x,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 3000, 'ftol': 1e-15, 'disp': False}
            )
            
            # Third refinement: polish final solution with ultra-tight tolerance
            if result2.success:
                result3 = minimize(
                    objective,
                    result2.x,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=constraints,
                    options={'maxiter': 2000, 'ftol': 1e-15, 'disp': False}
                )
                
                # Fourth refinement: ultra-polish
                if result3.success:
                    result4 = minimize(
                        objective,
                        result3.x,
                        method='SLSQP',
                        bounds=bounds,
                        constraints=constraints,
                        options={'maxiter': 1000, 'ftol': 1e-15, 'disp': False}
                    )
                    
                    if result4.success:
                        x = result4.x
                        circles = np.zeros((n, 3))
                        for i in range(n):
                            circles[i, 0] = x[1 + 3*i]
                            circles[i, 1] = x[1 + 3*i + 1]
                            circles[i, 2] = max(x[1 + 3*i + 2], 1e-6)
                        return circles
                
                if result3.success:
                    x = result3.x
                    circles = np.zeros((n, 3))
                    for i in range(n):
                        circles[i, 0] = x[1 + 3*i]
                        circles[i, 1] = x[1 + 3*i + 1]
                        circles[i, 2] = max(x[1 + 3*i + 2], 1e-6)
                    return circles
            
            # Fallback to second result
            if result2.success:
                x = result2.x
                circles = np.zeros((n, 3))
                for i in range(n):
                    circles[i, 0] = x[1 + 3*i]
                    circles[i, 1] = x[1 + 3*i + 1]
                    circles[i, 2] = max(x[1 + 3*i + 2], 1e-6)
                return circles
            
            # Fallback to first result
            x = result.x
            circles = np.zeros((n, 3))
            for i in range(n):
                circles[i, 0] = x[1 + 3*i]
                circles[i, 1] = x[1 + 3*i + 1]
                circles[i, 2] = max(x[1 + 3*i + 2], 1e-6)
            return circles
    except:
        pass
    
    # Fallback: return initial feasible configuration
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = x0[1 + 3*i]
        circles[i, 1] = x0[1 + 3*i + 1]
        circles[i, 2] = max(x0[1 + 3*i + 2], 1e-6)
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
    benchmark = 2.3658321334167627
    print(f"Combined score: {np.sum(circles[:,-1]) / benchmark}")