# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 21
    
    # Initialize with a reasonable grid arrangement
    cols = 5
    rows = (n + cols - 1) // cols
    initial_r = 0.08
    
    x = []
    y = []
    for i in range(n):
        row = i // cols
        col = i % cols
        x.append(initial_r + col * 2.5 * initial_r)
        y.append(initial_r + row * 2.5 * initial_r)
    
    x = np.array(x)
    y = np.array(y)
    r = np.full(n, initial_r)
    
    # Decision variables: width, height (w+h=2), then x, y, r for each circle
    w0 = 1.2
    h0 = 0.8
    x0 = np.concatenate([[w0, h0], x, y, r])
    
    def objective(vars):
        return -np.sum(vars[2 + 2*n : 2 + 3*n])  # Negative sum for minimization
    
    constraints = []
    
    # Width + height = 2 constraint
    constraints.append({'type': 'eq', 'fun': lambda v: v[0] + v[1] - 2.0})
    
    # Boundary constraints for each circle
    for i in range(n):
        xi = 2 + i
        yi = 2 + n + i
        ri = 2 + 2*n + i
        # x >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[xi] - v[ri]})
        # x <= w - r
        constraints.append({'type': 'ineq', 'fun': lambda v, xi=xi, ri=ri: v[0] - v[xi] - v[ri]})
        # y >= r
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[yi] - v[ri]})
        # y <= h - r
        constraints.append({'type': 'ineq', 'fun': lambda v, yi=yi, ri=ri: v[1] - v[yi] - v[ri]})
    
    # Non-overlapping constraints
    for i in range(n):
        for j in range(i+1, n):
            xi = 2 + i
            yi = 2 + n + i
            ri = 2 + 2*n + i
            xj = 2 + j
            yj = 2 + n + j
            rj = 2 + 2*n + j
            constraints.append({
                'type': 'ineq',
                'fun': lambda v, xi=xi, yi=yi, ri=ri, xj=xj, yj=yj, rj=rj: 
                    np.sqrt((v[xi]-v[xj])**2 + (v[yi]-v[yj])**2) - v[ri] - v[rj]
            })
    
    # Positive radii constraints
    for i in range(n):
        ri = 2 + 2*n + i
        constraints.append({'type': 'ineq', 'fun': lambda v, ri=ri: v[ri] - 1e-6})
    
    # Bounds
    bounds = [(0.1, 1.9), (0.1, 1.9)]  # width and height bounds
    for _ in range(n):
        bounds.append((0, 2.0))  # x
    for _ in range(n):
        bounds.append((0, 2.0))  # y
    for _ in range(n):
        bounds.append((1e-6, 0.5))  # r
    
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, options={'maxiter': 1000, 'ftol': 1e-9})
        vars_opt = result.x
    except:
        vars_opt = x0
    
    circles = np.zeros((n, 3))
    for i in range(n):
        circles[i, 0] = vars_opt[2 + i]
        circles[i, 1] = vars_opt[2 + n + i]
        circles[i, 2] = vars_opt[2 + 2*n + i]
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")