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
    
    # Initialize with hexagonal-like packing in unit rectangle (w=1, h=1 initially)
    # We'll optimize the aspect ratio too
    r_init = 0.08  # Initial radius guess
    circles = np.zeros((n, 3))
    
    # Create a grid-based initial configuration
    cols = 5
    for i in range(n):
        row = i // cols
        col = i % cols
        x = r_init + col * (2 * r_init + 0.01)
        y = r_init + row * (2 * r_init + 0.01) + (col % 2) * r_init  # Stagger odd rows
        circles[i] = [x, y, r_init]
    
    # Optimize width/height ratio (w + h = 2)
    def pack_objective(params):
        width = params[0]
        height = 2.0 - width
        x = params[1:1+n]
        y = params[1+n:1+2*n]
        r = params[1+2*n:]
        return -np.sum(r)  # Minimize negative sum = maximize sum
    
    def constraints(params):
        width = params[0]
        height = 2.0 - width
        x = params[1:1+n]
        y = params[1+n:1+2*n]
        r = params[1+2*n:]
        
        cons = []
        # Boundary constraints
        for i in range(n):
            cons.append(x[i] - r[i])  # >= 0
            cons.append(y[i] - r[i])  # >= 0
            cons.append(width - x[i] - r[i])  # >= 0
            cons.append(height - y[i] - r[i])  # >= 0
        
        # Non-overlap constraints
        for i in range(n):
            for j in range(i+1, n):
                dx = x[i] - x[j]
                dy = y[i] - y[j]
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - (r[i] + r[j]))  # >= 0
        
        # Positive radii
        for i in range(n):
            cons.append(r[i] - 1e-6)
        
        return np.array(cons)
    
    # Initial parameter vector: [width, x1...xn, y1...yn, r1...rn]
    initial_params = np.concatenate([[1.0], circles[:,0], circles[:,1], circles[:,2]])
    
    bounds = [(0.5, 1.5)] + [(0, 2)] * (2*n) + [(0.001, 0.5)] * n
    
    try:
        result = minimize(
            pack_objective,
            initial_params,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 1000, 'ftol': 1e-9}
        )
        
        if result.success:
            p = result.x
            circles[:,0] = p[1:1+n]
            circles[:,1] = p[1+n:1+2*n]
            circles[:,2] = p[1+2*n:]
    except Exception:
        pass
    
    # Ensure positive radii
    circles[:,2] = np.maximum(circles[:,2], 0.001)
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")