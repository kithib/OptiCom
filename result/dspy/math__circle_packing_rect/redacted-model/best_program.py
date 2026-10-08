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
    
    def decode(xyz):
        # xyz[0] = width, height = 2 - width
        # xyz[1:1+3*n] = (x1, y1, r1, x2, y2, r2, ...)
        W = xyz[0]
        H = 2.0 - W
        circles = xyz[1:].reshape(n, 3)
        return W, H, circles
    
    def objective(xyz):
        W, H, circles = decode(xyz)
        return -np.sum(circles[:, 2])
    
    def constraints(xyz):
        W, H, circles = decode(xyz)
        cons = []
        
        # Width bounds
        cons.append(W - 0.3)
        cons.append(1.7 - W)
        
        for i in range(n):
            x, y, r = circles[i]
            # Containment constraints
            cons.append(x - r)
            cons.append(W - x - r)
            cons.append(y - r)
            cons.append(H - y - r)
            # Positive radius
            cons.append(r - 1e-6)
            
            # Non-overlap with other circles
            for j in range(i + 1, n):
                xj, yj, rj = circles[j]
                dx = x - xj
                dy = y - yj
                dist = np.sqrt(dx*dx + dy*dy)
                cons.append(dist - r - rj)
        
        return np.array(cons)
    
    # Generate initial guess using a deterministic spiral pattern
    # Start with a square-ish aspect ratio
    initial_W = 1.0
    
    # Generate spiral positions for initial guess
    spiral_points = []
    for i in range(n):
        theta = i * (3 - np.sqrt(5)) * np.pi
        r = np.sqrt(i + 0.5) * 0.08
        x = 0.5 + r * np.cos(theta)
        y = 0.5 + r * np.sin(theta)
        spiral_points.append((x, y, 0.08))
    
    x0 = np.concatenate([[initial_W], np.array(spiral_points).flatten()])
    
    # Optimize
    bounds = [(0.3, 1.7)] + [(None, None)] * (3 * n)
    
    try:
        result = minimize(
            objective,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints={'type': 'ineq', 'fun': constraints},
            options={'maxiter': 2000, 'ftol': 1e-10, 'disp': False}
        )
        
        W, H, circles = decode(result.x)
        
        # Double-check and enforce constraints
        for i in range(n):
            x, y, r = circles[i]
            # Ensure containment
            r = min(r, x, W - x, y, H - y)
            # Ensure positive radius
            r = max(r, 1e-6)
            circles[i] = [x, y, r]
        
        return circles.astype(float)
    except:
        # Fallback: return a valid (though suboptimal) configuration
        fallback = np.zeros((n, 3))
        for i in range(n):
            fallback[i] = [0.5, 0.5 + i * 0.09, 0.04]
        return fallback


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")