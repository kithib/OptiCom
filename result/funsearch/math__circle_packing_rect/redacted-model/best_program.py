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
    
    # Rectangle width/height constraint: width + height = 2 (perimeter = 4)
    # Start with equal aspect ratio as reasonable initial guess
    aspect_ratio = 1.0
    width = 2 * aspect_ratio / (1 + aspect_ratio)
    height = 2 / (1 + aspect_ratio)
    
    # Initialize circles in a hexagonal packing pattern with small random perturbations
    # Grid pattern: try to space circles reasonably
    cols = int(np.ceil(np.sqrt(n * aspect_ratio)))
    rows = int(np.ceil(n / cols))
    
    circles = np.zeros((n, 3))
    idx = 0
    for i in range(rows):
        for j in range(cols):
            if idx >= n:
                break
            # Hexagonal packing offset on odd rows
            x_offset = 0.05 * width if i % 2 == 1 else 0.0
            circles[idx, 0] = (j + 0.5) * width / cols + x_offset + 0.01 * np.random.randn()
            circles[idx, 1] = (i + 0.5) * height / rows + 0.01 * np.random.randn()
            circles[idx, 2] = 0.03  # Small initial radius
            idx += 1
    
    # Optimization variables: [x0, y0, r0, x1, y1, r1, ..., x20, y20, r20, aspect_ratio]
    x0 = np.hstack([circles.flatten(), aspect_ratio])
    
    def objective(x):
        radii = x[2::3]
        return -np.sum(radii)  # Negative for minimization
    
    constraints = []
    eps = 1e-8
    
    # Boundary constraints for each circle
    for i in range(n):
        # Get aspect ratio from last variable and compute width/height
        def get_dims(aspect_ratio):
            w = 2 * aspect_ratio / (1 + aspect_ratio)
            h = 2 / (1 + aspect_ratio)
            return w, h
        
        # Left boundary: x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (2 * x[-1] / (1 + x[-1])) * x[3*i] / (2 * x[-1] / (1 + x[-1])) - x[3*i+2]
        })
        # Right boundary: width - x >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (2 * x[-1] / (1 + x[-1])) - x[3*i] - x[3*i+2]
        })
        # Bottom boundary: y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (2 / (1 + x[-1])) * x[3*i+1] / (2 / (1 + x[-1])) - x[3*i+2]
        })
        # Top boundary: height - y >= r
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: (2 / (1 + x[-1])) - x[3*i+1] - x[3*i+2]
        })
    
    # Non-overlap constraints between all pairs
    for i in range(n):
        for j in range(i+1, n):
            constraints.append({
                'type': 'ineq',
                'fun': lambda x, i=i, j=j: 
                    np.sqrt((x[3*i] - x[3*j])**2 + (x[3*i+1] - x[3*j+1])**2) 
                    - x[3*i+2] - x[3*j+2]
            })
    
    # Positive radius constraints
    for i in range(n):
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, i=i: x[3*i+2] - eps
        })
    
    # Aspect ratio bounds
    bounds = []
    for i in range(n):
        bounds.append((None, None))  # x
        bounds.append((None, None))  # y
        bounds.append((eps, 0.5))    # r
    bounds.append((0.2, 5.0))  # aspect_ratio
    
    # First try SLSQP
    try:
        result = minimize(objective, x0, method='SLSQP', bounds=bounds, 
                         constraints=constraints, 
                         options={'maxiter': 1000, 'ftol': 1e-10})
        x_opt = result.x
    except:
        # Fallback: try COBYLA if SLSQP fails
        x_opt = x0
    
    # Extract optimal circles
    optimal_circles = x_opt[:-1].reshape(n, 3)
    
    # Clamp values to ensure constraints are satisfied
    aspect_ratio_opt = x_opt[-1]
    w = 2 * aspect_ratio_opt / (1 + aspect_ratio_opt)
    h = 2 / (1 + aspect_ratio_opt)
    
    for i in range(n):
        r = max(optimal_circles[i, 2], eps)
        optimal_circles[i, 0] = np.clip(optimal_circles[i, 0], r, w - r)
        optimal_circles[i, 1] = np.clip(optimal_circles[i, 1], r, h - r)
        optimal_circles[i, 2] = r
    
    return optimal_circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")
    print(f"Circles:\n{circles}")