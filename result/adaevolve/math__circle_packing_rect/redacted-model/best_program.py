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
    
    # Fixed rectangle dimensions (perimeter 4 -> width + height = 2)
    # Using a balanced aspect ratio for better packing
    width = 1.1
    height = 0.9  # width + height = 2
    
    def objective(vars):
        # Minimize negative sum of radii to maximize sum
        radii = vars[2::3]
        return -np.sum(radii)
    
    def constraint_boundary(vars):
        # Vectorized computation: All circles must be within rectangle: x - r >= 0, x + r <= width, y - r >= 0, y + r <= height
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        return np.concatenate([x - r, width - (x + r), y - r, height - (y + r)])
    
    def constraint_overlap(vars):
        # Partially vectorized: No overlap: distance between centers >= sum of radii
        x = vars[0::3]
        y = vars[1::3]
        r = vars[2::3]
        constraints = []
        for i in range(n):
            dx = x[i] - x[i+1:]
            dy = y[i] - y[i+1:]
            dist = np.sqrt(dx*dx + dy*dy)
            constraints.extend(dist - (r[i] + r[i+1:]))
        return np.array(constraints)
    
    def constraint_positive_radii(vars):
        # Ensure strictly positive radii with small margin
        return vars[2::3] - 1e-6
    
    # Improved initial guess: grid-based placement with hexagonal offset and varied radii
    cols = 5
    rows = (n + cols - 1) // cols
    initial_radius_base = 0.06
    x_spacing = width / cols
    y_spacing = height / rows
    
    initial_vars = []
    for i in range(n):
        col = i % cols
        row = i // cols
        # Hexagonal offset on odd rows for denser packing
        x_offset = 0.5 * x_spacing * (row % 2)
        x = (col + 0.5) * x_spacing + x_offset
        y = (row + 0.5) * y_spacing
        # Slightly varied radii to help optimization explore larger sums
        center_dist = np.sqrt((x/width - 0.5)**2 + (y/height - 0.5)**2)
        r = initial_radius_base + 0.015 * np.cos(center_dist * np.pi)
        # Ensure initial positions respect bounds
        x = np.clip(x, r, width - r)
        y = np.clip(y, r, height - r)
        initial_vars.extend([x, y, r])
    initial_vars = np.array(initial_vars)
    
    # Constraints for SLSQP
    constraints = [
        {'type': 'ineq', 'fun': constraint_boundary},
        {'type': 'ineq', 'fun': constraint_overlap},
        {'type': 'ineq', 'fun': constraint_positive_radii}
    ]
    
    # Bounds: x in [0,width], y in [0,height], r in [0.001, min(width,height)/2]
    bounds = []
    for _ in range(n):
        bounds.extend([(0, width), (0, height), (0.001, min(width, height)/2)])
    
    try:
        # Extended main optimization with higher iterations and tighter tolerance
        result = minimize(
            objective,
            initial_vars,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1500, 'ftol': 1e-10}
        )
        optimized_vars = result.x
        
        # Polishing run if first succeeded for finer optimization
        if result.success:
            result2 = minimize(
                objective,
                optimized_vars,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 600, 'ftol': 1e-12}
            )
            if result2.success:
                optimized_vars = result2.x
    except Exception:
        # Fallback to initial guess if optimization fails
        optimized_vars = initial_vars
    
    # Reshape to (n,3) array
    circles = optimized_vars.reshape((n, 3))
    
    # Final safety: ensure all radii are positive and within bounds
    circles[:, 2] = np.clip(circles[:, 2], 0.001, min(width, height)/2)
    
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    print(f"Radii sum: {np.sum(circles[:,-1])}")