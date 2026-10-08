# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import basinhopping, dual_annealing, minimize
import time


def circle_packing21() -> np.ndarray:
    """
    Places 21 non-overlapping circles inside a rectangle of perimeter 4 in order to maximize the sum of their radii.

    Returns:
        circles: np.array of shape (21,3), where the i-th row (x,y,r) stores the (x,y) coordinates of the i-th circle of radius r.
    """
    np.random.seed(42)
    n = 21
    penalty = 600.0
    
    # Parameterization: x[0]=width, then x[1::3]=x_i, x[2::3]=y_i, x[3::3]=r_i
    # Total dim: 1 + 3*21 = 64
    
    # Improved initial guess combining best aspects from both exemplars
    width_init = 1.12
    r_init = 0.088
    # Predefined optimized seed positions
    x_coords = np.array([
        0.107, 0.290, 0.473, 0.656, 0.839, 0.198, 0.381, 0.564, 0.747, 0.930,
        0.107, 0.290, 0.473, 0.656, 0.839, 0.198, 0.381, 0.564, 0.747, 0.930,
        0.107
    ])[:n]
    y_coords = np.array([
        0.107, 0.107, 0.107, 0.107, 0.107, 0.265, 0.265, 0.265, 0.265, 0.265,
        0.423, 0.423, 0.423, 0.423, 0.423, 0.581, 0.581, 0.581, 0.581, 0.581,
        0.739
    ])[:n]
    
    x0 = np.zeros(1 + 3 * n)
    x0[0] = width_init
    for i in range(n):
        x0[1 + 3*i] = x_coords[i]
        x0[1 + 3*i + 1] = y_coords[i]
        x0[1 + 3*i + 2] = r_init
    
    def smooth_objective(x):
        rs = x[3::3]
        return np.sum(rs)
    
    def constraint_violations(x):
        width = x[0]
        height = 2.0 - width
        xs = x[1::3]
        ys = x[2::3]
        rs = x[3::3]
        
        violations = []
        # Boundary constraints: violation = max(0, constraint)**2
        violations.extend(np.maximum(0, rs - xs)**2)
        violations.extend(np.maximum(0, rs + xs - width)**2)
        violations.extend(np.maximum(0, rs - ys)**2)
        violations.extend(np.maximum(0, rs + ys - height)**2)
        # Positive radii
        violations.extend(np.maximum(0, -rs)**2)
        # No overlap - vectorized upper triangle
        i_idx, j_idx = np.triu_indices(n, k=1)
        dx = xs[i_idx] - xs[j_idx]
        dy = ys[i_idx] - ys[j_idx]
        dist = np.sqrt(dx*dx + dy*dy + 1e-12)
        sum_r = rs[i_idx] + rs[j_idx]
        overlap = np.maximum(0, sum_r - dist)
        violations.extend(overlap**2)
        return np.array(violations)
    
    def loss(x):
        obj = smooth_objective(x)
        constr = constraint_violations(x)
        return -obj + penalty * np.sum(constr)
    
    # Bounds
    bounds = [(0.15, 1.85)] + [(0.0, 2.0), (0.0, 2.0), (1e-4, 1.0)] * n
    
    best_loss = np.inf
    best_x = x0.copy()
    
    # Basin Hopping with controlled iterations (strict budget compliance)
    try:
        minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 300}}
        bh_result = basinhopping(loss, x0, niter=40, T=0.005, stepsize=0.05, 
                                  minimizer_kwargs=minimizer_kwargs, seed=42)
        if bh_result.fun < best_loss:
            best_loss = bh_result.fun
            best_x = bh_result.x.copy()
    except Exception:
        pass
    
    # Dual Annealing with capped iterations (1000 iterations per system instruction)
    try:
        da_result = dual_annealing(loss, bounds, maxiter=1000, seed=42)
        if da_result.fun < best_loss:
            best_loss = da_result.fun
            best_x = da_result.x.copy()
    except Exception:
        pass
    
    # First polish phase
    try:
        polish_result = minimize(loss, best_x, method='L-BFGS-B', bounds=bounds,
                                 options={'ftol':1e-10, 'gtol':1e-10, 'maxiter': 1000})
        if polish_result.fun < best_loss:
            best_loss = polish_result.fun
            best_x = polish_result.x.copy()
    except Exception:
        pass
    
    # Targeted perturbation: apply small controlled adjustments to escape local minima
    np.random.seed(123)  # Secondary seed for controlled perturbation
    for perturb_round in range(3):
        # Generate slightly perturbed candidates
        perturbed = best_x.copy()
        # Perturb positions more than radii
        perturbed[1::3] += np.random.normal(0, 0.008, n)
        perturbed[2::3] += np.random.normal(0, 0.008, n)
        perturbed[3::3] += np.random.normal(0, 0.002, n)
        perturbed[0] += np.random.normal(0, 0.01)
        # Local polish on perturbed candidate
        try:
            pert_result = minimize(loss, perturbed, method='L-BFGS-B', bounds=bounds,
                                   options={'ftol':1e-9, 'gtol':1e-9, 'maxiter': 500})
            if pert_result.fun < best_loss - 1e-8:  # Strict improvement check
                best_loss = pert_result.fun
                best_x = pert_result.x.copy()
        except Exception:
            pass
    
    # Final polish with tight tolerances
    try:
        final_polish = minimize(loss, best_x, method='L-BFGS-B', bounds=bounds,
                                options={'ftol':1e-12, 'gtol':1e-12, 'maxiter': 800})
        if final_polish.fun < best_loss:
            best_x = final_polish.x.copy()
    except Exception:
        pass
    
    # Project to feasible region
    width = np.clip(best_x[0], 0.1, 1.9)
    height = 2.0 - width
    xs = best_x[1::3].copy()
    ys = best_x[2::3].copy()
    rs = np.maximum(best_x[3::3], 1e-4)
    
    # Iterative constraint enforcement (enhanced)
    for _ in range(15):
        # Boundary enforcement
        for i in range(n):
            r = rs[i]
            xs[i] = np.clip(xs[i], r, width - r)
            ys[i] = np.clip(ys[i], r, height - r)
        # Reduce radii to fix overlaps
        for i in range(n):
            for j in range(i+1, n):
                dx = xs[i] - xs[j]
                dy = ys[i] - ys[j]
                dist = np.sqrt(dx*dx + dy*dy)
                sum_r = rs[i] + rs[j]
                if dist < sum_r and dist > 0:
                    scale = dist / sum_r * 0.9995
                    rs[i] *= scale
                    rs[j] *= scale
    
    circles = np.column_stack([xs, ys, rs])
    return circles


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    circles = circle_packing21()
    sum_r = np.sum(circles[:,-1])
    print(f"Radii sum: {sum_r}")
    benchmark = 2.3658321334167627
    print(f"Combined score: {sum_r / benchmark}")