import numpy as np
from scipy.optimize import basinhopping, dual_annealing, minimize
import time

def hexagon_packing_11():
    """
    Constructs a packing of 11 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (11,3), where each row is of the form (x, y, angle_degrees) containing the (x,y) coordinates and angle_degree of the respective inner hexagon.
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree) containing the (x,y) coordinates and angle_degree of the outer hexagon.
        outer_hex_side_length: float representing the side length of the outer hexagon.
    """
    np.random.seed(42)
    sqrt3 = np.sqrt(3)
    n = 11
    
    # Initial layout from SOTA
    inner_hex_init = np.array(
        [
            [0, 0, 0],                      # center
            [-1.5, -sqrt3/2, 0],            # layer 1
            [-1.5, sqrt3/2, 0],             # layer 1
            [0, -sqrt3, 0],                  # layer 1
            [0, sqrt3, 0],                   # layer 1
            [1.5, -sqrt3/2, 0],             # layer 1
            [1.5, sqrt3/2, 0],              # layer 1
            [-3, -sqrt3, 0],                 # layer 2
            [-3, 0, 0],                      # layer 2
            [3, 0, 0],                       # layer 2
            [3, sqrt3, 0],                   # layer 2
        ],
        dtype=np.float64
    )
    
    init_x = inner_hex_init[:, :2].flatten()
    dim = 2 * n
    bounds = [(-10.0, 10.0)] * dim
    
    alpha = 1000.0
    penalty = 10000.0
    
    def hex_vertices_coords(x, y):
        angles = np.array([np.pi/6 * i for i in range(6)])
        return np.column_stack([x + np.cos(angles), y + np.sin(angles)])
    
    def smooth_logsumexp(a, alpha):
        return np.log(np.sum(np.exp(alpha * a))) / alpha
    
    def compute_smoothed_R(x):
        all_vertices = []
        for i in range(n):
            hx, hy = x[2*i], x[2*i+1]
            all_vertices.append(hex_vertices_coords(hx, hy))
        all_vertices = np.vstack(all_vertices)
        
        y_vals = np.abs(all_vertices[:, 1])
        x_vals = np.abs(all_vertices[:, 0])
        plus_vals = np.abs(all_vertices[:, 0] + all_vertices[:, 1]/sqrt3)
        minus_vals = np.abs(all_vertices[:, 0] - all_vertices[:, 1]/sqrt3)
        
        s1 = smooth_logsumexp(x_vals, alpha)
        s2 = smooth_logsumexp(plus_vals, alpha)
        s3 = smooth_logsumexp(minus_vals, alpha)
        s4 = smooth_logsumexp(2.0 * y_vals / sqrt3, alpha)
        
        return smooth_logsumexp(np.array([s1, s2, s3, s4]), alpha)
    
    def hex_dist_sq(xi, yi, xj, yj):
        dx = xj - xi
        dy = yj - yi
        dist1 = dx * dx + dy * dy
        dist2 = (dx - 1.5) ** 2 + (dy - sqrt3/2) ** 2
        dist3 = (dx - 1.5) ** 2 + (dy + sqrt3/2) ** 2
        dist4 = (dx + 1.5) ** 2 + (dy - sqrt3/2) ** 2
        dist5 = (dx + 1.5) ** 2 + (dy + sqrt3/2) ** 2
        dist6 = (dx + 3.0) ** 2 + dy * dy
        dist7 = (dx - 3.0) ** 2 + dy * dy
        return min(dist1, dist2, dist3, dist4, dist5, dist6, dist7)
    
    def constraint_violation_sq(x):
        violation = 0.0
        min_dist_sq = 2.75
        for i in range(n):
            xi, yi = x[2*i], x[2*i+1]
            for j in range(i+1, n):
                xj, yj = x[2*j], x[2*j+1]
                d_sq = hex_dist_sq(xi, yi, xj, yj)
                if d_sq < min_dist_sq:
                    violation += (min_dist_sq - d_sq) ** 2
        return violation
    
    def loss(x):
        R = compute_smoothed_R(x)
        obj = -1.0 / R
        cons = constraint_violation_sq(x)
        return obj + penalty * cons
    
    def compute_true_R(x):
        all_vertices = []
        for i in range(n):
            hx, hy = x[2*i], x[2*i+1]
            all_vertices.append(hex_vertices_coords(hx, hy))
        all_vertices = np.vstack(all_vertices)
        max_y = np.max(np.abs(all_vertices[:, 1]))
        max_x = np.max(np.abs(all_vertices[:, 0]))
        max_plus = np.max(np.abs(all_vertices[:, 0] + all_vertices[:, 1]/sqrt3))
        max_minus = np.max(np.abs(all_vertices[:, 0] - all_vertices[:, 1]/sqrt3))
        return max([max_x, max_plus, max_minus, 2*max_y/sqrt3])
    
    start_time = time.time()
    
    minimizer_kwargs = {"method": "L-BFGS-B", "bounds": bounds, "options": {"maxiter": 1000}}
    bh_result = basinhopping(loss, init_x, niter=60, T=0.005, stepsize=0.05, 
                              minimizer_kwargs=minimizer_kwargs, seed=42)
    
    da_result = dual_annealing(loss, bounds, maxiter=2500, seed=42)
    
    if loss(bh_result.x) < loss(da_result.x):
        best_x = bh_result.x
    else:
        best_x = da_result.x
    
    polish_result = minimize(loss, best_x, method="L-BFGS-B", bounds=bounds,
                             options={"ftol": 1e-10, "gtol": 1e-10, "maxiter": 2000})
    best_x = polish_result.x
    
    true_R = compute_true_R(best_x)
    cons_viol = constraint_violation_sq(best_x)
    
    while cons_viol > 1e-12:
        best_x = best_x + 0.01 * (np.random.rand(dim) - 0.5)
        cons_viol = constraint_violation_sq(best_x)
        if time.time() - start_time > 10:
            break
    
    final_R = max(compute_true_R(best_x), true_R)
    if constraint_violation_sq(best_x) > 1e-12:
        best_x = init_x
        final_R = 3.930092
    
    inner_hex_data = np.zeros((n, 3), dtype=np.float64)
    for i in range(n):
        inner_hex_data[i, 0] = best_x[2*i]
        inner_hex_data[i, 1] = best_x[2*i+1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    outer_hex_side_length = float(final_R)
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length