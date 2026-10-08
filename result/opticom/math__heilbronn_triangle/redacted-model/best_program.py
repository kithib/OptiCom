import numpy as np
from itertools import combinations
from scipy.optimize import basinhopping, dual_annealing, minimize
import time

SQRT3 = np.sqrt(3)
HEIGHT = SQRT3 / 2
N_POINTS = 11
BENCHMARK = 0.036529889880030156
ALPHA = 2200.0
PENALTY = 5500.0

TRIPLET_INDICES = np.array(list(combinations(range(N_POINTS), 3)))

CURRENT_BEST_X = np.array([
    0.00000000e+00, 0.00000000e+00,
    1.00000000e+00, 0.00000000e+00,
    5.00000000e-01, 8.66025404e-01,
    2.50000000e-01, 4.33012702e-01,
    7.50000000e-01, 4.33012702e-01,
    5.00000000e-01, 2.88675135e-01,
    1.25000000e-01, 2.16506351e-01,
    8.75000000e-01, 2.16506351e-01,
    3.75000000e-01, 6.49519053e-01,
    6.25000000e-01, 6.49519053e-01,
    5.00000000e-01, 0.00000000e+00
])

def project_to_triangle(x):
    points = x.copy().reshape((N_POINTS, 2))
    points[:, 1] = np.clip(points[:, 1], 0, HEIGHT)
    left_mask = points[:, 1] > SQRT3 * points[:, 0] + 1e-12
    points[left_mask, 0] = points[left_mask, 1] / SQRT3
    right_mask = points[:, 1] > SQRT3 * (1 - points[:, 0]) + 1e-12
    points[right_mask, 0] = 1 - points[right_mask, 1] / SQRT3
    points[:, 0] = np.clip(points[:, 0], 0, 1)
    return points.flatten()

def true_min_area(x):
    points = project_to_triangle(x).reshape((N_POINTS, 2))
    min_area = float('inf')
    for triplet in combinations(points, 3):
        a, b, c = triplet
        area = 0.5 * abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if area < min_area:
            min_area = area
    return min_area

def constraint_violations_sq(x):
    points = x.reshape((N_POINTS, 2))
    viol_x_low = np.maximum(0, -points[:, 0])**2
    viol_x_high = np.maximum(0, points[:, 0] - 1)**2
    viol_y_low = np.maximum(0, -points[:, 1])**2
    viol_y_high = np.maximum(0, points[:, 1] - HEIGHT)**2
    viol_left = np.maximum(0, points[:, 1] - SQRT3 * points[:, 0])**2
    viol_right = np.maximum(0, points[:, 1] - SQRT3 * (1 - points[:, 0]))**2
    return np.sum(viol_x_low) + np.sum(viol_x_high) + np.sum(viol_y_low) + np.sum(viol_y_high) + np.sum(viol_left) + np.sum(viol_right)

def loss(x, alpha=ALPHA, penalty=PENALTY):
    points = x.reshape((N_POINTS, 2))
    a = points[TRIPLET_INDICES[:, 0]]
    b = points[TRIPLET_INDICES[:, 1]]
    c = points[TRIPLET_INDICES[:, 2]]
    signed_area2 = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    areas = 0.5 * np.sqrt(signed_area2**2 + 1e-18)
    smooth_min = - (1.0 / alpha) * np.log(np.sum(np.exp(-alpha * areas)) + 1e-22)
    viol_sq = constraint_violations_sq(x)
    return -smooth_min + penalty * viol_sq

def heilbronn_triangle11() -> np.ndarray:
    np.random.seed(42)
    overall_start = time.time()
    
    dim = N_POINTS * 2
    bounds = [(0.0, 1.0), (0.0, HEIGHT)] * N_POINTS
    x0 = CURRENT_BEST_X.copy()
    
    MAX_OVERALL = 116
    BASINHOPPING_TIME = 58
    
    def bh_callback(x, f, accept):
        return time.time() - overall_start > min(BASINHOPPING_TIME, MAX_OVERALL - 18)
    
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 600}}
    bh_result = basinhopping(
        loss, x0, niter=90, T=0.004, stepsize=0.048,
        minimizer_kwargs=minimizer_kwargs, callback=bh_callback, seed=42
    )
    bh_loss_val = float(loss(bh_result.x))
    elapsed = time.time() - overall_start
    
    remaining = max(18, min(50, MAX_OVERALL - elapsed - 18))
    da_maxiter = max(2800, min(4500, int(4200 * remaining / 50)))
    da_result = dual_annealing(loss, bounds, maxiter=da_maxiter, initial_temp=5500.0, restart_temp_ratio=1.8e-5, seed=42)
    da_loss_val = float(loss(da_result.x))
    
    if da_loss_val < bh_loss_val:
        winner_x = da_result.x
    else:
        winner_x = bh_result.x
    
    polish_options = {'ftol': 1e-14, 'gtol': 1e-14, 'maxiter': 3200}
    polish_result = minimize(loss, winner_x, method='L-BFGS-B', bounds=bounds, options=polish_options)
    polished_x = polish_result.x
    
    points = project_to_triangle(polished_x).reshape((N_POINTS, 2))
    return points.astype(np.float64)

if __name__ == "__main__":
    start = time.time()
    pts = heilbronn_triangle11()
    elapsed = time.time() - start
    true_min = true_min_area(pts.flatten())
    print("Points shape:", pts.shape)
    print(f"True min triangle area: {true_min:.15f}")
    print(f"Combined score: {true_min / BENCHMARK:.6f}")
    print(f"Eval time: {elapsed:.2f}s")