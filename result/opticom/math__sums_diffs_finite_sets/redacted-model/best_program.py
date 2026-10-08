import numpy as np
from dataclasses import dataclass
from scipy.optimize import minimize, basinhopping, dual_annealing
import warnings
import time
warnings.filterwarnings('ignore')

@dataclass
class Hyperparameters:
    max_integer: int = 180
    alpha_smooth: float = 450.0
    penalty_weight: float = 90.0

def sigmoid(x, alpha=9.0):
    return 1.0 / (1.0 + np.exp(-np.clip(alpha * x, -500, 500)))

def smooth_approx_objective(x, hypers):
    alpha = hypers.alpha_smooth
    M = hypers.max_integer
    s_vals = sigmoid(9.0 * x)
    s0 = s_vals[0]
    penalty_0 = (1.0 - s0) ** 2
    n_smooth = np.sum(s_vals)
    if n_smooth < 0.5:
        return 100.0, 100.0
    i_arr = np.arange(M + 1, dtype=np.float64)
    log_terms = alpha * i_arr * s_vals
    max_log = np.max(log_terms)
    max_smooth = (max_log + np.log(np.sum(np.exp(log_terms - max_log)))) / alpha
    if max_smooth < 0.1:
        return 10.0, 10.0
    fft_len = 2 * M + 1
    fft_s = np.fft.fft(s_vals, fft_len)
    conv_sum = np.real(np.fft.ifft(fft_s * fft_s))
    log_sum_terms = alpha * conv_sum
    lse_max = np.max(log_sum_terms)
    sumset_smooth = (lse_max + np.log(np.sum(np.exp(log_sum_terms - lse_max)))) / alpha
    fft_s_rev = np.fft.fft(s_vals[::-1], fft_len)
    conv_diff = np.real(np.fft.ifft(fft_s * fft_s_rev))
    log_diff_terms = alpha * conv_diff
    lse_max_d = np.max(log_diff_terms)
    diffset_smooth = (lse_max_d + np.log(np.sum(np.exp(log_diff_terms - lse_max_d)))) / alpha
    if sumset_smooth < 0.1 or diffset_smooth < 0.1:
        return 5.0, 5.0
    ratio_smooth = diffset_smooth / sumset_smooth
    denom = np.log(2.0 * max_smooth + 1.0)
    if denom < 1e-8:
        return 2.0, 2.0
    c6_smooth = 1.0 + np.log(np.maximum(ratio_smooth, 0.1)) / denom
    return -c6_smooth, penalty_0

def composite_loss(x, hypers):
    loss_obj, penalty = smooth_approx_objective(x, hypers)
    return loss_obj + hypers.penalty_weight * penalty

def true_objective_from_x(x, hypers):
    x_proj = np.clip(x, 0.0, 1.0)
    mask = x_proj > 0.5
    mask[0] = True
    U = np.where(mask)[0]
    if len(U) < 2:
        return -1.0, U
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    size_U_plus_U = len(np.unique(sums))
    size_U_minus_U = len(np.unique(diffs))
    max_U = U[-1]
    if max_U == 0:
        return -1.0, U
    ratio = size_U_minus_U / size_U_plus_U
    c6_bound = 1 + np.log(ratio) / np.log(2 * max_U + 1)
    return c6_bound, U

def get_incumbent_warm_start(hypers, rng):
    dim = hypers.max_integer + 1
    x = np.zeros(dim, dtype=np.float64)
    high_value_indices = [
        0, 1, 3, 4, 9, 11, 12, 13, 14, 16, 17, 21, 23, 26,
        27, 30, 34, 38, 39, 41, 42, 43, 44, 45, 46, 47, 48,
        52, 53, 54, 57, 59, 60, 63, 65, 69, 71, 72, 75, 76,
        78, 80, 83, 85, 86, 87, 89, 90, 91, 93, 95, 98, 99,
        102, 103, 104, 105, 108, 110, 111, 113, 114, 115, 116,
        119, 120, 123, 124, 126, 129, 131, 132, 134, 135, 137,
        138, 141, 143, 144, 145, 147, 149, 150, 152, 153, 155,
        156, 159, 161, 162, 164, 165, 167, 168, 170, 171, 173,
        174, 176, 177, 179
    ]
    for idx in high_value_indices:
        if idx < dim:
            x[idx] = 0.85 + 0.1 * rng.random()
    for i in range(1, min(65, dim)):
        if x[i] < 0.2:
            x[i] = 0.15 * rng.random()
    return x

def local_polish_set(U, hypers, iterations=60):
    if len(U) < 2:
        return U, -1.0
    M = hypers.max_integer
    mask = np.zeros(M + 1, dtype=bool)
    mask[U] = True
    def obj(m):
        Us = np.where(m)[0]
        if len(Us) < 2:
            return -1.0
        sums = Us[:, None] + Us[None, :]
        diffs = Us[:, None] - Us[None, :]
        ratio = len(np.unique(diffs)) / len(np.unique(sums))
        return 1 + np.log(ratio) / np.log(2 * Us[-1] + 1)
    best_c6 = obj(mask)
    improved = True
    iter_count = 0
    while improved and iter_count < iterations:
        improved = False
        for i in range(1, M + 1):
            new_mask = mask.copy()
            new_mask[i] = not new_mask[i]
            if not new_mask[0]:
                continue
            current_c6 = obj(new_mask)
            if current_c6 > best_c6:
                mask = new_mask
                best_c6 = current_c6
                improved = True
        iter_count += 1
    return np.where(mask)[0], best_c6

def run():
    t0 = time.time()
    hypers = Hyperparameters()
    dim = hypers.max_integer + 1
    bounds = [(0.0, 1.0)] * dim
    rng = np.random.default_rng(seed=42)
    
    warm_start = get_incumbent_warm_start(hypers, rng)
    best_true_c6 = -float('inf')
    best_set = None
    
    def opt_loss_vec(x):
        return composite_loss(x, hypers)
    
    print("Running multi-start L-BFGS-B with smooth composite loss...")
    starts = [warm_start.copy()]
    for _ in range(6):
        noisy_x = warm_start + rng.normal(0, 0.02, size=dim)
        noisy_x = np.clip(noisy_x, 0.0, 1.0)
        noisy_x[0] = 1.0
        starts.append(noisy_x)
    
    for x0 in starts:
        result = minimize(opt_loss_vec, x0, method='L-BFGS-B', bounds=bounds,
                          options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 1000})
        c6_val, current_set = true_objective_from_x(result.x, hypers)
        if c6_val > best_true_c6 and len(current_set) >= 2:
            best_true_c6 = c6_val
            best_set = current_set.copy()
    print(f"Best after multi-start: {best_true_c6:.10f}")
    
    print("Running Basin-Hopping global search...")
    x0_bh = warm_start.copy()
    minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds,
                        'options': {'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 600}}
    bh_result = basinhopping(opt_loss_vec, x0_bh, niter=50, T=0.005, stepsize=0.05,
                              minimizer_kwargs=minimizer_kwargs, seed=42)
    bh_c6, bh_set = true_objective_from_x(bh_result.x, hypers)
    print(f"Basin-Hopping loss: {bh_result.fun:.6f}, True C6: {bh_c6:.10f}")
    if bh_c6 > best_true_c6 and len(bh_set) >= 2:
        best_true_c6 = bh_c6
        best_set = bh_set.copy()
    
    print("Running Dual Annealing global search...")
    da_result = dual_annealing(opt_loss_vec, bounds, maxiter=1200, seed=42)
    da_c6, da_set = true_objective_from_x(da_result.x, hypers)
    print(f"Dual Annealing loss: {da_result.fun:.6f}, True C6: {da_c6:.10f}")
    if da_c6 > best_true_c6 and len(da_set) >= 2:
        best_true_c6 = da_c6
        best_set = da_set.copy()
    
    print("Polishing best candidate with tight L-BFGS-B...")
    if best_set is not None:
        x_polish = np.zeros(dim, dtype=np.float64)
        x_polish[best_set] = 1.0
        polish_result = minimize(opt_loss_vec, x_polish, method='L-BFGS-B', bounds=bounds,
                                  options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 1200})
        polish_c6, polish_set = true_objective_from_x(polish_result.x, hypers)
        if polish_c6 > best_true_c6 and len(polish_set) >= 2:
            best_true_c6 = polish_c6
            best_set = polish_set.copy()
    
    print("Running final local polish...")
    if best_set is not None:
        final_set, final_c6 = local_polish_set(best_set, hypers, iterations=60)
        if final_c6 > best_true_c6:
            best_true_c6 = final_c6
            best_set = final_set
    
    print(f"\nTotal runtime: {time.time() - t0:.1f} seconds")
    print(f"Best C6 bound: {best_true_c6:.14f}")
    print(f"Current target: 1.158417281556896")
    
    return best_set if best_set is not None else np.array([0]), best_true_c6

def run_single_trial(hypers, key):
    rng_trial = np.random.default_rng(seed=hash(key) % (2**32 - 1))
    u_mask = np.zeros(hypers.max_integer + 1, dtype=bool)
    u_mask[0] = True
    for _ in range(22):
        idx = rng_trial.integers(1, hypers.max_integer + 1)
        u_mask[idx] = True
    def obj(mask):
        U = np.where(mask)[0]
        if len(U) < 2:
            return 1.0
        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]
        ratio = len(np.unique(diffs)) / len(np.unique(sums))
        return -(1 + np.log(ratio) / np.log(2 * U[-1] + 1))
    current_loss = obj(u_mask)
    for step in range(300):
        temp = 0.02 * np.exp(-3.0 * step / 300)
        idx = rng_trial.integers(1, len(u_mask))
        new_mask = u_mask.copy()
        new_mask[idx] = not new_mask[idx]
        new_loss = obj(new_mask)
        if new_loss < current_loss or rng_trial.random() < np.exp((current_loss - new_loss) / temp):
            u_mask, current_loss = new_mask, new_loss
    return current_loss, np.where(u_mask)[0]

if __name__ == "__main__":
    run()