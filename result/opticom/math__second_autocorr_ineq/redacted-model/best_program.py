# EVOLVE-BLOCK-START
import numpy as np
from dataclasses import dataclass
from scipy.optimize import basinhopping, dual_annealing, minimize

@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 128
    learning_rate: float = 0.01
    num_steps: int = 15000
    warmup_steps: int = 1000

def logsumexp_smooth_max(x, alpha=800.0):
    """Smooth approximation to max using LogSumExp."""
    # Shift for numerical stability
    x_max = np.max(x)
    return x_max + np.log(np.sum(np.exp(alpha * (x - x_max)))) / alpha

def compute_true_c2(f_values):
    """Compute true (non-smoothed, non-penalized) C2 value for reporting."""
    f_non_negative = np.maximum(f_values, 0.0)
    N = len(f_non_negative)
    h = 1.0 / N
    n_fft = 2 * N
    fft_f = np.fft.fft(f_non_negative, n=n_fft)
    convolution = np.fft.ifft(fft_f * fft_f).real * h
    h_conv = h
    y = convolution
    # L2-norm squared with Simpson's rule
    l2_norm_squared = np.sum(y[1:-1]**2) * 2 + y[0]**2 + y[-1]**2
    l2_norm_squared = l2_norm_squared * h_conv / 3.0
    # L1-norm with trapezoidal rule
    norm_1 = (np.sum(convolution[1:-1]) + 0.5 * (convolution[0] + convolution[-1])) * h_conv
    # True max for reporting
    norm_inf = np.max(convolution)
    denominator = norm_1 * norm_inf + 1e-15
    c2_ratio = l2_norm_squared / denominator
    return c2_ratio, l2_norm_squared, norm_1, norm_inf

def make_loss(N):
    """Create smooth differentiable loss for N-dimensional decision vector."""
    def loss(x, alpha=800.0, penalty=10.0):
        # Decision variable x is flat float vector of dim N
        # Smooth non-negative transform for gradient flow
        f_non_negative = np.maximum(x, 0.0)
        h = 1.0 / N
        n_fft = 2 * N
        fft_f = np.fft.fft(f_non_negative, n=n_fft)
        convolution = np.fft.ifft(fft_f * fft_f).real * h
        h_conv = h
        y = convolution
        # L2-norm squared with Simpson's rule (smooth)
        l2_norm_squared = np.sum(y[1:-1]**2) * 2 + y[0]**2 + y[-1]**2
        l2_norm_squared = l2_norm_squared * h_conv / 3.0
        # L1-norm with trapezoidal rule (smooth)
        norm_1 = (np.sum(convolution[1:-1]) + 0.5 * (convolution[0] + convolution[-1])) * h_conv
        # Smooth max using LogSumExp with large alpha
        norm_inf_smooth = logsumexp_smooth_max(convolution, alpha=alpha)
        # Smooth approx of objective
        denominator = norm_1 * norm_inf_smooth + 1e-15
        c2_ratio_smooth = l2_norm_squared / denominator
        # Constraint violation: sum of squares of negative parts of x
        neg_constraint = np.sum(np.maximum(-x, 0.0)**2)
        # loss = -smooth_objective + penalty * constraint_violation
        return -c2_ratio_smooth + penalty * neg_constraint
    return loss

class C2Optimizer:
    """Optimizes using numerical optimization: basinhopping + dual_annealing + polished L-BFGS-B"""
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def run_optimization(self):
        N = self.hypers.num_intervals
        loss_fn = make_loss(N)
        # Bounds on decision variables: reasonable range for function values
        bounds = [(0.0, 15.0) for _ in range(N)]
        # Determinism: seed all randomness
        np.random.seed(42)
        # Initial guess: starting from a reasonable prior shape inspired by step-function champion
        x_grid = np.linspace(0, 1, N)
        x0 = np.where(x_grid < 0.33, 1.0, np.where(x_grid < 0.66, 0.6, 0.3))
        x0 = np.clip(x0 + np.random.randn(N) * 0.05, 0.0, 15.0)

        # Multi-start strategy 1: basinhopping from current best layout
        print(f"Running basin hopping (niter=60)...")
        minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds, 'options': {'maxiter': 600}}
        bh_result = basinhopping(loss_fn, x0, niter=60, T=0.005, stepsize=0.05, minimizer_kwargs=minimizer_kwargs, seed=42, niter_success=15)
        bh_loss = float(loss_fn(bh_result.x))
        print(f"Basin hopping final loss: {bh_loss:.12f}")

        # Multi-start strategy 2: dual annealing as global fallback
        print(f"Running dual annealing (maxiter=2500)...")
        da_result = dual_annealing(loss_fn, bounds, maxiter=2500, seed=42)
        da_loss = float(loss_fn(da_result.x))
        print(f"Dual annealing final loss: {da_loss:.12f}")

        # Pick whichever achieves lower loss
        if bh_loss < da_loss:
            best_x = bh_result.x
            best_loss = bh_loss
            print("Selected basin hopping result")
        else:
            best_x = da_result.x
            best_loss = da_loss
            print("Selected dual annealing result")

        # Polish winner with tight convergence
        print(f"Polishing with tight L-BFGS-B (ftol=1e-10, gtol=1e-10)...")
        polish_result = minimize(loss_fn, best_x, method='L-BFGS-B', bounds=bounds, options={'ftol':1e-10, 'gtol':1e-10, 'maxiter':2500})
        # Project back to feasible region: enforce non-negativity directly
        final_x = np.clip(polish_result.x, 0.0, None)
        # Recompute metrics with true non-smoothed objective for reporting
        final_c2, _, _, _ = compute_true_c2(final_x)
        print(f"Final polished loss: {polish_result.fun:.12f}")
        print(f"Final C2 lower bound found: {final_c2:.14f}")
        return final_x, final_c2

def run():
    """Entry point for running the optimization (public API preserved)."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()
    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)
    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals

# EVOLVE-BLOCK-END