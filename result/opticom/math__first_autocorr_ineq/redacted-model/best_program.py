import numpy as np
from scipy.optimize import minimize
from dataclasses import dataclass
import time


@dataclass
class Hyperparameters:
    num_intervals: int = 500
    learning_rate: float = 0.005
    end_lr_factor: float = 1e-4
    num_steps: int = 40000
    warmup_steps: int = 2000
    dx: float = 0.5 / 500


class AutocorrelationOptimizer:
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals
        self.N = self.hypers.num_intervals
        self.hypers.dx = self.dx
        self._alpha = 300.0
        self._conv_size = 2 * self.N

    def _incumbent_warmstart(self) -> np.ndarray:
        rng = np.random.default_rng(seed=42)
        idx = np.arange(self.N)
        center = self.N / 2.0
        half_width = self.N / 3.8
        triangle = np.maximum(0.0, 1.0 - np.abs(idx - center) / half_width) ** 1.2
        return triangle + 0.01 * rng.uniform(size=self.N)

    def _smoothed_loss_and_grad(self, f_values: np.ndarray):
        f_non_negative = np.maximum(f_values, 0.0)
        integral_f = np.sum(f_non_negative) * self.dx
        eps = 1e-9
        integral_f_safe = max(integral_f, eps)

        padded = np.zeros(self._conv_size)
        padded[: self.N] = f_non_negative
        fft_f = np.fft.rfft(padded)
        conv = np.fft.irfft(fft_f * fft_f, n=self._conv_size) * self.dx

        alpha = self._alpha
        shifted = alpha * conv
        m = np.max(shifted)
        exp_shifted = np.exp(shifted - m)
        sum_exp = np.sum(exp_shifted)
        soft_max = m + np.log(sum_exp)
        soft_max_conv = soft_max / alpha

        loss = float(soft_max_conv / (integral_f_safe**2))

        w = exp_shifted / sum_exp
        fft_w = np.fft.rfft(w)
        grad_padded = 2.0 * np.fft.irfft(fft_w * np.conj(fft_f), n=self._conv_size) * self.dx
        active = (f_values >= 0).astype(np.float64)
        grad_f = grad_padded[: self.N] * active

        grad_f = grad_f / (integral_f_safe**2) - 2.0 * soft_max_conv / (integral_f_safe**3) * self.dx * active
        grad_f /= alpha
        return loss, grad_f.astype(np.float64)

    def _loss_and_grad(self, f_values: np.ndarray):
        return self._smoothed_loss_and_grad(f_values)

    def _true_c1(self, f_values: np.ndarray) -> float:
        f_non_negative = np.maximum(f_values, 0.0)
        integral_f = np.sum(f_non_negative) * self.dx
        eps = 1e-9
        integral_f_safe = max(integral_f, eps)

        padded = np.zeros(self._conv_size)
        padded[: self.N] = f_non_negative
        fft_f = np.fft.rfft(padded)
        conv = np.fft.irfft(fft_f * fft_f, n=self._conv_size) * self.dx
        return float(np.max(conv) / (integral_f_safe**2))

    def _generate_starts(self, warm: np.ndarray, rng: np.random.Generator):
        starts = [warm.copy()]
        scale = np.maximum(np.max(np.abs(warm)), 1e-6)
        for _ in range(16):
            perturbed = warm + scale * 0.02 * rng.standard_normal(self.N)
            starts.append(np.maximum(perturbed, 0.0))

        idx = np.arange(self.N)
        center = self.N / 2.0
        for k in range(1, 5):
            freq = 2.0 * k * np.pi / self.N
            envelope = np.maximum(0.0, 1.0 - (np.abs(idx - center) / (self.N / 2.2)) ** 1.5)
            pattern = envelope * (1.0 + 0.25 * np.cos(freq * (idx - center)))
            pattern = pattern + 0.008 * rng.standard_normal(self.N)
            starts.append(np.maximum(pattern, 0.0))

        return starts

    def run_optimization(self):
        rng = np.random.default_rng(seed=42)
        primary_warm = self._incumbent_warmstart()
        starts = self._generate_starts(primary_warm, rng)

        bounds = [(0.0, None) for _ in range(self.N)]
        options = {"ftol": 1e-10, "gtol": 1e-10, "maxiter": 1500}

        best_f = np.maximum(primary_warm, 0.0)
        best_c1 = self._true_c1(best_f)
        start_time = time.time()
        time_limit = 85.0

        for i, s in enumerate(starts):
            if time.time() - start_time > time_limit:
                print(f"[WARN] Time budget reached after {i} starts.")
                break
            try:
                result = minimize(
                    fun=self._loss_and_grad,
                    x0=s,
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options=options,
                )
                candidate_f = np.maximum(result.x, 0.0)
                current_c1 = self._true_c1(candidate_f)
                if current_c1 < best_c1:
                    best_c1 = current_c1
                    best_f = candidate_f.copy()
                if i % 4 == 0 or i == len(starts) - 1:
                    print(
                        f"Start {i:3d} | smoothed_loss={result.fun:.8f} | true C1={current_c1:.10f} | best={best_c1:.10f}"
                    )
            except Exception as e:
                print(f"Start {i} failed: {e}")
                continue

        if time.time() - start_time < time_limit:
            try:
                polish = minimize(
                    fun=self._loss_and_grad,
                    x0=best_f.copy(),
                    jac=True,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"ftol": 1e-12, "gtol": 1e-12, "maxiter": 2000},
                )
                polished_f = np.maximum(polish.x, 0.0)
                polished_c1 = self._true_c1(polished_f)
                if polished_c1 < best_c1:
                    best_c1 = polished_c1
                    best_f = polished_f.copy()
                    print(f"Polish improved C1 to {best_c1:.12f}")
            except Exception as e:
                print(f"Polish failed: {e}")

        print(f"Final best true C1 found: {best_c1:.12f}")
        return best_f, best_c1


def run():
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)
    optimized_f, final_loss_val = optimizer.run_optimization()
    final_c1 = float(final_loss_val)
    f_values_np = np.array(optimized_f)
    return f_values_np, final_c1, final_loss_val, hypers.num_intervals


if __name__ == "__main__":
    f_vals, c1, loss_val, n = run()
    print(f"\n=== Final Results ===")
    print(f"C1 constant   : {c1:.12f}")
    print(f"Beat 1.5052939684401607? : {c1 < 1.5052939684401607}")
    print(f"num_intervals : {n}")
    print(f"min f : {f_vals.min():.6e}, max f: {f_vals.max():.6e}")