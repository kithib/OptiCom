import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 3500
    learning_rate: float = 0.00110
    num_steps: int = 260000
    warmup_steps: int = 14000
    min_integral_sq: float = 1.5e-3
    soft_temp: float = 0.0088
    grad_clip: float = 2.20
    finetune_steps: int = 70000
    finetune_lr: float = 4e-5
    ft_soft_temp: float = 0.0022


class C3Optimizer:
    """
    Optimizes a function f to find an upper bound for the C3 constant.

    Evaluator-matched numerics:
    - ∫f = dx * sum(f_values),  dx = 0.5 / N
    - (f★f)(t) ≈ dx * numpy.convolve(f_values, f_values, mode='full')
    - C3 = max |conv| / (∫f)^2
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals
        self._active_soft_temp = hypers.soft_temp

    def compute_true_c3(self, f_values: jnp.ndarray) -> tuple:
        """
        Computes the TRUE C3 exactly as the evaluator does.
        Returns (c3, integral_f_sq).
        """
        N = self.hypers.num_intervals
        dx = self.dx

        integral_f = jnp.sum(f_values) * dx
        integral_f_sq = integral_f * integral_f
        eps = 1e-9
        integral_f_sq_safe = jnp.maximum(integral_f_sq, eps)

        # Full linear convolution matching numpy.convolve(..., mode='full').
        padded_f = jnp.pad(f_values, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real[: 2 * N - 1]
        scaled_conv = conv_f_f * dx
        max_abs_conv = jnp.max(jnp.abs(scaled_conv))

        c3 = max_abs_conv / integral_f_sq_safe
        return c3, integral_f_sq

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Smooth training loss matching true C3.
        - Soft (log-sum-exp) max for gradient flow.
        - Aggressive barrier to keep ∫² away from zero.
        """
        N = self.hypers.num_intervals
        dx = self.dx

        integral_f = jnp.sum(f_values) * dx
        integral_f_sq = integral_f * integral_f
        eps_denom = 1e-6
        integral_f_sq_safe = jnp.maximum(integral_f_sq, eps_denom)

        padded_f = jnp.pad(f_values, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real[: 2 * N - 1]
        scaled_conv = conv_f_f * dx
        valid = jnp.abs(scaled_conv)

        # Soft max
        max_valid = jnp.max(valid)
        shifted = (valid - max_valid) / self._active_soft_temp
        soft_max = max_valid + self._active_soft_temp * jnp.log(jnp.sum(jnp.exp(shifted)) + 1e-12)

        # Barrier: penalty grows quickly when ∫² < min_integral_sq
        barrier = 700.0 * jnp.where(
            integral_f_sq < self.hypers.min_integral_sq,
            (self.hypers.min_integral_sq - integral_f_sq) ** 2,
            0.0,
        )

        return (soft_max / integral_f_sq_safe) + barrier

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Single training step with gradient clipping for stability."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Run the optimization with warmup-cosine schedule and best-solution tracking."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.grad_clip),
            optax.adam(learning_rate=schedule),
        )

        key = jax.random.PRNGKey(42)
        # Refined smooth initial guess with higher harmonics + tiny jitter.
        x = jnp.linspace(-0.25, 0.25, self.hypers.num_intervals)
        base = (1.0
                + 0.630 * jnp.cos(2.0 * jnp.pi * x * 2.0)
                - 0.320 * jnp.cos(2.0 * jnp.pi * x * 4.0)
                + 0.120 * jnp.cos(2.0 * jnp.pi * x * 6.0)
                - 0.043 * jnp.cos(2.0 * jnp.pi * x * 8.0)
                + 0.0140 * jnp.cos(2.0 * jnp.pi * x * 10.0)
                - 0.0044 * jnp.cos(2.0 * jnp.pi * x * 12.0)
                + 0.0014 * jnp.cos(2.0 * jnp.pi * x * 14.0)
                - 0.00045 * jnp.cos(2.0 * jnp.pi * x * 16.0))
        perturbation = jax.random.normal(key, (self.hypers.num_intervals,)) * 0.016
        f_values = base + perturbation

        opt_state = self.optimizer.init(f_values)
        print(
            f"N={self.hypers.num_intervals}, steps={self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)
        true_c3_jit = jax.jit(self.compute_true_c3)

        loss = jnp.inf
        best_c3_true = jnp.inf
        best_f = f_values
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            true_c3, integral_sq = true_c3_jit(f_values)
            if true_c3 < best_c3_true:
                best_c3_true = true_c3
                best_f = f_values
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(
                    f"Step {step:6d} | soft C3 ≈ {loss:.10f} | true C3 ≈ {true_c3:.10f} | best ≈ {best_c3_true:.10f} | ∫²≈ {integral_sq:.6f}"
                )

        # Finetune phase: smaller LR, tighter soft_temp
        print(f"\nStarting finetune phase from best C3 = {best_c3_true:.10f}")
        self._active_soft_temp = self.hypers.ft_soft_temp
        finetune_schedule = optax.constant_schedule(self.hypers.finetune_lr)
        finetune_optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.grad_clip),
            optax.adam(learning_rate=finetune_schedule, b1=0.9, b2=0.999),
        )
        ft_state = finetune_optimizer.init(best_f)

        @jax.jit
        def ft_step(f_vals, state):
            loss_ft, grads_ft = jax.value_and_grad(self._objective_fn)(f_vals)
            updates_ft, state_new = finetune_optimizer.update(grads_ft, state, f_vals)
            f_vals_new = optax.apply_updates(f_vals, updates_ft)
            return f_vals_new, state_new, loss_ft

        f_values = best_f
        for step in range(self.hypers.finetune_steps):
            f_values, ft_state, loss = ft_step(f_values, ft_state)
            true_c3, integral_sq = true_c3_jit(f_values)
            if true_c3 < best_c3_true:
                best_c3_true = true_c3
                best_f = f_values
            if step % 10000 == 0 or step == self.hypers.finetune_steps - 1:
                print(
                    f"FT Step {step:5d} | soft C3 ≈ {loss:.10f} | true C3 ≈ {true_c3:.10f} | best ≈ {best_c3_true:.10f} | ∫²≈ {integral_sq:.6f}"
                )

        final_c3 = best_c3_true
        print(f"Final C3 upper bound found: {final_c3:.10f}")
        return best_f, final_c3


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()

    loss_val = float(final_c3_val)
    f_values_np = np.array(optimized_f)

    return f_values_np, loss_val, loss_val, hypers.num_intervals