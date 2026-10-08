# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1600
    learning_rate: float = 0.0118
    num_steps: int = 140000
    warmup_steps: int = 8500
    regul_weight: float = 6.8e-7
    adamw_weight_decay: float = 1.35e-5
    second_diff_coeff: float = 0.32


class C3Optimizer:
    """
    Optimizes a function f (with positive and negative values) to find an
    upper bound for the C3 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the C3 ratio. The goal is to minimize this value.
        """
        # The squared integral of f.
        integral_f = jnp.sum(f_values) * self.dx
        eps = 1e-16
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

        # The max of the absolute value of the autoconvolution.
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_values, (N, N))

        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

        # Scale the unscaled convolution sum by dx to approximate the integral.
        scaled_conv_f_f = conv_f_f * self.dx

        # Take the maximum of the absolute value.
        max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))

        c3_ratio = max_abs_conv / integral_f_sq_safe

        # Regularize both first and second differences to smooth the function
        # while allowing for sharp but controlled changes
        diff1 = jnp.diff(f_values)
        diff2 = jnp.diff(diff1)
        regul = self.hypers.regul_weight * (
            jnp.sum(diff1 * diff1) + self.hypers.second_diff_coeff * jnp.sum(diff2 * diff2)
        )

        # We want to MINIMIZE the ratio.
        return c3_ratio + regul

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 7.5e-6,
        )
        self.optimizer = optax.adamw(
            learning_rate=schedule, weight_decay=self.hypers.adamw_weight_decay
        )

        # Initialize with further refined basis function approximation based on known good shapes
        x = jnp.linspace(-0.25, 0.25, self.hypers.num_intervals)
        f_values = (
            jnp.cos(4.183 * jnp.pi * x)
            - 0.3755 * jnp.cos(12.818 * jnp.pi * x)
            + 0.1183 * jnp.sin(6.838 * jnp.pi * x)
            - 0.0783 * jnp.cos(20.455 * jnp.pi * x)
            + 0.0283 * jnp.sin(15.075 * jnp.pi * x)
            + 0.0183 * jnp.cos(28.125 * jnp.pi * x)
            - 0.0103 * jnp.sin(24.065 * jnp.pi * x)
            + 0.0048 * jnp.cos(34.02 * jnp.pi * x)
            - 0.0025 * jnp.sin(39.85 * jnp.pi * x)
        )

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_loss = jnp.inf
        best_f = f_values
        plateau_count = 0
        prev_best = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            # Check for improvement every 1000 steps
            if step % 1000 == 0 and step > 0:
                if best_loss < prev_best - 8e-14:
                    prev_best = best_loss
                    plateau_count = 0
                else:
                    plateau_count += 1
                # Early stopping if no improvement for many checks
                if plateau_count >= 32:
                    print(f"Early stopping at step {step} - no significant improvement")
                    break
            if step % 3500 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C3 ≈ {loss:.14f} | Best ≈ {best_loss:.14f}")

        final_c3 = best_loss
        print(f"Final C3 upper bound found: {final_c3:.14f}")
        return best_f, final_c3


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()

    loss_val = final_c3_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c3_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END