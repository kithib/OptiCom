import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1500
    learning_rate: float = 0.036
    num_steps: int = 175000
    warmup_steps: int = 8750


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
        eps = 1e-12
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

        # The max of the absolute value of the autoconvolution.
        N = self.hypers.num_intervals
        # Symmetric padding to properly center the convolution
        padded_f = jnp.pad(f_values, (N // 2, N // 2))

        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

        # Scale the unscaled convolution sum by dx to approximate the integral.
        scaled_conv_f_f = conv_f_f * self.dx

        # Take the maximum of the absolute value.
        max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))

        c3_ratio = max_abs_conv / integral_f_sq_safe

        # We want to MINIMIZE the ratio.
        return c3_ratio

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
            end_value=self.hypers.learning_rate * 6e-6,
        )
        # Optimized chain with adjusted clipping, betas and regularization
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.68),
            optax.adamw(learning_rate=schedule, b1=0.905, b2=0.9975, weight_decay=2.2e-6)
        )

        # Enhanced initialization with additional tuned frequency components
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(-1.0, 1.0, self.hypers.num_intervals)
        # Multi-modal basis with refined components and added higher frequencies
        f1 = jnp.sin(3.0 * jnp.pi * x) * jnp.exp(-3.45 * x**2)
        f2 = 0.465 * jnp.sin(7.0 * jnp.pi * x) * jnp.exp(-4.35 * x**2)
        f3 = 0.23 * jnp.cos(1.0 * jnp.pi * x) * jnp.exp(-1.85 * x**2)
        f4 = -0.185 * jnp.sin(5.0 * jnp.pi * x) * jnp.exp(-3.78 * x**2)
        f5 = 0.108 * jnp.sin(9.0 * jnp.pi * x) * jnp.exp(-4.95 * x**2)
        f6 = -0.063 * jnp.cos(11.0 * jnp.pi * x) * jnp.exp(-5.55 * x**2)
        f7 = 0.038 * jnp.sin(13.0 * jnp.pi * x) * jnp.exp(-6.15 * x**2)
        f8 = -0.022 * jnp.cos(15.0 * jnp.pi * x) * jnp.exp(-6.78 * x**2)
        f9 = 0.011 * jnp.sin(17.0 * jnp.pi * x) * jnp.exp(-7.45 * x**2)
        f10 = -0.0048 * jnp.cos(19.0 * jnp.pi * x) * jnp.exp(-8.15 * x**2)
        # Additional components for improved basis coverage
        f11 = 0.088 * jnp.cos(2.0 * jnp.pi * x) * jnp.exp(-2.48 * x**2)
        f12 = 0.0018 * jnp.sin(21.0 * jnp.pi * x) * jnp.exp(-8.85 * x**2)
        f13 = -0.0009 * jnp.cos(23.0 * jnp.pi * x) * jnp.exp(-9.55 * x**2)
        f_base = f1 + f2 + f3 + f4 + f5 + f6 + f7 + f8 + f9 + f10 + f11 + f12 + f13
        f_noise = jax.random.normal(key, (self.hypers.num_intervals,)) * 0.013
        f_values = f_base + f_noise

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_loss = jnp.inf
        best_f = f_values
        # More sensitive plateau tracking with adaptive step frequency
        plateau_counter = 0
        prev_best = jnp.inf

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_f = f_values

            # Early plateau detection with increased sensitivity
            if step % 2800 == 0 and step > 0:
                if best_loss >= prev_best * 0.999985:
                    plateau_counter += 1
                else:
                    plateau_counter = 0
                prev_best = best_loss

            if step % 2500 == 0 or step == self.hypers.num_steps - 1:
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