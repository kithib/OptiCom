import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1750
    learning_rate: float = 0.036
    num_steps: int = 155000
    warmup_steps: int = 8500


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
        Uses symmetric padding and tighter epsilon for more accurate evaluation.
        """
        # The squared integral of f.
        integral_f = jnp.sum(f_values) * self.dx
        eps = 1e-20
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

        # Symmetric padding captures full t range [-1/2, 1/2] for more accurate convolution
        N = self.hypers.num_intervals
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
            end_value=self.hypers.learning_rate * 1.2e-7,
        )
        # AdamW with refined weight decay for better stability and fine-grained convergence
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=4.8e-7)

        # Enhanced structured initialization: carefully tuned amplitudes based on insights from prior runs
        # Extended higher frequency components to capture finer oscillations with empirically derived amplitudes
        key = jax.random.PRNGKey(42)
        n = self.hypers.num_intervals
        x = jnp.linspace(-0.25, 0.25, n, endpoint=False)
        f_values = (
            jnp.cos(4 * jnp.pi * x) * 1.0
            + jnp.cos(12 * jnp.pi * x) * 0.643
            + jnp.cos(20 * jnp.pi * x) * 0.443
            + jnp.cos(28 * jnp.pi * x) * 0.323
            + jnp.cos(36 * jnp.pi * x) * 0.223
            + jnp.cos(44 * jnp.pi * x) * 0.143
            + jnp.cos(52 * jnp.pi * x) * 0.091
            + jnp.cos(60 * jnp.pi * x) * 0.057
            + jnp.cos(68 * jnp.pi * x) * 0.034
            + jnp.cos(76 * jnp.pi * x) * 0.018
            + jnp.cos(84 * jnp.pi * x) * 0.0095
            + jnp.cos(92 * jnp.pi * x) * 0.0045
            + jnp.cos(100 * jnp.pi * x) * 0.0020
            + jnp.cos(108 * jnp.pi * x) * 0.0009
            + jnp.cos(116 * jnp.pi * x) * 0.00035
            + jnp.sin(4 * jnp.pi * x) * 0.171
            + jnp.sin(12 * jnp.pi * x) * 0.101
            + jnp.sin(20 * jnp.pi * x) * 0.061
            + jnp.sin(28 * jnp.pi * x) * 0.039
            + jnp.sin(36 * jnp.pi * x) * 0.029
            + jnp.sin(44 * jnp.pi * x) * 0.019
            + jnp.sin(52 * jnp.pi * x) * 0.011
            + jnp.sin(60 * jnp.pi * x) * 0.0060
            + jnp.sin(68 * jnp.pi * x) * 0.0030
            + jnp.sin(76 * jnp.pi * x) * 0.0015
        )
        # Slightly reduced noise for more focused exploration near the promising region
        f_values = f_values + jax.random.normal(key, (n,)) * 0.0145

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_loss = jnp.inf
        best_f = f_values
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Track best solution found
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C3 ≈ {loss:.12f} | Best ≈ {best_loss:.12f}")

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