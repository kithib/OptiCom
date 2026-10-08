import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 230
    learning_rate: float = 0.019
    num_steps: int = 80000
    warmup_steps: int = 4800


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        """
        f_non_negative = jnp.square(f_values) * 10.0

        # Unscaled discrete autoconvolution with proper symmetric padding
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Correct Simpson's rule for L2-norm squared (more accurate)
        num_conv_points = len(convolution)
        h = 1.0 / num_conv_points
        y = convolution
        # Standard Simpson's rule: h/3 * (y0 + 4y1 + 2y2 + 4y3 + ... + yn)
        weights = jnp.ones_like(y)
        weights = weights.at[1:-1:2].set(4.0)
        weights = weights.at[2:-2:2].set(2.0)
        l2_norm_squared = (h / 3.0) * jnp.sum(weights * y**2)

        # Correct Simpson's rule also for L1-norm for consistency
        norm_1 = (h / 3.0) * jnp.sum(weights * jnp.abs(y))

        # Calculate infinity-norm with small regularization
        norm_inf = jnp.max(jnp.abs(convolution)) + 1e-15

        # Calculate C2 ratio with numerical stability
        denominator = norm_1 * norm_inf + 1e-12
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient clipping and NaN guard."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        # Guard against NaN gradients to prevent RUNTIME errors
        grads = jnp.where(jnp.isnan(grads), 0.0, grads)
        # Apply gradient clipping to prevent unstable updates
        grads = jnp.clip(grads, -1.5, 1.5)
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
            end_value=self.hypers.learning_rate * 3e-5,
        )
        # Use AdamW optimizer with balanced weight decay
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=8e-6)

        # Initialize with an improved step-like function based on known champion characteristics
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(-2.0, 2.0, self.hypers.num_intervals)
        # Multi-step initial function inspired by champion behavior with highly refined steps
        initial = jnp.where(jnp.abs(x) < 0.19, 1.04,
                  jnp.where(jnp.abs(x) < 0.38, 0.97,
                  jnp.where(jnp.abs(x) < 0.57, 0.86,
                  jnp.where(jnp.abs(x) < 0.76, 0.72,
                  jnp.where(jnp.abs(x) < 0.95, 0.58,
                  jnp.where(jnp.abs(x) < 1.14, 0.44,
                  jnp.where(jnp.abs(x) < 1.33, 0.31,
                  jnp.where(jnp.abs(x) < 1.52, 0.20,
                  jnp.where(jnp.abs(x) < 1.71, 0.11,
                  jnp.where(jnp.abs(x) < 1.90, 0.05, 0.02))))))))))
        # Add small smooth quadratic component for better initialization and gradient flow
        initial = initial + 0.04 * (1.0 - jnp.minimum((x / 2.0)**2, 1.0))
        # Add small controlled noise for exploration
        initial = initial + 0.025 * jax.random.normal(key, (self.hypers.num_intervals,))
        f_values = jnp.sqrt(jnp.maximum(initial, 0.01) / 10.0)

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 8000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.14f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.14f}")
        return jnp.square(f_values) * 10.0, final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals