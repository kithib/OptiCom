# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 50
    learning_rate: float = 0.01
    num_steps: int = 15000
    warmup_steps: int = 1000


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the mathematical simplification.
        Objective: Maximize C2 = ||f ★ f||₂² / ((∫f)² * ||f ★ f||_{∞})
        """
        f_non_negative = jax.nn.relu(f_values)

        # Scale f to have L1 norm = 1/num_intervals so integral ≈ 1
        # This makes (∫f)² ≈ 1 and simplifies the optimization
        f_non_negative = f_non_negative + 1e-8
        f_sum = jnp.sum(f_non_negative)
        f_scaled = f_non_negative / f_sum * self.hypers.num_intervals

        # Discrete autoconvolution via FFT
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_scaled, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # L2-norm squared of convolution using Simpson's rule
        num_conv_points = len(convolution)
        h = 1.0 / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))

        # Infinity norm of convolution
        norm_inf = jnp.max(jnp.abs(convolution) + 1e-10)

        # (∫f)² - using scaled f this should be approximately 1
        integral_f_sq = (jnp.sum(f_scaled) / N) ** 2

        # Calculate C2 ratio
        c2_ratio = l2_norm_squared / (integral_f_sq * norm_inf)

        return -c2_ratio

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
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        # Improved initialization using domain knowledge: step function pattern
        key = jax.random.PRNGKey(42)
        # Initialize with step function pattern similar to known good solutions
        x = jnp.linspace(0, 1, self.hypers.num_intervals)
        # Create peaks at strategic positions
        f_values = 0.5 + 0.5 * jnp.sin(3 * jnp.pi * x) + jax.random.uniform(key, x.shape) * 0.1

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.10f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.12f}")
        return jax.nn.relu(f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END