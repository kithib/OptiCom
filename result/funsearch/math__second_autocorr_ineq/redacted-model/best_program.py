# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import math


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 128
    learning_rate: float = 0.015
    num_steps: int = 25000
    warmup_steps: int = 1500
    clip_norm: float = 10.0


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
        f_non_negative = jax.nn.softplus(f_values)

        # Use symmetry: create even function (f(-x) = f(x)) 
        # to exploit known symmetry properties of optimal functions
        f_symmetric = jnp.concatenate([f_non_negative[::-1], f_non_negative])

        # Unscaled discrete autoconvolution with proper padding
        N = len(f_symmetric)
        padded_f = jnp.pad(f_symmetric, (N, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate L2-norm squared of the convolution (rigorous method)
        num_conv_points = len(convolution)
        h = 1.0 / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate L1-norm of the convolution
        norm_1 = jnp.sum(jnp.abs(convolution)) * h

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(jnp.abs(convolution))

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
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
            end_value=self.hypers.learning_rate * 5e-5,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.clip_norm),
            optax.adam(learning_rate=schedule)
        )

        # Initialize with step-function-like pattern inspired by AlphaEvolve champion
        key = jax.random.PRNGKey(42)
        n = self.hypers.num_intervals
        
        # Create a promising initialization: centered peak pattern
        x = jnp.linspace(0, 1, n)
        mu1, mu2, mu3 = 0.2, 0.5, 0.8
        sigma = 0.15
        f_values = 0.5 + 2.0 * (
            jnp.exp(-((x - mu1)**2) / (2 * sigma**2)) +
            jnp.exp(-((x - mu2)**2) / (2 * sigma**2)) +
            jnp.exp(-((x - mu3)**2) / (2 * sigma**2))
        )
        # Add small noise for exploration
        f_values = f_values + 0.05 * jax.random.uniform(key, (n,))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_c2 = 0.0
        best_f = f_values
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            current_c2 = -loss
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f = f_values
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {current_c2:.10f} | Best C2 ≈ {best_c2:.10f}")

        final_c2 = -self._objective_fn(best_f)
        print(f"Final C2 lower bound found: {final_c2:.12f}")
        return jax.nn.softplus(best_f), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END