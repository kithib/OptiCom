# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 100
    learning_rate: float = 0.015
    num_steps: int = 25000
    warmup_steps: int = 1500


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
        f_non_negative = jnp.square(f_values)

        # Unscaled discrete autoconvolution - using proper FFT padding
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate norms using consistent trapezoidal integration
        h = 1.0 / N
        
        # L2-norm squared of convolution (trapezoidal rule)
        conv_sq = convolution ** 2
        l2_norm_squared = h * (0.5 * conv_sq[0] + jnp.sum(conv_sq[1:-1]) + 0.5 * conv_sq[-1])

        # L1-norm of convolution (trapezoidal rule)
        conv_abs = jnp.abs(convolution)
        norm_1 = h * (0.5 * conv_abs[0] + jnp.sum(conv_abs[1:-1]) + 0.5 * conv_abs[-1])

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
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-6)

        key = jax.random.PRNGKey(42)
        f_values = jnp.sqrt(jax.random.uniform(key, (self.hypers.num_intervals,)))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.10f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.10f}")
        return jnp.square(f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END