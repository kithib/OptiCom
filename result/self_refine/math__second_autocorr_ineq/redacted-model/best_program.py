import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 128
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
        Computes the objective function using the unitless norm calculation.
        """
        f_non_negative = jax.nn.softplus(f_values)

        # Unscaled discrete autoconvolution with proper padding for linear convolution
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate grid spacing and use Simpson's rule for integrals
        num_conv_points = len(convolution)
        h = 1.0 / num_conv_points

        # L2-norm squared of convolution (Simpson's rule)
        y = convolution
        l2_norm_squared = h * jnp.sum(y[:-1]**2 + y[:-1] * y[1:] + y[1:]**2) / 3.0

        # L1-norm of convolution (Simpson's rule)
        norm_1 = h * jnp.sum(jnp.abs(y[:-1]) + jnp.abs(y[:-1] + y[1:]) + jnp.abs(y[1:])) / 6.0

        # Infinity-norm of the convolution
        norm_inf = jnp.max(jnp.abs(convolution))

        # Calculate C2 ratio with stability epsilon
        denominator = norm_1 * norm_inf + 1e-12
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
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        # Initialize towards step-like function based on AlphaEvolve success
        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        step_template = jnp.where(jnp.arange(N) < N // 4, 1.0, 0.1)
        f_values = step_template + 0.05 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 1000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.12f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.12f}")
        return jax.nn.softplus(f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals