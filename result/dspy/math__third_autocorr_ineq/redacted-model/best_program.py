# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 800
    learning_rate: float = 0.008
    num_steps: int = 50000
    warmup_steps: int = 3000


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
        padded_f = jnp.pad(f_values, (0, N))

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
            end_value=self.hypers.learning_rate * 5e-5,
        )
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-5)

        # Better initialization combining random with known structure hints
        key = jax.random.PRNGKey(42)
        n = self.hypers.num_intervals
        x = jnp.linspace(-0.25, 0.25, n)
        random_part = 0.1 * jax.random.normal(key, (n,))
        # Initialize with structured components known to help
        cos_part = jnp.cos(4 * jnp.pi * x)
        sin_part = jnp.sin(6 * jnp.pi * x)
        exp_part = jnp.exp(-(x**2) * 100)
        f_values = random_part + 0.5 * cos_part + 0.3 * sin_part + exp_part

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_f = f_values
        best_loss = jnp.inf
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            
            # Track best solution found
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C3 ≈ {loss:.10f} | Best: {best_loss:.10f}")

        final_c3 = best_loss
        print(f"Final C3 upper bound found: {final_c3:.12f}")
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