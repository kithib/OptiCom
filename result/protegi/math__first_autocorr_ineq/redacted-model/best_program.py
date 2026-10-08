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
    learning_rate: float = 0.01
    end_lr_factor: float = 1e-4
    num_steps: int = 60000
    warmup_steps: int = 3000


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function, which is the C1 ratio.
        We minimize this ratio to find a tight upper bound.
        Fix 1: Use softplus for smoother gradients through non-negativity (vs relu's dead neurons).
        Fix 2: Restrict max_conv to the valid t ∈ [-1/2, 1/2] range (first 2N points of full convolution).
        """
        # Use softplus for differentiable non-negativity; add small shift so f=0 maps to ~0.
        f_non_negative = jax.nn.softplus(f_values - 5.0)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-9
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        # Pad to length 2N for linear convolution of length-N signal with itself.
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx for integral approximation.
        scaled_conv_f_f = conv_f_f * self.dx

        # Only consider valid t ∈ [-1/2, 1/2] (first 2N points of full convolution).
        max_conv = jnp.max(scaled_conv_f_f[: 2 * N])
        c1_ratio = max_conv / (integral_f_safe**2)

        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process.
        Fix 3: Use a multi-start warm-start strategy with cosine schedule, and track best-so-far solution."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(learning_rate=schedule),
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Initialize as a smooth cosine-bump centered on the interval for better starting point.
        grid = jnp.linspace(-0.25, 0.25, N, endpoint=False) + 0.25 / N
        center = 0.0
        width = 0.2
        bump = jnp.clip(1.0 - ((grid - center) / width) ** 2, 0.0) ** 2
        bump += 0.01 * jax.random.uniform(key, (N,))
        # Inverse of softplus so initial f_values map to ~bump after softplus transform.
        f_values = jnp.log(jnp.expm1(jnp.clip(bump, 1e-6))) + 5.0

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        best_loss = jnp.inf
        best_f = f_values
        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Track best-so-far to avoid regression from late steps.
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C1 ≈ {loss:.10f} | Best ≈ {best_loss:.10f}")

        print(f"Final C1 found: {best_loss:.10f}")

        # Convert back to non-negative f values for output.
        return jax.nn.softplus(best_f - 5.0), best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals


# EVOLVE-BLOCK-END