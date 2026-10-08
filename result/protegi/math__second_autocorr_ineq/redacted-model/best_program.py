# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 384
    learning_rate: float = 0.0175
    num_steps: int = 160000
    warmup_steps: int = 6000


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

        # Unscaled discrete autoconvolution using proper symmetric domain
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate L2-norm squared using Simpson's rule
        num_conv_points = len(convolution)
        h = 1.0 / N
        y = convolution
        # Simpson's rule weights: (h/3) * (y0 + 4y1 + 2y2 + 4y3 + ... + 4y_{n-1} + yn)
        alt_weights = 4.0 * jnp.ones(num_conv_points)
        alt_weights = alt_weights.at[::2].set(2.0)
        alt_weights = alt_weights.at[0].set(1.0)
        alt_weights = alt_weights.at[-1].set(1.0)
        l2_norm_squared = (h / 3.0) * jnp.sum(alt_weights * y**2)

        # Calculate L1-norm using Simpson's rule (equals (integral f)^2)
        norm_1 = (h / 3.0) * jnp.sum(alt_weights * jnp.abs(convolution))

        # Calculate infinity-norm of the convolution
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio with tight epsilon for numerical stability
        denominator = norm_1 * norm_inf + 1e-18
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
            end_value=self.hypers.learning_rate * 5e-9,
        )
        # AdamW with gradient clipping and precisely tuned regularization
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.15),
            optax.adamw(learning_rate=schedule, weight_decay=2.5e-6)
        )

        # Initialize with ultra-refined 25-level step approximation of optimum shape
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(0, 1, self.hypers.num_intervals)
        # Higher granularity step function with finely tuned heights and breakpoints
        f_values = jnp.where(x < 0.020, 1.61,
                       jnp.where(x < 0.060, 1.55,
                       jnp.where(x < 0.100, 1.49,
                       jnp.where(x < 0.140, 1.43,
                       jnp.where(x < 0.180, 1.37,
                       jnp.where(x < 0.220, 1.31,
                       jnp.where(x < 0.260, 1.25,
                       jnp.where(x < 0.300, 1.19,
                       jnp.where(x < 0.340, 1.13,
                       jnp.where(x < 0.380, 1.07,
                       jnp.where(x < 0.420, 1.01,
                       jnp.where(x < 0.460, 0.95,
                       jnp.where(x < 0.500, 0.89,
                       jnp.where(x < 0.540, 0.83,
                       jnp.where(x < 0.580, 0.77,
                       jnp.where(x < 0.620, 0.71,
                       jnp.where(x < 0.660, 0.66,
                       jnp.where(x < 0.700, 0.61,
                       jnp.where(x < 0.740, 0.56,
                       jnp.where(x < 0.780, 0.51,
                       jnp.where(x < 0.820, 0.47,
                       jnp.where(x < 0.860, 0.43,
                       jnp.where(x < 0.900, 0.395,
                       jnp.where(x < 0.950, 0.36, 0.33))))))))))))))))))))))))
        # Balanced controlled noise for fine exploration while preserving general shape
        f_values = f_values + jax.random.uniform(key, (self.hypers.num_intervals,), minval=-0.007, maxval=0.007)

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
            # Track best solution with very high precision improvement detection
            current_c2 = -loss
            if current_c2 > best_c2 + 1e-16:
                best_c2 = current_c2
                best_f = f_values
                
            if step % 8000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C2 ≈ {current_c2:.14f} | Best ≈ {best_c2:.14f}")

        # Use best found solution throughout entire trajectory
        final_c2 = -self._objective_fn(best_f)
        print(f"Final C2 lower bound found: {final_c2:.16f}")
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