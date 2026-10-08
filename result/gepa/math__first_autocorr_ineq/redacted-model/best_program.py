# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1200
    learning_rate: float = 0.0078
    end_lr_factor: float = 1.2e-5
    num_steps: int = 100000
    warmup_steps: int = 4500
    clip_norm: float = 12.5


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
        Uses refined combined smoothness penalty with balanced TV and second derivative terms.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-14
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx for integral approximation
        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        # Refined penalty: balanced total variation + second derivative for optimal smoothness
        tv = jnp.sum(jnp.abs(f_non_negative[1:] - f_non_negative[:-1]))
        sd = jnp.sum(jnp.abs(f_non_negative[2:] - 2*f_non_negative[1:-1] + f_non_negative[:-2]))
        penalty = (4.2e-6 * tv + 1.6e-6 * sd) / (jnp.sum(f_non_negative) + eps)

        return c1_ratio + penalty

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient clipping for stability."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process with best-solution tracking."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        # Gradient clipping with AdamW - optimized betas and weight decay
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.clip_norm),
            optax.adamw(learning_rate=schedule, b1=0.875, b2=0.9965, weight_decay=3.8e-5)
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Enhanced initial guess: optimized multi-component cosine structure with narrow support
        x = jnp.linspace(-1/4, 1/4, N)
        # Multi-component initial function with improved narrow support and shape
        cos1 = jnp.where(jnp.abs(x) <= 0.235, jnp.cos(1.85 * jnp.pi * x)**7, 0.0)
        cos2 = jnp.where(jnp.abs(x) <= 0.235, jnp.cos(2.85 * jnp.pi * x)**7, 0.0)
        cos3 = jnp.where(jnp.abs(x) <= 0.22, jnp.cos(3.7 * jnp.pi * x)**9, 0.0)
        f_base = 0.55 * cos1 + 0.30 * cos2 + 0.15 * cos3
        f_base = jnp.maximum(f_base, 0.007)
        # Refined soft mapping for parameter space with improved initialization
        f_values = jnp.log(jnp.cosh(8.5 * f_base))
        # Reduced noise for more stable initialization
        f_values += 0.006 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf  # Initialize loss
        best_loss = jnp.inf
        best_f = f_values
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Track best solution encountered
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            if step % 4000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C1 ≈ {loss:.16f} | Best ≈ {best_loss:.16f}")

        print(f"Final C1 found: {best_loss:.16f}")

        return jax.nn.softplus(best_f), best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals


# EVOLVE-BLOCK-END