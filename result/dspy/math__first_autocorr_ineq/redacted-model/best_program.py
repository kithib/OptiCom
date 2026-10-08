import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2600
    learning_rate: float = 0.014
    end_lr_factor: float = 2e-6
    num_steps: int = 220000
    warmup_steps: int = 15000


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
        Uses softplus for smoother gradients and higher precision.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-18
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient clipping."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        grads = jnp.clip(grads, -20.0, 20.0)
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
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=3e-7)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        
        # Advanced initialization: combination of optimal forms with enhanced windowing
        x = jnp.linspace(-0.25, 0.25, N)
        # Triangular base (known near-optimal core)
        triangular = 1.0 - 4.0 * jnp.abs(x)
        triangular = triangular * (triangular > 0)
        # Cos^6 envelope for even better edge behavior
        cos6_env = jnp.cos(2.0 * jnp.pi * x) ** 6
        # Gaussian component for smooth tapering
        gaussian = jnp.exp(-16.0 * x**2)
        # Exponential decay for asymmetric tail handling
        exp_decay = jnp.exp(-12.0 * jnp.abs(x))
        # Cos^2 for additional shape flexibility
        cos2_env = jnp.cos(2.0 * jnp.pi * x) ** 2
        # Weighted combination capturing known optimal characteristics
        f_base = 0.38 * triangular + 0.28 * cos6_env * triangular + 0.14 * gaussian * triangular + 0.12 * exp_decay + 0.08 * cos2_env * triangular
        # Blackman-Harris window for superior spectral properties
        blackman_harris = 0.35875 + 0.48829 * jnp.cos(4.0 * jnp.pi * x) + 0.14128 * jnp.cos(8.0 * jnp.pi * x) + 0.01168 * jnp.cos(12.0 * jnp.pi * x)
        # Nuttall window for alternative ultra-smooth side-lobe suppression
        nuttall = 0.355768 + 0.487396 * jnp.cos(4.0 * jnp.pi * x) + 0.144232 * jnp.cos(8.0 * jnp.pi * x) + 0.012604 * jnp.cos(12.0 * jnp.pi * x)
        # Composite window for optimal tradeoff
        composite_window = 0.5 * blackman_harris + 0.5 * nuttall
        f_values = f_base * composite_window
        f_values += 0.004 * jax.random.uniform(key, (N,))

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
            
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
                
            if step % 11000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C1 ≈ {loss:.18f} | Best: {best_loss:.18f}")

        print(f"Final C1 found: {best_loss:.18f}")

        return jax.nn.softplus(best_f), best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals