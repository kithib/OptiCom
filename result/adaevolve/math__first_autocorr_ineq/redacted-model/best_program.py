# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1800
    learning_rate: float = 0.007
    end_lr_factor: float = 3e-6
    num_steps: int = 140000
    warmup_steps: int = 7000


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
        Uses softmax with temperature for smooth gradients, but uses true max
        for best_f tracking to ensure correct value.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-15
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx.
        scaled_conv_f_f = conv_f_f * self.dx

        # Even more accurate shifted log-sum-exp with tighter temperature
        temperature = 0.0005
        max_val = jnp.max(scaled_conv_f_f)
        max_conv = max_val + temperature * jax.scipy.special.logsumexp(
            (scaled_conv_f_f - max_val) / temperature
        )
        
        c1_ratio = max_conv / (integral_f_safe**2)

        # Return the value to be MINIMIZED.
        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient norm clipping for stability."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        # Tighter gradient norm clipping for stability at higher resolution
        grad_norm = jnp.linalg.norm(grads)
        grads = jnp.where(grad_norm > 20.0, grads * (20.0 / (grad_norm + 1e-10)), grads)
        grads = jnp.clip(grads, -35.0, 35.0)
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
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=1.5e-8, b1=0.9, b2=0.9995)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Even sharper initialization: cos^16 is more peaked and closer to known optima
        x = jnp.linspace(-0.25, 0.25, N, endpoint=False)
        f_values = jnp.cos(2.0 * jnp.pi * x) ** 16
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
            # Use strict max for best f to avoid overestimating smooth loss
            f_non_neg = jax.nn.softplus(f_values)
            int_f = jnp.sum(f_non_neg) * self.dx
            padded = jnp.pad(f_non_neg, (0, N))
            conv = jnp.fft.ifft(jnp.fft.fft(padded)**2).real * self.dx
            actual_c1 = jnp.max(conv) / jnp.maximum(int_f, 1e-15)**2
            if actual_c1 < best_loss:
                best_loss = actual_c1
                best_f = f_values
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C1 ≈ {actual_c1:.14f} | Best ≈ {best_loss:.14f}")

        print(f"Final C1 found: {best_loss:.14f}")

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