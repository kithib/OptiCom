import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1500
    learning_rate: float = 0.016
    num_steps: int = 100000
    warmup_steps: int = 7000


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
        Uses a soft maximum + sharpened penalty combination for improved gradients.
        """
        integral_f = jnp.sum(f_values) * self.dx
        eps = 1e-14
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_values, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

        scaled_conv_f_f = conv_f_f * self.dx

        conv_abs = jnp.abs(scaled_conv_f_f)
        
        # LogSumExp with very high alpha for ultra-sharp softmax approximating true max
        alpha = 500.0
        soft_max = (1.0 / alpha) * jax.scipy.special.logsumexp(alpha * conv_abs)
        
        # Top values for extra gradient signal near the maximum with finer granularity
        top_k_mean = jnp.mean(jax.lax.top_k(conv_abs, 4)[0])
        second_top_mean = jnp.mean(jax.lax.top_k(conv_abs, 12)[0])
        
        # Combined surrogate: heavy weight on soft_max for proximity to true max
        combined_obj = 0.88 * soft_max + 0.09 * top_k_mean + 0.03 * second_top_mean
        
        c3_ratio = combined_obj / integral_f_sq_safe

        return c3_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient clipping."""
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
            end_value=self.hypers.learning_rate * 1e-6,
        )
        # AdamW with gradient clipping and very low regularization for fine-grained adaptation
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.7),
            optax.adamw(learning_rate=schedule, weight_decay=1e-6)
        )

        key = jax.random.PRNGKey(42)
        f_values = jax.random.normal(key, (self.hypers.num_intervals,)) * 0.15

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_loss = jnp.inf
        best_f = f_values
        
        # Compute actual max for tracking
        N = self.hypers.num_intervals
        dx = self.dx
        
        @jax.jit
        def compute_true_c3(f):
            integral_f = jnp.sum(f) * dx
            integral_f_sq_safe = jnp.maximum(integral_f**2, 1e-14)
            padded_f = jnp.pad(f, (0, N))
            fft_f = jnp.fft.fft(padded_f)
            conv_f_f = jnp.fft.ifft(fft_f * fft_f).real
            scaled_conv_f_f = conv_f_f * dx
            max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))
            return max_abs_conv / integral_f_sq_safe
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            # Track using true C3 metric
            true_c3 = compute_true_c3(f_values)
            if true_c3 < best_loss:
                best_loss = true_c3
                best_f = f_values
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | Surrogate ≈ {loss:.16f} | True C3 ≈ {true_c3:.16f} | Best ≈ {best_loss:.16f}")

        final_c3 = best_loss
        print(f"Final C3 upper bound found: {final_c3:.16f}")
        return best_f, final_c3


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()

    loss_val = final_c3_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c3_val), float(loss_val), hypers.num_intervals