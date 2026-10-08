import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1200
    learning_rate: float = 0.01
    end_lr_factor: float = 1e-5
    num_steps: int = 80000
    warmup_steps: int = 4000
    restart_times: int = 3


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
        Uses softplus for smoother non-negativity and better gradient flow.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-12
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx.
        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        # Return the value to be MINIMIZED.
        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss

    def run_single_optimization(self, key: jax.random.PRNGKey, init_f: jnp.ndarray = None) -> tuple:
        """Runs a single optimization run with configurable initialization."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        # Use AdamW for better weight regularization
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=1e-6)

        N = self.hypers.num_intervals
        if init_f is None:
            # Multi-modal initialization covering different promising regions
            f_values = jnp.zeros((N,))
            # Central region initialization
            start_idx, end_idx = N // 4, 3 * N // 4
            f_values = f_values.at[start_idx:end_idx].set(1.0)
            f_values += 0.03 * jax.random.uniform(key, (N,))
        else:
            f_values = init_f + 0.01 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)

        train_step_jit = jax.jit(self.train_step)

        best_loss = jnp.inf
        best_f = f_values

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)

            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            if step % 4000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C1 ≈ {loss:.10f} | Best ≈ {best_loss:.10f}")

        # Return softplus version for non-negativity
        return jax.nn.softplus(best_f), best_loss

    def run_optimization(self):
        """Sets up and runs the full optimization with restarts."""
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, Restarts: {self.hypers.restart_times}"
        )

        key = jax.random.PRNGKey(42)
        
        best_overall_loss = jnp.inf
        best_overall_f = None
        current_init = None
        
        for restart in range(self.hypers.restart_times + 1):
            print(f"\n--- Run {restart + 1}/{self.hypers.restart_times + 1} ---")
            key, subkey = jax.random.split(key)
            
            optimized_f, loss = self.run_single_optimization(subkey, current_init)
            
            if loss < best_overall_loss:
                best_overall_loss = loss
                best_overall_f = optimized_f
                print(f"New best C1: {best_overall_loss:.10f}")
            
            # Use best solution for next initialization
            current_init = jnp.array(np.arcsinh(np.array(best_overall_f)))  # Inverse of softplus approx

        print(f"\nFinal best C1 found: {best_overall_loss:.12f}")

        return best_overall_f, best_overall_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)
    # Ensure strict non-negativity
    f_values_np = np.maximum(f_values_np, 0.0)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals