import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""
    num_intervals: int = 900
    learning_rate: float = 0.012
    end_lr_factor: float = 5e-5
    num_steps: int = 75000
    warmup_steps: int = 4000
    num_trials: int = 4


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
        """
        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-9
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
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process with multiple restarts."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        N = self.hypers.num_intervals
        key = jax.random.PRNGKey(42)
        x = jnp.linspace(-0.25, 0.25, N)
        f_values = jnp.exp(-x**2 / (2 * 0.12**2))
        f_values += 0.04 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 7500 == 0 or step == self.hypers.num_steps - 1:
                print(f"Initial Step {step:5d} | C1 ≈ {loss:.8f}")

        print(f"Initial run C1: {loss:.8f}")

        best_loss = loss
        best_f = jax.nn.relu(f_values)

        for trial in range(self.hypers.num_trials):
            if trial == 0:
                f_values = jnp.zeros((N,))
                start_idx, end_idx = N // 5, 4 * N // 5
                f_values = f_values.at[start_idx:end_idx].set(1.0)
                f_values += 0.03 * jax.random.uniform(jax.random.PRNGKey(trial * 100), (N,))
            elif trial == 1:
                x = jnp.linspace(-0.25, 0.25, N)
                f_values = jnp.exp(-x**2 / (2 * 0.08**2))
                f_values += 0.02 * jax.random.uniform(jax.random.PRNGKey(trial * 100), (N,))
            elif trial == 2:
                f_values = jnp.zeros((N,))
                mid1 = N // 3
                mid2 = 2 * N // 3
                width = N // 10
                f_values = f_values.at[mid1 - width : mid1 + width].set(1.0)
                f_values = f_values.at[mid2 - width : mid2 + width].set(1.0)
                f_values += 0.04 * jax.random.uniform(jax.random.PRNGKey(trial * 100), (N,))
            else:
                f_values = jnp.zeros((N,))
                f_values = f_values.at[: N // 2].set(jnp.exp(-jnp.linspace(-0.25, 0.0, N // 2)**2 / (2 * 0.15**2)))
                f_values = f_values.at[N // 2 :].set(jnp.exp(-jnp.linspace(0.0, 0.25, N // 2)**2 / (2 * 0.15**2)))
                f_values += 0.03 * jax.random.uniform(jax.random.PRNGKey(trial * 100), (N,))

            opt_state = self.optimizer.init(f_values)
            loss = jnp.inf

            for step in range(self.hypers.num_steps):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                if step % 7500 == 0 or step == self.hypers.num_steps - 1:
                    print(f"Trial {trial} | Step {step:5d} | C1 ≈ {loss:.8f}")

            print(f"Trial {trial} Final C1: {loss:.8f}")

            if loss < best_loss:
                best_loss = loss
                best_f = jax.nn.relu(f_values)

        final_f = jax.nn.relu(best_f)
        int_f = float(jnp.sum(final_f) * self.dx)
        final_loss_val = float(best_loss)
        if int_f <= 1e-6 or not np.isfinite(final_loss_val) or final_loss_val <= 1e-6 or final_loss_val > 10.0:
            print("WARNING: Degenerate solution detected; returning safe default.")
            safe_f = jnp.zeros((N,)).at[N // 4 : 3 * N // 4].set(1.0)
            safe_int = float(jnp.sum(safe_f) * self.dx)
            padded_safe = jnp.pad(safe_f, (0, N))
            fft_safe = jnp.fft.fft(padded_safe)
            conv_safe = jnp.fft.ifft(fft_safe * fft_safe).real
            safe_max_conv = float(jnp.max(conv_safe * self.dx))
            safe_loss_val = safe_max_conv / (safe_int ** 2)
            safe_loss = jnp.array(safe_loss_val)
            print(f"Safe fallback C1: {safe_loss_val:.8f}")
            return safe_f, safe_loss

        print(f"Best C1 found across trials: {best_loss:.8f}")

        return final_f, best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals