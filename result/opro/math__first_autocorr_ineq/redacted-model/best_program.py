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
    learning_rate: float = 0.01
    end_lr_factor: float = 1e-5
    num_steps: int = 120000
    warmup_steps: int = 4000
    ema_decay: float = 0.9995


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _eval_c1(self, f_values: jnp.ndarray) -> tuple:
        """Actual C1 evaluation for final tracking (no relaxation)."""
        f_non_negative = jnp.where(f_values > 30, f_values, jnp.log1p(jnp.exp(f_values)))
        integral_f = jnp.sum(f_non_negative) * self.dx
        eps = 1e-12
        integral_f_safe = jnp.maximum(integral_f, eps)
        
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N // 2, N // 2))
        
        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real
        scaled_conv_f_f = conv_f_f * self.dx
        
        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)
        return c1_ratio, f_non_negative

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function, which is the C1 ratio.
        We minimize this ratio to find a tight upper bound.
        Uses numerically stable softplus, symmetric padding, and soft maximum relaxation.
        """
        f_non_negative = jnp.where(f_values > 30, f_values, jnp.log1p(jnp.exp(f_values)))
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-12
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N // 2, N // 2))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        scaled_conv_f_f = conv_f_f * self.dx

        temperature = 0.0005
        max_conv = jnp.max(scaled_conv_f_f) + temperature * jnp.log(jnp.sum(jnp.exp((scaled_conv_f_f - jnp.max(scaled_conv_f_f)) / temperature)))
        c1_ratio = max_conv / (integral_f_safe**2)

        return c1_ratio

    def train_step(self, state: tuple, opt_state: optax.OptState) -> tuple:
        """Performs a single training step with gradient clipping, weight decay, and EMA."""
        f_values, ema_f = state
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        grads = jnp.clip(grads, -1.0, 1.0)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        ema_f = self.hypers.ema_decay * ema_f + (1.0 - self.hypers.ema_decay) * f_values

        return (f_values, ema_f), opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adamw(learning_rate=schedule, weight_decay=1e-5)
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        
        x = jnp.linspace(-0.25, 0.25, N)
        # Better initial guess: peaked Gaussian with polynomial envelope
        f_values = jnp.exp(-(x ** 2) * 60) * (1.0 - 4.0 * jnp.abs(x)) ** 3.5
        # Add step pattern component from prior successes
        N = self.hypers.num_intervals
        step_pattern = jnp.zeros((N,))
        first_q = N // 4
        second_q = N // 2
        third_q = 3 * N // 4
        step_pattern = step_pattern.at[first_q - N//12:first_q + N//12].set(1.2)
        step_pattern = step_pattern.at[second_q - N//12:second_q + N//12].set(1.5)
        step_pattern = step_pattern.at[third_q - N//6:third_q].set(1.0)
        # Blend initial guesses
        f_values = 0.7 * f_values + 0.3 * step_pattern
        f_values += 0.02 * jax.random.uniform(key, (N,))
        # Transform to numerically stable log-space for optimization
        f_values = jnp.where(f_values > 1e-9, jnp.log(jnp.expm1(f_values)), f_values)
        ema_f = f_values.copy()

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)
        eval_jit = jax.jit(self._eval_c1)

        state = (f_values, ema_f)
        loss = jnp.inf
        best_loss = jnp.inf
        _, best_f = eval_jit(f_values)
        
        for step in range(self.hypers.num_steps):
            state, opt_state, loss = train_step_jit(state, opt_state)
            # Track best using actual C1, also check EMA solution
            current_c1, current_f = eval_jit(state[0])
            ema_c1, ema_f = eval_jit(state[1])
            if current_c1 < best_loss:
                best_loss = current_c1
                best_f = current_f
            if ema_c1 < best_loss:
                best_loss = ema_c1
                best_f = ema_f
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | Train C1 ≈ {loss:.10f} | Best actual C1 ≈ {best_loss:.12f}")

        print(f"Best actual C1 found: {best_loss:.12f}")

        return best_f, best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals


# EVOLVE-BLOCK-END