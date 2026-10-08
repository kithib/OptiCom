import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2200
    learning_rate: float = 0.0033
    end_lr_factor: float = 4e-6
    num_steps: int = 280000
    warmup_steps: int = 14000
    aux_loss_weight: float = 0.048
    weight_decay: float = 5e-6


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
        Adds auxiliary losses to encourage smoothness and better solutions.
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

        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        # Auxiliary loss 1: Smoothness penalty on f (second derivative-based)
        first_diff = f_non_negative[1:] - f_non_negative[:-1]
        second_diff = first_diff[1:] - first_diff[:-1]
        smoothness = jnp.sum(jnp.square(second_diff)) * self.hypers.aux_loss_weight * 0.0012
        
        # Auxiliary loss 2: L2 regularization
        l2_reg = jnp.sum(jnp.square(f_non_negative)) * self.hypers.weight_decay
        
        # Auxiliary loss 3: Penalty on autoconvolution variance to flatten the peak
        conv_variance = jnp.sum(jnp.square(scaled_conv_f_f - max_conv)) / (2 * N)
        variance_penalty = conv_variance * self.hypers.aux_loss_weight * 0.0006

        return c1_ratio + smoothness + l2_reg + variance_penalty

    def _pure_c1(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """Compute pure C1 without auxiliary losses for accurate tracking."""
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-12
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
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.92),
            optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Improved initialization: symmetric smooth peaks pattern 
        # Literature suggests optimal functions have symmetric peaks at 1/4 and 3/4 positions
        x = jnp.linspace(-1/4, 1/4, N)
        # Create a function with shape similar to known near-optimal solutions
        # Using multi-lobe cosine^n pattern that has shown good results in literature
        f_values = (jnp.cos(x * jnp.pi * 4.0) ** 18) * 0.58 + (jnp.cos(x * jnp.pi * 1.35) ** 9) * 0.42
        f_values += 0.0025 * jax.random.normal(key, (N,))
        # Force symmetry in initialization
        f_values = (f_values + jnp.flip(f_values)) / 2.0

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)
        pure_c1_jit = jax.jit(self._pure_c1)

        loss = jnp.inf
        best_c1 = jnp.inf
        best_f = f_values
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            
            # Track best solution
            current_c1 = pure_c1_jit(f_values)
            if current_c1 < best_c1:
                best_c1 = current_c1
                best_f = f_values
            
            if step % 14000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C1 ≈ {current_c1:.20f} | Best: {best_c1:.20f}")

        print(f"Final C1 found: {best_c1:.20f}")

        return jax.nn.softplus(best_f), best_c1


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals