import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1500
    learning_rate: float = 0.0082
    end_lr_factor: float = 1.2e-5
    num_steps: int = 115000
    warmup_steps: int = 5800
    use_softplus: bool = True
    smoothness_reg: float = 1.8e-5
    second_diff_reg: float = 0.9e-5
    weight_decay: float = 1.4e-5


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
        # Use softplus for smoother non-negativity constraint (improves gradients)
        if self.hypers.use_softplus:
            f_non_negative = jax.nn.softplus(f_values - 0.5)  # Offset to get approximate f=0 at f_values=0
        else:
            f_non_negative = jax.nn.relu(f_values)
        
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-12
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        # Use symmetric padding for better convolution behavior at boundaries
        padded_f = jnp.pad(f_non_negative, (N // 2, N // 2))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx for integral approximation
        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        # Add small smoothness regularization to encourage nicer functions
        if self.hypers.smoothness_reg > 0:
            diffs = f_non_negative[1:] - f_non_negative[:-1]
            smoothness_penalty = self.hypers.smoothness_reg * jnp.sum(diffs**2)
            c1_ratio = c1_ratio + smoothness_penalty
        
        # Add second derivative regularization for even smoother functions
        if self.hypers.second_diff_reg > 0 and N > 2:
            second_diffs = f_non_negative[2:] - 2 * f_non_negative[1:-1] + f_non_negative[:-2]
            second_diff_penalty = self.hypers.second_diff_reg * jnp.sum(second_diffs**2)
            c1_ratio = c1_ratio + second_diff_penalty

        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        # Clip gradients for stability
        grads = jnp.clip(grads, -0.7, 0.7)
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
        # Use AdamW with weight decay for better generalization
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        
        # Improved initialization: blend of triangle + cos^8 + cos^4 + sin^6 (literature-inspired)
        x = jnp.linspace(-0.25, 0.25, N)
        triangle = 1.0 - 4.0 * jnp.abs(x)
        cos8 = jnp.cos(2 * jnp.pi * x) ** 8
        cos4 = jnp.cos(2 * jnp.pi * x) ** 4
        sin6 = (jnp.sin(4 * jnp.pi * x) ** 2) ** 3  # sin^6 shape, non-negative
        f_values = 0.52 * triangle + 0.28 * cos8 + 0.14 * cos4 + 0.06 * sin6
        # Add very small noise for exploration
        f_values += 0.011 * jax.random.uniform(key, (N,), minval=-0.0055, maxval=0.0055)
        # Transform to softplus space
        if self.hypers.use_softplus:
            # f = softplus(y - 0.5), so y = inverse_softplus(f) + 0.5
            safe_f = jnp.maximum(f_values, 1e-6)
            f_values = jnp.log(jnp.exp(safe_f) - 1.0) + 0.5

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
            
            # Track best solution
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            
            if step % 3600 == 0 or step == self.hypers.num_steps - 1:
                # Compute actual C1 without regularization for printing
                if self.hypers.use_softplus:
                    f_nonneg = jax.nn.softplus(f_values - 0.5)
                else:
                    f_nonneg = jax.nn.relu(f_values)
                integral = jnp.sum(f_nonneg) * self.dx
                padded = jnp.pad(f_nonneg, (N // 2, N // 2))
                conv = jnp.fft.ifft(jnp.fft.fft(padded) ** 2).real * self.dx
                actual_c1 = jnp.max(conv) / (jnp.maximum(integral, 1e-12) ** 2)
                print(f"Step {step:5d} | C1 ≈ {actual_c1:.14f} | Best (with reg): {best_loss:.14f}")

        # Final computation for best_f
        if self.hypers.use_softplus:
            best_f_nonneg = jax.nn.softplus(best_f - 0.5)
        else:
            best_f_nonneg = jax.nn.relu(best_f)
        
        integral_best = jnp.sum(best_f_nonneg) * self.dx
        padded_best = jnp.pad(best_f_nonneg, (N // 2, N // 2))
        conv_best = jnp.fft.ifft(jnp.fft.fft(padded_best) ** 2).real * self.dx
        final_c1 = jnp.max(conv_best) / (jnp.maximum(integral_best, 1e-12) ** 2)
        
        print(f"Final C1 found: {float(final_c1):.16f}")

        return best_f_nonneg, final_c1


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals