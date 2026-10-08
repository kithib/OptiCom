# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1664
    learning_rate: float = 0.028
    end_lr_factor: float = 8e-7
    num_steps: int = 220000
    warmup_steps: int = 9000


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
        Uses softplus for smooth non-negativity and LogSumExp for 
        differentiable max approximation with accurate true max for final loss.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-15
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        # Proper symmetric padding for correct convolution range [-1/2, 1/2]
        padded_f = jnp.pad(f_non_negative, (N // 2, N // 2))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        scaled_conv_f_f = conv_f_f * self.dx

        # Use LogSumExp for stable differentiable max approximation
        # Better than softmax-weighted sum for true max approximation
        temperature = 0.0008
        lse_max = temperature * jax.scipy.special.logsumexp(scaled_conv_f_f / temperature)
        c1_ratio = lse_max / (integral_f_safe**2)

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
            optax.clip_by_global_norm(0.35),
            optax.adamw(learning_rate=schedule, weight_decay=6e-6)
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Improved initialization based on known near-optimal shapes
        # Multi-lobed structure with central peak and symmetric side lobes
        x = jnp.linspace(-0.25, 0.25, N)
        support_radius = 0.244
        f_values = jnp.where(
            jnp.abs(x) < support_radius,
            # Shape with mild oscillations that often produce lower C1
            (0.85 + 0.82 * jnp.cos(x * 15.5) ** 2 + 0.48 * jnp.cos(x * 31) ** 2 + 0.22 * jnp.cos(x * 46.5) ** 2 + 0.08 * jnp.cos(x * 62) ** 2) 
            * (1.0 - (jnp.abs(x) / support_radius) ** 6) * 2.65,
            0.0
        )
        f_values += 0.004 * jax.random.normal(key, (N,))

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        best_loss = jnp.inf
        best_f = f_values
        
        # Compute actual true C1 for tracking (using hard max, not differentiable approx)
        def compute_true_c1(f_vals):
            f_pos = jax.nn.softplus(f_vals)
            int_f = jnp.sum(f_pos) * self.dx
            padded = jnp.pad(f_pos, (N // 2, N // 2))
            conv = jnp.fft.ifft(jnp.fft.fft(padded) ** 2).real * self.dx
            return jnp.max(conv) / (jnp.maximum(int_f, 1e-15) ** 2)
        
        compute_true_c1_jit = jax.jit(compute_true_c1)

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            
            # Track best using actual hard max C1
            current_true_c1 = compute_true_c1_jit(f_values)
            if current_true_c1 < best_loss:
                best_loss = current_true_c1
                best_f = f_values
            
            if step % 6000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | Approx C1 ≈ {loss:.10f} | True Best ≈ {best_loss:.14f}")

        print(f"Final approx C1: {loss:.10f}, Best true C1 overall: {best_loss:.14f}")

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