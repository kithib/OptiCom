import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1450
    learning_rate: float = 0.026
    num_steps: int = 320000
    warmup_steps: int = 16000
    dx: float = 0.00355


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant.
    Uses symmetry and correct discretization scaling.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the C2 objective function with correct scaling.
        Uses even function symmetry: f(-x) = f(x) to simplify.
        """
        f_non_negative = jax.nn.softplus(f_values)

        dx = self.hypers.dx

        # Create full even function
        # f_non_negative represents values for x >= 0
        full_f = jnp.concatenate([jnp.flip(f_non_negative[1:]), f_non_negative])

        # Compute integral of f
        int_f = jnp.sum(full_f) * dx

        # Compute autoconvolution via FFT with proper padding
        M = len(full_f)
        padded_len = 2 * M
        padded_f = jnp.pad(full_f, (0, padded_len - M))
        fft_f = jnp.fft.fft(padded_f)
        conv_complex = jnp.fft.ifft(fft_f * fft_f)
        convolution = conv_complex.real * dx

        # ||f ★ f||_2^2
        l2_norm_squared = jnp.sum(convolution**2) * dx

        # ||f ★ f||_1 = (∫f)^2
        norm_1 = int_f**2

        # ||f ★ f||_inf
        norm_inf = jnp.max(convolution)

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        c2_ratio = l2_norm_squared / denominator

        return -c2_ratio

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
            end_value=self.hypers.learning_rate * 7e-6,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.82),
            optax.adamw(learning_rate=schedule, weight_decay=3.5e-7)
        )

        key = jax.random.PRNGKey(42)
        # Initialize with highly refined step-like profile based on AlphaEvolve optimum
        x = jnp.arange(self.hypers.num_intervals) * self.hypers.dx
        # Ultra-precise 11-plateau initial guess matching known optimum with fine-grained tail
        f_initial = jnp.where(x < 0.485, 1.0319, 
                       jnp.where(x < 1.085, 0.6915,
                       jnp.where(x < 1.725, 0.4317,
                       jnp.where(x < 2.485, 0.2315,
                       jnp.where(x < 3.25, 0.1117,
                       jnp.where(x < 4.0, 0.0412,
                       jnp.where(x < 4.8, 0.0172,
                       jnp.where(x < 5.6, 0.0072,
                       jnp.where(x < 6.3, 0.0031,
                       jnp.where(x < 7.0, 0.00125,
                       jnp.where(x < 7.8, 0.00055, 0.00022)))))))))))
        f_values = f_initial + 0.0028 * jax.random.uniform(key, (self.hypers.num_intervals,))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        best_c2 = 0.0
        best_f = f_values
        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            current_c2 = -loss
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f = f_values
            if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {current_c2:.12f} | Best ≈ {best_c2:.12f}")

        final_c2 = -self._objective_fn(best_f)
        print(f"Final C2 lower bound found: {final_c2:.14f}")
        return jax.nn.softplus(best_f), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals