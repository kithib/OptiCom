import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 512
    learning_rate: float = 0.0148
    num_steps: int = 145000
    warmup_steps: int = 5600
    symmetry_reg: float = 0.00085
    smoothing_reg: float = 1.8e-8
    sharpness_penalty: float = 5.5e-7
    weight_decay: float = 2.8e-6


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using precise numerical integration and gradient-based optimization.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def compute_true_c2(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """Compute C2 without any regularization penalties - pure objective value."""
        f_non_negative = jax.nn.softplus(f_values)
        N = self.hypers.num_intervals
        dx = 1.0 / N
        padded_f = jnp.pad(f_non_negative, (N - 1, N - 1))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real * dx
        
        y = convolution
        h = dx
        
        # Simpson's rule for L2 norm squared - enhanced precision
        l2_norm_squared = h / 3 * (y[0] ** 2 + 4 * jnp.sum(y[1:-1:2] ** 2) + 
                                    2 * jnp.sum(y[2:-2:2] ** 2) + y[-1] ** 2)
        
        # Simpson's rule for L1 norm
        norm_1 = h / 3 * (y[0] + 4 * jnp.sum(y[1:-1:2]) + 
                           2 * jnp.sum(y[2:-2:2]) + y[-1])
        
        norm_inf = jnp.max(y)
        denominator = norm_1 * norm_inf + 1e-18
        return l2_norm_squared / denominator

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """Computes the objective function with symmetry, smoothing, and sharpness regularization."""
        f_non_negative = jax.nn.softplus(f_values)

        # Enforce even symmetry constraint (weaker regularization for more flexibility)
        mid = self.hypers.num_intervals // 2
        symmetry_loss = self.hypers.symmetry_reg * jnp.sum(
            (f_non_negative[:mid] - jnp.flip(f_non_negative[mid:])) ** 2
        )

        # Light smoothing regularization
        diff1 = jnp.diff(f_non_negative)
        smoothing_loss = self.hypers.smoothing_reg * jnp.sum(diff1 ** 2)
        
        # Sharpness penalty (penalizes extreme second derivative changes)
        diff2 = jnp.diff(f_non_negative, 2)
        sharpness_loss = self.hypers.sharpness_penalty * jnp.sum(diff2 ** 2)

        # Proper convolution setup with correct domain spacing
        N = self.hypers.num_intervals
        dx = 1.0 / N
        padded_f = jnp.pad(f_non_negative, (N - 1, N - 1))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real * dx

        y = convolution
        h = dx
        
        # Simpson's rule for L2 norm squared
        l2_norm_squared = h / 3 * (y[0] ** 2 + 4 * jnp.sum(y[1:-1:2] ** 2) + 
                                    2 * jnp.sum(y[2:-2:2] ** 2) + y[-1] ** 2)
        
        # Simpson's rule for L1 norm
        norm_1 = h / 3 * (y[0] + 4 * jnp.sum(y[1:-1:2]) + 
                           2 * jnp.sum(y[2:-2:2]) + y[-1])

        # Infinity norm
        norm_inf = jnp.max(y)

        # Calculate C2 ratio with stability epsilon
        denominator = norm_1 * norm_inf + 1e-18
        c2_ratio = l2_norm_squared / denominator

        return -c2_ratio + symmetry_loss + smoothing_loss + sharpness_loss

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
            end_value=self.hypers.learning_rate * 6e-6,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.48),
            optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)
        )

        # Meta-evolved ultra-refined initialization: 16-block asymmetric pattern
        # Based on cross-analysis of top exemplars with enhanced micro-structure and gradient calibration
        x = jnp.linspace(0, 1, self.hypers.num_intervals)
        
        # Highly optimized 16-block pattern with precision asymmetric tuning
        # Each block boundary/height calibrated from gradient flow analysis of top solutions
        block1 = jnp.where((x > 0.088) & (x < 0.252), 1.0000, 0.0)   # Primary peak 1 (full height)
        block2 = jnp.where((x > 0.260) & (x < 0.272), 0.0120, 0.0)   # Micro-bump A1 (fine-tuned)
        block3 = jnp.where((x > 0.276) & (x < 0.288), 0.0240, 0.0)   # Micro-bump A2
        block4 = jnp.where((x > 0.292) & (x < 0.308), 0.0480, 0.0)   # Micro-bump A3
        block5 = jnp.where((x > 0.312) & (x < 0.328), 0.0850, 0.0)   # Transition bump A
        block6 = jnp.where((x > 0.332) & (x < 0.383), 0.1935, 0.0)   # Secondary peak 1 (enhanced)
        block7 = jnp.where((x > 0.395) & (x < 0.420), 0.0420, 0.0)   # Central transition 1a
        block8 = jnp.where((x > 0.424) & (x < 0.446), 0.0720, 0.0)   # Central transition 1b
        block9 = jnp.where((x > 0.554) & (x < 0.576), 0.0720, 0.0)   # Central transition 2b
        block10 = jnp.where((x > 0.580) & (x < 0.605), 0.0420, 0.0)  # Central transition 2a
        block11 = jnp.where((x > 0.617) & (x < 0.668), 0.1935, 0.0)  # Secondary peak 2 (enhanced)
        block12 = jnp.where((x > 0.672) & (x < 0.688), 0.0850, 0.0)  # Transition bump B
        block13 = jnp.where((x > 0.692) & (x < 0.708), 0.0480, 0.0)  # Micro-bump B3
        block14 = jnp.where((x > 0.712) & (x < 0.724), 0.0240, 0.0)  # Micro-bump B2
        block15 = jnp.where((x > 0.728) & (x < 0.740), 0.0120, 0.0)  # Micro-bump B1
        block16 = jnp.where((x > 0.748) & (x < 0.912), 0.9765, 0.0)  # Primary peak 2 (precision height)
        
        f_values = block1 + block2 + block3 + block4 + block5 + block6 + block7 + block8 + block9 + block10 + block11 + block12 + block13 + block14 + block15 + block16
        f_values = f_values + 0.0065 * jax.random.normal(jax.random.PRNGKey(42), f_values.shape)

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)
        compute_c2_jit = jax.jit(self.compute_true_c2)

        best_c2 = 0.0
        best_f = f_values
        prev_best = 0.0
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, _ = train_step_jit(f_values, opt_state)
            current_c2 = compute_c2_jit(f_values)
            
            if current_c2 > best_c2:
                best_c2 = current_c2
                best_f = f_values
            
            if best_c2 > prev_best + 1e-13:
                prev_best = best_c2
            
            if step % 2500 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C2 ≈ {current_c2:.18f} | Best C2 ≈ {best_c2:.18f}")

        final_c2 = float(best_c2)
        print(f"Final C2 lower bound found: {final_c2:.20f}")
        return jax.nn.softplus(best_f), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals