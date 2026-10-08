import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2800
    learning_rate: float = 0.022
    end_lr_factor: float = 5e-7
    num_steps: int = 350000
    warmup_steps: int = 12000


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> tuple:
        """
        Computes the objective function, which is the C1 ratio.
        We minimize this ratio to find a tight upper bound.
        Uses log-transformed parameterization for strict non-negativity.
        """
        f_non_negative = jnp.exp(f_values)
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

        # L2 regularization + mild TV regularization for smoothness - fine-tuned coefficients
        l2_reg = 1.5e-5 * jnp.sum(f_non_negative ** 2)
        tv_reg = 4.5e-6 * jnp.sum(jnp.abs(jnp.diff(f_non_negative)))
        
        return c1_ratio + l2_reg + tv_reg, c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        (loss, c1), grads = jax.value_and_grad(self._objective_fn, has_aux=True)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss, c1

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
            optax.clip_by_global_norm(0.45),  # Tighter gradient clipping for stability
            optax.adamw(learning_rate=schedule, weight_decay=4e-6)
        )

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Enhanced initialization: multi-lobe structure with further refined envelope and bumps
        x = jnp.linspace(-0.25, 0.25, N)
        envelope = 1.0 - (x / 0.25) ** 16  # Sharper envelope to taper at domain edges
        central_bump = 1.55 * jnp.exp(-(x ** 2) / 0.0052)  # Narrower/taller central peak
        inner_side_bumps = 0.9 * (jnp.exp(-((x - 0.108) ** 2) / 0.0025) +
                                   jnp.exp(-((x + 0.108) ** 2) / 0.0025))
        middle_side_bumps = 0.62 * (jnp.exp(-((x - 0.168) ** 2) / 0.002) +
                                     jnp.exp(-((x + 0.168) ** 2) / 0.002))
        outer_side_bumps = 0.32 * (jnp.exp(-((x - 0.228) ** 2) / 0.001) +
                                    jnp.exp(-((x + 0.228) ** 2) / 0.001))
        f_values = envelope * (central_bump + inner_side_bumps + middle_side_bumps + outer_side_bumps)
        f_values = jnp.log(f_values + 1e-8)  # Transform for exp parameterization
        f_values += 0.002 * jax.random.uniform(key, (N,))  # Minimal noise for clean start

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        best_c1 = jnp.inf
        best_f = f_values
        plateau_count = 0
        prev_best = jnp.inf
        
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss, c1 = train_step_jit(f_values, opt_state)
            
            if c1 < best_c1:
                best_c1 = c1
                best_f = f_values
            
            # Plateau detection for best-of-many tracking
            if step % 1000 == 0:
                if best_c1 < prev_best:
                    prev_best = best_c1
                    plateau_count = 0
                else:
                    plateau_count += 1
            
            if step % 15000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C1 ≈ {c1:.14f} | Best C1 ≈ {best_c1:.14f}")

        print(f"Final C1 found: {best_c1:.16f}")

        return jnp.exp(best_f), best_c1


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals