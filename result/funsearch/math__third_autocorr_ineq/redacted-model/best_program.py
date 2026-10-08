# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1000
    learning_rate: float = 0.012
    num_steps: int = 50000
    warmup_steps: int = 5000
    num_restarts: int = 5
    restart_tolerance: float = 1e-5
    weight_decay: float = 2e-6
    init_scale: float = 0.25
    clip_norm: float = 1.2


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
        """
        # The squared integral of f.
        integral_f = jnp.sum(f_values) * self.dx
        eps = 1e-9
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)

        # The max of the absolute value of the autoconvolution.
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_values, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real

        # Scale the unscaled convolution sum by dx to approximate the integral.
        scaled_conv_f_f = conv_f_f * self.dx

        # Take the maximum of the absolute value.
        max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))

        c3_ratio = max_abs_conv / integral_f_sq_safe

        # We want to MINIMIZE the ratio.
        return c3_ratio

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
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.clip_norm),
            optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)
        )

        best_overall_c3 = jnp.inf
        best_overall_f = None
        seeds = [42, 123, 456, 789, 101112, 131415, 161718]
        
        for seed_idx, seed in enumerate(seeds[:self.hypers.num_restarts]):
            print(f"\n=== Optimization run {seed_idx + 1}/{self.hypers.num_restarts} with seed {seed} ===")
            key = jax.random.PRNGKey(seed)
            f_values = jax.random.normal(key, (self.hypers.num_intervals,)) * self.hypers.init_scale

            opt_state = self.optimizer.init(f_values)
            train_step_jit = jax.jit(self.train_step)

            loss = jnp.inf
            best_loss = jnp.inf
            best_f = f_values
            plateau_counter = 0

            for step in range(self.hypers.num_steps):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                
                if loss < best_loss - self.hypers.restart_tolerance:
                    best_loss = loss
                    best_f = f_values
                    plateau_counter = 0
                else:
                    plateau_counter += 1

                if step % 5000 == 0 or step == self.hypers.num_steps - 1:
                    print(f"Step {step:5d} | C3 ≈ {loss:.12f} | Best = {best_loss:.12f}")

            print(f"Run {seed_idx + 1} final best C3: {best_loss:.12f}")
            
            if best_loss < best_overall_c3:
                best_overall_c3 = best_loss
                best_overall_f = best_f
                print(f"** New overall best: {best_overall_c3:.12f} **")

        print(f"\nFinal best C3 upper bound found: {best_overall_c3:.12f}")
        return best_overall_f, best_overall_c3


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()

    loss_val = final_c3_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c3_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END