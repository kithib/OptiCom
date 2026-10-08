import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 1280
    learning_rate: float = 0.0144
    num_steps: int = 98000
    warmup_steps: int = 9800
    num_restarts: int = 9
    weight_decay: float = 5.95e-6


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
        eps = 1e-14
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

        # Light regularization - small coefficient to not bias the solution significantly
        reg = 7.3e-7 * jnp.sum(f_values**2) * self.dx

        return c3_ratio + reg

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process with multi-restart."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 4.0e-5,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.99),
            optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)
        )

        overall_best_f = None
        overall_best_loss = jnp.inf
        
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}, Restarts: {self.hypers.num_restarts}"
        )

        for restart in range(self.hypers.num_restarts):
            key = jax.random.PRNGKey(42 + restart * 100)
            # Enhanced structured trigonometric initialization with refined envelope and coefficients
            n = self.hypers.num_intervals
            x = jnp.linspace(-0.25, 0.25, n)
            envelope = ((1.0 - (4.0 * x) ** 2) ** 2) ** 1.575
            # Further refined trigonometric basis with carefully tuned coefficients and higher harmonics
            structured_part = envelope * (
                jnp.cos(4 * jnp.pi * x) - 0.3315 * jnp.cos(8 * jnp.pi * x) + 
                0.1911 * jnp.cos(12 * jnp.pi * x) - 0.0893 * jnp.sin(6 * jnp.pi * x) +
                0.05085 * jnp.sin(10 * jnp.pi * x) - 0.0255 * jnp.cos(16 * jnp.pi * x) +
                0.0154 * jnp.sin(14 * jnp.pi * x) - 0.0103 * jnp.cos(20 * jnp.pi * x) -
                0.00835 * jnp.sin(18 * jnp.pi * x) + 0.00523 * jnp.cos(24 * jnp.pi * x) -
                0.00327 * jnp.sin(22 * jnp.pi * x) + 0.00212 * jnp.cos(28 * jnp.pi * x) -
                0.001405 * jnp.sin(26 * jnp.pi * x) + 0.000925 * jnp.cos(32 * jnp.pi * x) -
                0.00062 * jnp.sin(30 * jnp.pi * x) + 0.000425 * jnp.cos(36 * jnp.pi * x) -
                0.00028 * jnp.sin(34 * jnp.pi * x) + 0.000195 * jnp.cos(40 * jnp.pi * x) -
                0.000135 * jnp.sin(38 * jnp.pi * x) + 0.00010 * jnp.cos(44 * jnp.pi * x) -
                0.00007 * jnp.sin(42 * jnp.pi * x) + 0.000052 * jnp.cos(48 * jnp.pi * x)
            )
            random_part = 0.0545 * jax.random.normal(key, (n,))
            f_values = random_part + structured_part

            opt_state = self.optimizer.init(f_values)
            train_step_jit = jax.jit(self.train_step)

            loss = jnp.inf
            best_loss = jnp.inf
            best_f = f_values
            
            for step in range(self.hypers.num_steps):
                f_values, opt_state, loss = train_step_jit(f_values, opt_state)
                # Track best solution within this restart
                if loss < best_loss:
                    best_loss = loss
                    best_f = f_values
                if step % 19600 == 0 or step == self.hypers.num_steps - 1:
                    print(f"Restart {restart+1} Step {step:5d} | C3 ≈ {loss:.14f} | Best (restart) ≈ {best_loss:.14f}")

            if best_loss < overall_best_loss:
                overall_best_loss = best_loss
                overall_best_f = best_f
                print(f"New overall best C3: {overall_best_loss:.14f}")

        print(f"Final C3 upper bound found: {overall_best_loss:.14f}")
        return overall_best_f, overall_best_loss


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()

    loss_val = final_c3_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c3_val), float(loss_val), hypers.num_intervals