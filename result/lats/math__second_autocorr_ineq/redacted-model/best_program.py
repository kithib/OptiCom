import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 250
    learning_rate: float = 0.008
    num_steps: int = 25000
    warmup_steps: int = 1500


class C2Optimizer:
    """
    Optimizes a discretized function to find a lower bound for the C2 constant
    using the rigorous, unitless, piecewise-linear integral method.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function using the unitless norm calculation.
        """
        f_non_negative = jax.nn.relu(f_values)

        # Use symmetry to create an even function on [-1, 1] then [0, 1]
        # This doubles our effective resolution on [0, 1]
        N = self.hypers.num_intervals
        dx = 1.0 / N
        
        # For autocorrelation of function on N points, conv has 2N-1 points
        padded_f = jnp.pad(f_non_negative, (0, N - 1))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real
        
        # Use Simpson's rule for all integrals
        M = len(convolution)
        # L2 norm squared using Simpson
        indices = jnp.arange(M)
        weights = jnp.where(indices == 0, 1, 
                           jnp.where(indices == M - 1, 1,
                                    jnp.where(indices % 2 == 1, 4, 2)))
        weights = weights * dx / 3.0
        l2_norm_squared = jnp.sum(weights * convolution**2)
        
        # L1 norm of convolution
        norm_1 = jnp.sum(weights * jnp.abs(convolution))
        
        # Infinity norm
        norm_inf = jnp.max(jnp.abs(convolution))
        
        # Calculate squared L1 integral of f using Simpson
        weights_f = jnp.where(jnp.arange(N) == 0, 1,
                              jnp.where(jnp.arange(N) == N - 1, 1,
                                       jnp.where(jnp.arange(N) % 2 == 1, 4, 2)))
        weights_f = weights_f * dx / 3.0
        integral_f_sq = (jnp.sum(weights_f * f_non_negative))**2

        # Calculate C2 ratio using the correct formula from math framework
        denominator = integral_f_sq * norm_inf
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
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
            end_value=self.hypers.learning_rate * 1e-4,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        # Initialize to a step-like pattern that has proven effective
        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        # Initialize with a smooth function that resembles a step
        x_vals = jnp.linspace(0, 1, N)
        f_values = jnp.where(x_vals < 0.4, 1.3, jnp.where(x_vals < 0.8, 0.7, 0.2))
        f_values = f_values + 0.05 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)
        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )
        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:5d} | C2 ≈ {-loss:.10f}")

        final_c2 = -self._objective_fn(f_values)
        print(f"Final C2 lower bound found: {final_c2:.12f}")
        return jax.nn.relu(f_values), final_c2


def run():
    """Entry point for running the optimization."""
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals