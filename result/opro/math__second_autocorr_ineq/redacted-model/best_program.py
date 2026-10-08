# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 180
    learning_rate: float = 0.015
    num_steps: int = 55000
    warmup_steps: int = 2000
    num_restarts: int = 5
    gradient_clip: float = 1.0
    weight_decay: float = 1e-5
    epsilon: float = 1e-10


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
        f_non_negative = jax.nn.softplus(f_values)

        # Unscaled discrete autoconvolution with proper symmetric padding
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (N, N))
        fft_f = jnp.fft.fft(padded_f)
        convolution = jnp.fft.ifft(fft_f * fft_f).real

        # Calculate L2-norm squared of the convolution (rigorous Simpson's method)
        num_conv_points = len(convolution)
        h = 1.0 / (num_conv_points + 1)
        y_points = jnp.concatenate([jnp.array([0.0]), convolution, jnp.array([0.0])])
        y1, y2 = y_points[:-1], y_points[1:]
        l2_norm_squared = jnp.sum((h / 3) * (y1**2 + y1 * y2 + y2**2))

        # Calculate L1-norm of the convolution
        norm_1 = jnp.sum(jnp.abs(convolution)) / (len(convolution) + 1)

        # Calculate infinity-norm of the convolution with small epsilon for stability
        norm_inf = jnp.max(jnp.abs(convolution)) + self.hypers.epsilon

        # Calculate C2 ratio
        denominator = norm_1 * norm_inf
        c2_ratio = l2_norm_squared / denominator

        # We want to MAXIMIZE C2, so the optimizer must MINIMIZE its negative.
        return -c2_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_single_optimization(self, init_values=None, run_idx=0):
        """Sets up and runs a single optimization run with optional initialization."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-5,
        )
        # Use AdamW optimizer with weight decay for regularization and gradient clipping
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.gradient_clip),
            optax.adamw(learning_rate=schedule, weight_decay=self.hypers.weight_decay)
        )

        if init_values is None:
            # Initialize with step-function-like profile (current champion hint) with small noise
            key = jax.random.PRNGKey(42 + run_idx * 100)
            x = jnp.linspace(0, 1, self.hypers.num_intervals)
            f_values = jnp.where((x > 0.2) & (x < 0.8), 1.0, 0.3)
            f_values += jax.random.uniform(key, (self.hypers.num_intervals,)) * 0.1
        else:
            f_values = init_values

        opt_state = self.optimizer.init(f_values)
        train_step_jit = jax.jit(self.train_step)

        best_f = f_values
        best_loss = jnp.inf

        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            
            # Track best solution found
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                current_c2 = -loss
                best_c2 = -best_loss
                print(f"Step {step:5d} | C2 ≈ {current_c2:.10f} | Best: {best_c2:.10f}")

        final_c2 = -self._objective_fn(best_f)
        print(f"Run complete. C2 ≈ {final_c2:.10f}")
        return jax.nn.softplus(best_f), final_c2

    def run_optimization(self):
        """Runs optimization with multiple restarts and returns best result."""
        print(f"Number of intervals (N): {self.hypers.num_intervals}, Steps per run: {self.hypers.num_steps}, Restarts: {self.hypers.num_restarts}")
        
        best_overall_c2 = 0.0
        best_overall_f = None
        
        for i in range(self.hypers.num_restarts):
            print(f"\n=== Run {i+1}/{self.hypers.num_restarts}:")
            if i == 0:
                f, c2 = self.run_single_optimization(run_idx=i)
            else:
                # Slightly perturb previous best solution for next initialization
                key = jax.random.PRNGKey(np.random.randint(0, 10000))
                noise = jax.random.normal(key, best_overall_f.shape) * 0.05
                # Inverse softplus to get proper init_values for perturbed exploration
                perturbed = best_overall_f + noise
                # Transform back through inverse softplus for optimization
                init_f = jnp.log(jnp.exp(perturbed) - 1.0 + self.hypers.epsilon)
                f, c2 = self.run_single_optimization(init_values=init_f, run_idx=i)
            
            if c2 > best_overall_c2:
                best_overall_c2 = c2
                best_overall_f = f
                print(f"New best C2: {best_overall_c2:.10f}")
        
        print(f"\nBest C2 lower bound found across all runs: {best_overall_c2:.12f}")
        return best_overall_f, best_overall_c2


def run():
    """Entry point for running the optimization."""
    np.random.seed(42)  # Ensure reproducibility
    hypers = Hyperparameters()
    optimizer = C2Optimizer(hypers)
    optimized_f, final_c2_val = optimizer.run_optimization()

    loss_val = -final_c2_val
    f_values_np = np.array(optimized_f)

    return f_values_np, float(final_c2_val), float(loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END