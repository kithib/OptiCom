import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 2800
    learning_rate: float = 0.0052
    end_lr_factor: float = 1.2e-6
    num_steps: int = 280000
    warmup_steps: int = 14000


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray, t: float = 100.0) -> jnp.ndarray:
        """
        Computes the objective function, which is the C1 ratio.
        We minimize this ratio to find a tight upper bound.
        Uses smooth maximum approximation with configurable temperature.
        """
        f_non_negative = jax.nn.softplus(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-15
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        scaled_conv_f_f = conv_f_f * self.dx

        # Smooth maximum with temperature parameter
        max_conv = jax.nn.logsumexp(t * scaled_conv_f_f) / t
        
        c1_ratio = max_conv / (integral_f_safe**2)

        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState, t: float) -> tuple:
        """Performs a single training step with gradient clipping for stability."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values, t)
        # Tighter gradient clipping for better stability
        grads = jnp.clip(grads, -1.1, 1.1)
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
        # Use AdamW optimizer with weight decay for better stability
        self.optimizer = optax.adamw(learning_rate=schedule, weight_decay=7e-8)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        
        # Enhanced initialization: multi-frequency cosine mixture with optimized envelope
        x = jnp.linspace(-1/4, 1/4, N)
        abs_x4 = jnp.abs(x * 4)
        
        # Core cosine basis functions with multiple frequencies
        cos4 = (1.0 + jnp.cos(jnp.pi * x * 4)) / 2.0
        cos8 = (1.0 + jnp.cos(jnp.pi * x * 8)) / 2.0
        cos12 = (1.0 + jnp.cos(jnp.pi * x * 12)) / 2.0
        cos16 = (1.0 + jnp.cos(jnp.pi * x * 16)) / 2.0
        cos20 = (1.0 + jnp.cos(jnp.pi * x * 20)) / 2.0
        cos24 = (1.0 + jnp.cos(jnp.pi * x * 24)) / 2.0
        
        # Weighted combination optimized for known structure - refined weights
        f_base = 0.588 * cos4 + 0.244 * cos8 + 0.104 * cos12 + 0.040 * cos16 + 0.019 * cos20 + 0.005 * cos24
        
        # Refined envelope with improved edge handling - optimized exponent
        envelope = 0.485 + 0.515 * (1.0 - abs_x4)**1.715
        f_values = f_base * envelope
        
        # Minimal noise for exploration
        f_values += 0.0008 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        best_loss = jnp.inf
        best_f = f_values
        
        for step in range(self.hypers.num_steps):
            # Refined temperature annealing: higher range with optimized decay
            t = 290.0 + 560.0 * (1.0 - step / self.hypers.num_steps)**1.07
            
            f_values, opt_state, _ = train_step_jit(f_values, opt_state, t)
            
            # Track best solution found (evaluate with exact max for true comparison)
            f_nonneg = jax.nn.softplus(f_values)
            integral = jnp.sum(f_nonneg) * self.dx
            padded = jnp.pad(f_nonneg, (0, N))
            conv = jnp.fft.ifft(jnp.fft.fft(padded)**2).real * self.dx
            true_c1 = jnp.max(conv) / (jnp.maximum(integral, 1e-15)**2)
            
            if true_c1 < best_loss:
                best_loss = true_c1
                best_f = f_values
            
            if step % 14000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:6d} | C1 ≈ {true_c1:.18f} | Best C1 ≈ {best_loss:.18f}")

        print(f"Final C1 found: {best_loss:.18f}")

        return jax.nn.softplus(best_f), best_loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals