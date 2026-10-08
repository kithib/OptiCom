import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 360
    learning_rate: float = 0.02
    num_steps: int = 80000
    penalty_strength: float = 7000000.0
    log_alpha: float = 75.0
    clip_norm: float = 1.2
    lr_decay_alpha: float = 0.0003
    sharpness_factor: float = 4.0
    ema_decay: float = 0.999
    log_alpha_step: float = 80.0
    sharpness_step: float = 3.0


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, latent_h_values: jnp.ndarray, step: jnp.ndarray) -> jnp.ndarray:
        """
        The loss function includes the objective and a penalty for the constraint.
        Uses LogSumExp relaxation for differentiable max approximation with annealed penalty.
        """
        # Annealed sigmoid sharpening for steeper transitions late in training - slower sharpening ramp
        sharpen = 1.0 + (self.hypers.sharpness_factor - 1.0) * jnp.minimum(1.0, step / (self.hypers.num_steps * 0.7))
        h = jax.nn.sigmoid(sharpen * latent_h_values)

        # Calculate cross-correlation via FFT
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx

        # LogSumExp with annealed alpha - extended ramp for better approximation
        annealed_alpha = 25.0 + (self.hypers.log_alpha - 25.0) * jnp.minimum(1.0, step / (self.hypers.num_steps * 0.4))
        objective_loss = (1.0 / annealed_alpha) * jax.scipy.special.logsumexp(annealed_alpha * scaled_correlation)
        # Tighter lower bound ensuring we never under-estimate the max
        current_max = jnp.max(scaled_correlation)
        objective_loss = jnp.maximum(objective_loss, current_max)
        # Small penalty for having multiple peaks far from the max to encourage flattening
        peak_spread_penalty = jnp.mean((scaled_correlation - current_max) ** 2) * 0.01
        objective_loss = objective_loss + peak_spread_penalty

        # Calculate the penalty for the integral constraint
        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        # Annealed penalty strength - extended ramp to focus on objective longer
        anneal_factor = jnp.minimum(1.0, step / (self.hypers.num_steps * 0.1))
        current_penalty = self.hypers.penalty_strength * anneal_factor

        # Combine the objective with the penalty
        total_loss = objective_loss + current_penalty * constraint_loss
        return total_loss

    def run_optimization(self):
        # Learning rate schedule with cosine decay - extended fine-tuning phase
        lr_schedule = optax.cosine_decay_schedule(
            init_value=self.hypers.learning_rate,
            decay_steps=self.hypers.num_steps,
            alpha=self.hypers.lr_decay_alpha
        )
        # Gradient clipping for stability
        optimizer = optax.chain(
            optax.clip_by_global_norm(self.hypers.clip_norm),
            optax.adam(lr_schedule)
        )

        # Better initialization: centered at 0 with small noise (targeting sigmoid(0)=0.5 for integral ~1)
        key = jax.random.PRNGKey(42)
        latent_h_values = 0.05 * jax.random.normal(key, (self.hypers.num_intervals,))

        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def compute_objective(h):
            j = 1.0 - h
            N = self.hypers.num_intervals
            h_padded = jnp.pad(h, (0, N))
            j_padded = jnp.pad(j, (0, N))
            corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
            correlation = jnp.fft.ifft(corr_fft).real
            return jnp.max(correlation * self.dx)

        @jax.jit
        def train_step(latent_h_values, opt_state, step):
            loss, grads = jax.value_and_grad(self._objective_fn)(latent_h_values, step)
            updates, opt_state = optimizer.update(grads, opt_state)
            latent_h_values = optax.apply_updates(latent_h_values, updates)
            return latent_h_values, opt_state, loss

        print(f"Optimizing a step function with {self.hypers.num_intervals} intervals...")
        
        # EMA for tracking best latent values encountered during training
        best_latent = latent_h_values
        best_loss = jnp.inf
        
        for step in tqdm.tqdm(range(self.hypers.num_steps), desc="Optimizing"):
            latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state, jnp.array(step, dtype=jnp.float32))
            # Update best tracking after initial burn-in period
            if step > self.hypers.num_steps // 4 and loss < best_loss:
                best_loss = loss
                best_latent = latent_h_values * self.hypers.ema_decay + best_latent * (1 - self.hypers.ema_decay)
        
        # Use EMA-smoothed best latent values for final solution
        latent_h_values = best_latent

        # Final h is sigmoid of latent values - with post-hoc sharpening for binary-like behavior
        final_h = jax.nn.sigmoid(self.hypers.sharpness_step * latent_h_values)

        # Multi-pass integral constraint projection with tighter convergence
        integral_h = jnp.sum(final_h) * self.dx
        adjustment_factor = 1.0 / integral_h
        final_h = jnp.clip(final_h * adjustment_factor, 0.0, 1.0)
        # Second adjustment: uniform delta shift
        integral_h2 = jnp.sum(final_h) * self.dx
        delta = (1.0 - integral_h2) / (self.hypers.num_intervals * self.dx)
        final_h = jnp.clip(final_h + delta, 0.0, 1.0)
        # Extended third to tenth corrections: proportional adjustments for very tight integral
        for _ in range(10):
            final_integral = jnp.sum(final_h) * self.dx
            residual = 1.0 - final_integral
            if jnp.abs(residual) < 1e-14:
                break
            if residual > 0:
                available = jnp.sum(1.0 - final_h)
                if available > 1e-15:
                    increment = residual / available
                    final_h = final_h + increment * (1.0 - final_h)
            else:
                available = jnp.sum(final_h)
                if available > 1e-15:
                    decrement = -residual / available
                    final_h = final_h * (1.0 - decrement)

        # Re-calculate final objective loss without the penalty
        c5_bound = compute_objective(final_h)

        print(f"Optimization complete. Final C5 upper bound: {c5_bound:.14f}")
        return np.array(final_h), float(c5_bound)


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals