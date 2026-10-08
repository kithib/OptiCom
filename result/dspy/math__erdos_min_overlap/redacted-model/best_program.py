# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 400
    learning_rate: float = 0.01
    num_steps: int = 30000
    penalty_strength: float = 2000000.0


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, latent_h_values: jnp.ndarray) -> jnp.ndarray:
        """
        The loss function includes the objective and a penalty for the constraint.
        Uses softplus for tighter [0,1] constraint and softmax for smoother max approximation.
        """
        # Enforce h(x) in [0, 1] via sigmoid with temperature for sharper transitions
        h = jax.nn.sigmoid(2.0 * latent_h_values)

        # Calculate the primary objective (max correlation)
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx
        
        # Use softmax for a smoother approximation of max to improve gradient flow
        temperature = 10.0
        weights = jax.nn.softmax(temperature * scaled_correlation)
        objective_loss = jnp.sum(weights * scaled_correlation)
        
        # Also add the true max for harder convergence at the end
        objective_loss = 0.5 * objective_loss + 0.5 * jnp.max(scaled_correlation)

        # Calculate the penalty for the integral constraint
        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        # Combine the objective with the penalty
        total_loss = objective_loss + self.hypers.penalty_strength * constraint_loss
        return total_loss

    def run_optimization(self):
        # Use learning rate scheduling for better convergence
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=500,
            decay_steps=self.hypers.num_steps,
            end_value=1e-6
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )

        # Better initialization: start near uniform distribution (integral = 1)
        key = jax.random.PRNGKey(42)
        uniform_val = self.hypers.num_intervals / 2.0  # uniform 0.5 gives integral 1
        latent_h_values = jnp.full((self.hypers.num_intervals,), jnp.log(uniform_val / (self.hypers.num_intervals - uniform_val)))
        latent_h_values = latent_h_values + 0.1 * jax.random.normal(key, (self.hypers.num_intervals,))

        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def train_step(latent_h_values, opt_state):
            loss, grads = jax.value_and_grad(self._objective_fn)(latent_h_values)
            updates, opt_state = optimizer.update(grads, opt_state)
            latent_h_values = optax.apply_updates(latent_h_values, updates)
            return latent_h_values, opt_state, loss

        print(f"Optimizing a step function with {self.hypers.num_intervals} intervals...")
        for step in tqdm.tqdm(range(self.hypers.num_steps), desc="Optimizing"):
            latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state)

        # Final h with sharper sigmoid
        final_h = jax.nn.sigmoid(3.0 * latent_h_values)
        # Force integral constraint by renormalizing
        integral_h = jnp.sum(final_h) * self.dx
        final_h = final_h / integral_h
        # Clip to [0, 1] after renormalization
        final_h = jnp.clip(final_h, 0.0, 1.0)

        # Re-calculate final objective loss without the penalty for the report
        j = 1.0 - final_h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(final_h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        c5_bound = jnp.max(correlation * self.dx)

        print(f"Optimization complete. Final C5 upper bound: {c5_bound:.12f}")
        return np.array(final_h), float(c5_bound)


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals


# EVOLVE-BLOCK-END