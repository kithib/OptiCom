# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 200
    learning_rate: float = 0.005
    num_steps: int = 20000
    penalty_strength: float = 1000000.0


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
        """
        # Enforce h(x) in [0, 1] via sigmoid (hard constraint)
        h = jax.nn.sigmoid(latent_h_values)

        # Calculate the primary objective (max correlation)
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx
        objective_loss = jnp.max(scaled_correlation)

        # Calculate the penalty for the integral constraint
        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        # Combine the objective with the penalty
        total_loss = objective_loss + self.hypers.penalty_strength * constraint_loss
        return total_loss

    def _calculate_c5_bound(self, h: jnp.ndarray) -> jnp.ndarray:
        """Helper method to calculate C5 bound for a given h - avoids code duplication."""
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        c5_bound = jnp.max(correlation * self.dx)
        return c5_bound

    def run_optimization(self):
        # Use learning rate scheduling for better convergence
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate * 2,
            warmup_steps=1000,
            decay_steps=self.hypers.num_steps
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(10.0),  # Gradient clipping to prevent instability
            optax.adam(lr_schedule)
        )

        key = jax.random.PRNGKey(42)
        # Better initialization centered around 0.5 (since h values are in [0,1])
        latent_h_values = jnp.zeros((self.hypers.num_intervals,))

        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def train_step(latent_h_values, opt_state):
            loss, grads = jax.value_and_grad(self._objective_fn)(latent_h_values)
            updates, opt_state = optimizer.update(grads, opt_state)
            latent_h_values = optax.apply_updates(latent_h_values, updates)
            return latent_h_values, opt_state, loss

        print(f"Optimizing a step function with {self.hypers.num_intervals} intervals...")
        best_latent = latent_h_values
        best_loss = float('inf')
        
        for step in tqdm.tqdm(range(self.hypers.num_steps), desc="Optimizing"):
            latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state)
            
            # Track best solution found during optimization
            if loss < best_loss:
                best_loss = loss
                best_latent = latent_h_values

        # Use best solution from optimization trajectory instead of final point
        final_h = jax.nn.sigmoid(best_latent)

        # Re-calculate final objective loss using helper method
        c5_bound = self._calculate_c5_bound(final_h)

        print(f"Optimization complete. Final C5 upper bound: {c5_bound:.8f}")
        return np.array(final_h), float(c5_bound)


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals


# EVOLVE-BLOCK-END