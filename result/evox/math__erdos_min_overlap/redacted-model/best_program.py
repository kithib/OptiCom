import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 400
    learning_rate: float = 0.003
    num_steps: int = 40000
    penalty_strength: float = 2000000.0
    log_barrier_strength: float = 10.0


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
        The loss function includes the objective, log-barrier smooth-max, and penalties.
        """
        # Enforce h(x) in [0, 1] via sigmoid
        h = jax.nn.sigmoid(latent_h_values)

        # Calculate correlation
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx

        # Smooth-max (log-sum-exp) for better gradient flow
        alpha = self.hypers.log_barrier_strength
        objective_loss = (1.0 / alpha) * jnp.log(jnp.sum(jnp.exp(alpha * scaled_correlation)))

        # Integral constraint penalty
        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        total_loss = objective_loss + self.hypers.penalty_strength * constraint_loss
        return total_loss

    def run_optimization(self):
        # Learning rate schedule
        schedule = optax.cosine_decay_schedule(
            self.hypers.learning_rate,
            decay_steps=self.hypers.num_steps
        )
        optimizer = optax.adam(schedule)

        key = jax.random.PRNGKey(42)
        latent_h_values = jax.random.normal(key, (self.hypers.num_intervals,)) * 0.5

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

        # Final h is the sigmoid of latent values
        final_h = jax.nn.sigmoid(latent_h_values)

        # Re-calculate final objective without penalty for report
        j = 1.0 - final_h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(final_h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        c5_bound = jnp.max(correlation * self.dx)

        print(f"Optimization complete. Final C5 upper bound: {c5_bound:.15f}")
        return np.array(final_h), float(c5_bound)


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals