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
    num_steps: int = 40000
    penalty_strength: float = 1000000.0
    logit_scale: float = 5.0


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
        Uses logit annealing to encourage binary-like solutions.
        """
        # Enforce h(x) in [0, 1] via scaled sigmoid to encourage crisp decisions
        h = jax.nn.sigmoid(self.hypers.logit_scale * latent_h_values)

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

    def run_optimization(self):
        # Use learning rate scheduling
        lr_schedule = optax.cosine_decay_schedule(
            init_value=self.hypers.learning_rate,
            decay_steps=self.hypers.num_steps,
            alpha=0.01
        )
        optimizer = optax.adam(lr_schedule)

        # Better initialization - uniform distribution centered around 0
        key = jax.random.PRNGKey(42)
        latent_h_values = jax.random.uniform(key, (self.hypers.num_intervals,), minval=-1.0, maxval=1.0)

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

        # Final h is just the sigmoid of the latent values
        final_h = jax.nn.sigmoid(self.hypers.logit_scale * latent_h_values)

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