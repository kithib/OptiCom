import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np


@dataclass
class Hyperparameters:
    max_integer: int = 70
    num_restarts: int = 2
    num_search_steps: int = 80
    initial_temperature: float = 0.012


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)

    @staticmethod
    def _objective_fn(u_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculates the C6 lower bound using jnp.unique"""
        U = jnp.where(u_mask)[0]

        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]

        size_U_plus_U = jnp.unique(sums).shape[0]
        size_U_minus_U = jnp.unique(diffs).shape[0]
        max_U = jnp.max(U)

        # Handle the case where max_U is 0 to avoid log(1)=0 in denominator
        if max_U == 0:
            return -1.0  # Return a low value for trivial sets

        ratio = size_U_minus_U / size_U_plus_U
        c6_bound = 1 + jnp.log(ratio) / jnp.log(2 * max_U + 1)

        return -c6_bound  # Return negative for maximization

    def anneal_step(self, key, temp, current_mask, current_loss):
        """Performs one step of Simulated Annealing (not JIT-compiled)."""
        key1, key2 = jax.random.split(key)
        
        # Propose 1 random mutation for speed and stability
        idx_to_flip = jax.random.randint(key1, (), 1, len(current_mask))
        neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])

        neighbor_loss = self._objective_fn(neighbor_mask)
        delta_loss = neighbor_loss - current_loss

        # Metropolis acceptance criterion
        should_accept = False
        if delta_loss < 0:
            should_accept = True
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key2) < accept_prob:
                should_accept = True

        if should_accept:
            return neighbor_mask, neighbor_loss
        else:
            return current_mask, current_loss


def run():
    hypers = Hyperparameters()
    main_key = jax.random.PRNGKey(42)

    best_loss = float("inf")
    best_set_np = None

    for i in range(hypers.num_restarts):
        restart_key, main_key = jax.random.split(main_key)
        loss, u_set_np = run_single_trial(hypers, restart_key)

        if loss < best_loss:
            best_loss = loss
            best_set_np = u_set_np

    c6_bound = -best_loss
    return best_set_np, c6_bound


def run_single_trial(hypers, key):
    # Initialize set with small size that typically achieves good ratios
    u_mask = jnp.zeros(hypers.max_integer + 1, dtype=jnp.bool_)
    u_mask = u_mask.at[0].set(True)
    
    # Seed with a known good configuration pattern to start from a good point
    key, subkey = jax.random.split(key)
    initial_set_size = 8
    indices = jax.random.choice(subkey, jnp.arange(1, hypers.max_integer + 1), 
                                 shape=(initial_set_size,), replace=False)
    u_mask = u_mask.at[indices].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    current_mask = u_mask
    for step in range(hypers.num_search_steps):
        key, subkey = jax.random.split(key)
        # Faster exponential cooling schedule
        current_temp = hypers.initial_temperature * jnp.exp(-step / (hypers.num_search_steps * 0.25))
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-6), current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set