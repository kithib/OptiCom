# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 500
    num_restarts: int = 8
    num_search_steps: int = 3000
    initial_temperature: float = 0.05
    cooldown_factor: float = 0.999


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)

    @staticmethod
    @jit
    def _objective_fn(u_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculates the C6 lower bound using jnp.unique - JIT compiled for speed."""
        U = jnp.where(u_mask)[0]

        sums = U[:, None] + U[None, :]
        diffs = U[:, None] - U[None, :]

        size_U_plus_U = jnp.unique(sums).shape[0]
        size_U_minus_U = jnp.unique(diffs).shape[0]
        max_U = jnp.max(U)

        # Handle the case where max_U is 0 to avoid log(1)=0 in denominator
        ratio = jnp.where(max_U == 0, 0.5, size_U_minus_U / size_U_plus_U)
        denom = jnp.log(2 * max_U + 1)
        c6_bound = 1 + jnp.log(ratio) / jnp.where(denom == 0, 1.0, denom)
        c6_bound = jnp.where(max_U == 0, -1.0, c6_bound)

        return -c6_bound  # Return negative for maximization

    def anneal_step(self, key, temp, current_mask, current_loss):
        """Performs one step of Simulated Annealing with multiple mutation types."""
        # Multi-neighbor proposal: try 3 mutations and pick the best neighbor
        best_neighbor_mask = current_mask
        best_neighbor_loss = float('inf')
        
        for _ in range(3):
            key, subkey = jax.random.split(key)
            # Occasionally flip multiple bits for larger jumps (15% chance)
            if jax.random.uniform(subkey) < 0.15:
                # Flip 2-3 bits to explore further
                num_flips = jax.random.randint(subkey, (), 2, 4)
                neighbor_mask = current_mask
                for __ in range(num_flips):
                    key, subkey2 = jax.random.split(key)
                    idx_to_flip = jax.random.randint(subkey2, (), 1, len(current_mask))
                    neighbor_mask = neighbor_mask.at[idx_to_flip].set(1 - neighbor_mask[idx_to_flip])
            else:
                # Single bit flip
                idx_to_flip = jax.random.randint(subkey, (), 1, len(current_mask))
                neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])

            neighbor_loss = self._objective_fn(neighbor_mask)
            if neighbor_loss < best_neighbor_loss:
                best_neighbor_loss = neighbor_loss
                best_neighbor_mask = neighbor_mask

        delta_loss = best_neighbor_loss - current_loss

        # Metropolis acceptance criterion
        if delta_loss < 0:
            return best_neighbor_mask, best_neighbor_loss
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key) < accept_prob:
                return best_neighbor_mask, best_neighbor_loss
            else:
                return current_mask, current_loss


def run():
    hypers = Hyperparameters()
    main_key = jax.random.PRNGKey(42)

    best_loss = float("inf")
    best_set_np = None

    for i in range(hypers.num_restarts):
        print(f"\n{'='*20} Restart {i+1}/{hypers.num_restarts} {'='*20}")
        restart_key, main_key = jax.random.split(main_key)
        loss, u_set_np = run_single_trial(hypers, restart_key)

        if loss < best_loss:
            print(f"New best C6 bound found: {-loss:.8f}")
            best_loss = loss
            best_set_np = u_set_np

    c6_bound = -best_loss
    print(f"\nSearch complete. Best C6 lower bound found: {c6_bound:.8f}")
    return best_set_np, c6_bound


def run_single_trial(hypers, key):
    # Initialize with smarter sampling: denser near 0, sparser at higher values
    key, subkey = jax.random.split(key)
    # Create density gradient: higher probability of including small numbers
    positions = jnp.arange(hypers.max_integer + 1, dtype=jnp.float32)
    density_probs = jnp.exp(-positions / (hypers.max_integer / 4)) * 0.3 + 0.05
    u_mask = jax.random.bernoulli(subkey, p=density_probs, shape=(hypers.max_integer + 1,))
    u_mask = u_mask.at[0].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-float(current_loss):.6f}")

    current_mask = u_mask
    current_temp = hypers.initial_temperature
    
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Exponential cooldown for better temperature scheduling
        current_temp = current_temp * hypers.cooldown_factor
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-7), current_mask, current_loss
        )

    final_set = np.where(np.array(current_mask))[0]
    return float(current_loss), final_set


# EVOLVE-BLOCK-END