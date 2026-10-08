# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 300
    num_restarts: int = 8
    num_search_steps: int = 2000
    initial_temperature: float = 0.02


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
        """Performs one step of Simulated Annealing with enhanced neighborhood moves."""
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Choose mutation type: 50% single flip, 30% swap, 20% double flip
        move_type = jax.random.randint(key1, (), 0, 10)
        
        neighbor_mask = current_mask
        
        if move_type < 5:
            # Single flip mutation
            idx_to_flip = jax.random.randint(key2, (), 1, len(current_mask))
            neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])
        elif move_type < 8:
            # Swap: flip two bits to maintain approximate set size
            idx1 = jax.random.randint(key2, (), 1, len(current_mask))
            idx2 = jax.random.randint(key3, (), 1, len(current_mask))
            if idx1 != idx2 and current_mask[idx1] != current_mask[idx2]:
                val1 = current_mask[idx1]
                val2 = current_mask[idx2]
                neighbor_mask = current_mask.at[idx1].set(val2)
                neighbor_mask = neighbor_mask.at[idx2].set(val1)
        else:
            # Double flip: potentially explore further
            idx1 = jax.random.randint(key2, (), 1, len(current_mask))
            idx2 = jax.random.randint(key3, (), 1, len(current_mask))
            neighbor_mask = current_mask.at[idx1].set(1 - current_mask[idx1])
            neighbor_mask = neighbor_mask.at[idx2].set(1 - current_mask[idx2])

        neighbor_loss = self._objective_fn(neighbor_mask)
        delta_loss = neighbor_loss - current_loss

        # Metropolis acceptance criterion
        should_accept = False
        if delta_loss < 0:
            should_accept = True
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key) < accept_prob:
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
    # Initialize a random set with moderate density, ensuring 0 is included
    key, subkey = jax.random.split(key)
    # Try different sparsity levels per restart for better exploration
    sparsity = jax.random.uniform(subkey, minval=0.85, maxval=0.96)
    u_mask = jax.random.bernoulli(subkey, p=(1 - sparsity), shape=(hypers.max_integer + 1,))
    u_mask = u_mask.at[0].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    best_local_mask = current_mask
    best_local_loss = current_loss
    
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Exponential temperature decay for better exploitation late in search
        decay_factor = 0.9992
        current_temp = hypers.initial_temperature * (decay_factor ** step)
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-7), current_mask, current_loss
        )
        
        # Keep track of best in this trial
        if current_loss < best_local_loss:
            best_local_loss = current_loss
            best_local_mask = current_mask

    final_set = np.where(best_local_mask)[0]
    return best_local_loss, final_set


# EVOLVE-BLOCK-END