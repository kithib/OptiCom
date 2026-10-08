import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 220
    num_restarts: int = 6
    num_search_steps: int = 1100
    initial_temperature: float = 0.015
    cooldown_rate: float = 0.996


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers

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
        """Performs one step of Simulated Annealing with small mutations for efficiency."""
        key1, key2 = jax.random.split(key, 2)
        
        # Efficient single flip mutation (70% flip, 20% add, 10% remove)
        mutation_type = jax.random.randint(key1, (), 0, 100)
        
        if mutation_type < 70:
            # Single flip mutation (simple and efficient)
            idx_to_flip = jax.random.randint(key2, (), 1, len(current_mask))
            neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])
        elif mutation_type < 90:
            # Add a random 0 element
            false_indices = jnp.where(current_mask[1:] == False)[0] + 1
            if len(false_indices) > 0:
                chosen_idx = false_indices[jax.random.randint(key2, (), 0, len(false_indices))]
                neighbor_mask = current_mask.at[chosen_idx].set(True)
            else:
                idx_to_flip = jax.random.randint(key2, (), 1, len(current_mask))
                neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])
        else:
            # Remove a random element (except 0)
            true_indices = jnp.where(current_mask[1:] == True)[0] + 1
            if len(true_indices) > 2:  # Keep at least {0, x}
                chosen_idx = true_indices[jax.random.randint(key2, (), 0, len(true_indices))]
                neighbor_mask = current_mask.at[chosen_idx].set(False)
            else:
                idx_to_flip = jax.random.randint(key2, (), 1, len(current_mask))
                neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])

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
    # Efficient initialization: arithmetic progression with controlled randomness
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Start with a base set using arithmetic progression
    base_step = 5  # Fixed step for consistency and speed
    u_mask = jnp.zeros(hypers.max_integer + 1, dtype=jnp.bool_)
    for i in range(0, hypers.max_integer + 1, base_step):
        u_mask = u_mask.at[i].set(True)
    
    # Add moderate number of random elements (balanced for |U-U| / |U+U|)
    random_adds = jax.random.bernoulli(subkey2, p=0.1, shape=(hypers.max_integer + 1,))
    u_mask = jnp.logical_or(u_mask, random_adds)
    u_mask = u_mask.at[0].set(True)  # Ensure 0 is always included

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    current_temp = hypers.initial_temperature
    
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Exponential cooling
        current_temp = current_temp * hypers.cooldown_rate
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-7), current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set