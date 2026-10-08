import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 210
    num_restarts: int = 4
    num_search_steps: int = 800
    initial_temperature: float = 0.015


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)

    @staticmethod
    def _objective_fn(u_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculates the C6 lower bound using jnp.unique with optimized sizes"""
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
        """Performs one step of Simulated Annealing with adaptive neighborhood size."""
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # Flip 1 bit (occasionally 2 early on) for balanced exploration
        early_phase = jnp.float32(temp > 0.008)
        max_flips = 1 + jnp.array(early_phase, dtype=jnp.int32)
        num_flips = jax.random.randint(key1, (), 1, max_flips + 1)
        
        # Bias selection toward lower values (better performing region)
        weights = jnp.exp(-jnp.arange(1, len(current_mask)) / (len(current_mask) // 5))
        weights = weights / weights.sum()
        
        # With 50% chance, try flipping bits where current mask has Trues (removing elements)
        # This helps prune the set toward optimal sizes
        current_trues = jnp.where(current_mask[1:])[0] + 1  # Skip index 0
        if current_trues.shape[0] > 5 and jax.random.uniform(key4) < 0.5:
            idxs_to_flip = jax.random.choice(key2, current_trues, shape=(num_flips,), replace=False)
        else:
            idxs_to_flip = jax.random.choice(key2, jnp.arange(1, len(current_mask)),
                                              shape=(num_flips,), replace=False, p=weights)
        
        neighbor_mask = current_mask
        for idx in idxs_to_flip:
            neighbor_mask = neighbor_mask.at[idx].set(1 - neighbor_mask[idx])

        neighbor_loss = self._objective_fn(neighbor_mask)
        delta_loss = neighbor_loss - current_loss

        # Metropolis acceptance criterion with modified probability for equal quality
        should_accept = False
        if delta_loss < 0:
            should_accept = True
        elif delta_loss == 0:
            # Accept neutral moves with 50% chance to explore plateaus
            if jax.random.uniform(key3) < 0.5:
                should_accept = True
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key3) < accept_prob:
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
    print(f"\nSearch complete. Best C6 lower bound found: {c6_bound:.10f}")
    return best_set_np, c6_bound


def run_single_trial(hypers, key):
    # Fast initialization: focus on promising lower/mid range with known good density
    u_mask = jnp.zeros(hypers.max_integer + 1, dtype=bool)
    u_mask = u_mask.at[0].set(True)
    
    # Random 12-20 elements (smaller sets = faster objective evaluation)
    key, subkey = jax.random.split(key)
    num_initial = jax.random.randint(subkey, (), 12, 21)
    key, subkey = jax.random.split(key)
    
    # Bias selection toward lower values (historically better performing)
    weights = jnp.exp(-jnp.arange(1, hypers.max_integer + 1) / (hypers.max_integer // 5))
    weights = weights / weights.sum()
    
    initial_indices = jax.random.choice(subkey, jnp.arange(1, hypers.max_integer + 1),
                                        shape=(num_initial,), replace=False, p=weights)
    for idx in initial_indices:
        u_mask = u_mask.at[idx].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress", disable=True):
        key, subkey = jax.random.split(key)
        # Faster exponential cooling to reduce evaluation time
        current_temp = hypers.initial_temperature * (0.992 ** step)
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-7), current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set