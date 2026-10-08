# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
from jax import jit
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 350
    num_restarts: int = 8
    num_search_steps: int = 3000
    initial_temperature: float = 0.08
    cooling_exponent: float = 3.0
    min_temp: float = 1e-7


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)
        self._objective_jit = jit(self._objective_fn)

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
        """Performs one step of Simulated Annealing."""
        # Propose multiple mutations and pick best (variable neighborhood)
        key1, key2, key3 = jax.random.split(key, 3)
        
        # Single flip mutation
        idx_to_flip = jax.random.randint(key1, (), 1, len(current_mask))
        neighbor_mask1 = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])
        
        # Double flip mutation (for larger search steps early on)
        idx1, idx2 = jax.random.randint(key2, (2,), 1, len(current_mask))
        neighbor_mask2 = current_mask.at[idx1].set(1 - current_mask[idx1])
        neighbor_mask2 = neighbor_mask2.at[idx2].set(1 - neighbor_mask2[idx2])
        
        # Evaluate both neighbors
        loss1 = self._objective_jit(neighbor_mask1)
        loss2 = self._objective_jit(neighbor_mask2)
        
        # Pick better neighbor
        neighbor_mask = jnp.where(loss1 <= loss2, neighbor_mask1, neighbor_mask2)
        neighbor_loss = jnp.minimum(loss1, loss2)
        
        delta_loss = neighbor_loss - current_loss

        # Metropolis acceptance criterion
        should_accept = False
        if delta_loss < 0:
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
    print(f"\nSearch complete. Best C6 lower bound found: {c6_bound:.8f}")
    return best_set_np, c6_bound


def run_single_trial(hypers, key):
    # Initialize with better heuristic: include all small numbers + some larger
    key, subkey = jax.random.split(key)
    
    # First, create initial mask: always include 0 and small numbers, then random selection
    u_mask_np = np.zeros(hypers.max_integer + 1, dtype=bool)
    u_mask_np[0] = True
    
    # Include numbers with known good properties: powers of 2 and values around sqrt(max_val)
    sqrt_max = int(np.sqrt(hypers.max_integer))
    for i in range(1, min(sqrt_max + 5, hypers.max_integer)):
        u_mask_np[i] = True
    
    # Add all powers of 2
    p = 1
    while p <= hypers.max_integer:
        u_mask_np[p] = True
        p *= 2
    
    # Add some random numbers
    random_vals = jax.random.randint(subkey, (25,), sqrt_max + 5, hypers.max_integer + 1)
    for v in random_vals:
        if v <= hypers.max_integer:
            u_mask_np[v] = True
    
    u_mask = jnp.array(u_mask_np)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_jit(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Non-linear cooling schedule - slower cooling initially
        progress = step / hypers.num_search_steps
        current_temp = hypers.initial_temperature * (1 - progress) ** hypers.cooling_exponent
        current_temp = jnp.maximum(current_temp, hypers.min_temp)
        current_mask, current_loss = searcher.anneal_step(
            subkey, current_temp, current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set


# EVOLVE-BLOCK-END