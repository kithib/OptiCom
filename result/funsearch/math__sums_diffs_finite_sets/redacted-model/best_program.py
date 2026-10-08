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
    initial_temperature: float = 0.05


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
        """Performs one step of Simulated Annealing with multiple mutation strategies."""
        key1, key2 = jax.random.split(key)
        
        # First mutation: flip a single random element
        idx_to_flip = jax.random.randint(key1, (), 1, len(current_mask))
        neighbor_mask = current_mask.at[idx_to_flip].set(1 - current_mask[idx_to_flip])
        
        # Second mutation with 30% chance: flip a second element
        if jax.random.uniform(key2) < 0.3:
            idx2 = jax.random.randint(key2, (), 1, len(current_mask))
            neighbor_mask = neighbor_mask.at[idx2].set(1 - neighbor_mask[idx2])

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
    # Initialize with adaptive density based on max_integer
    key, subkey = jax.random.split(key)
    target_size = int(0.15 * hypers.max_integer)
    sparsity = 1.0 - (target_size / hypers.max_integer)
    sparsity = jnp.clip(sparsity, 0.8, 0.95)
    
    u_mask = jax.random.bernoulli(subkey, p=(1 - sparsity), shape=(hypers.max_integer + 1,))
    u_mask = u_mask.at[0].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    plateau_counter = 0
    prev_loss = current_loss
    
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        
        # Exponential temperature decay for better exploration/exploitation balance
        decay_rate = 4.0 / hypers.num_search_steps
        current_temp = hypers.initial_temperature * jnp.exp(-decay_rate * step)
        current_temp = jnp.maximum(current_temp, 1e-7)
        
        current_mask, current_loss = searcher.anneal_step(
            subkey, current_temp, current_mask, current_loss
        )
        
        # Plateau detection: if no improvement after many steps, random restart in-trial
        if jnp.abs(current_loss - prev_loss) < 1e-9:
            plateau_counter += 1
            if plateau_counter > 200:
                key, perturb_key = jax.random.split(key)
                # Randomly flip 5% of bits to escape local optimum
                perturbations = jax.random.bernoulli(perturb_key, p=0.05, shape=current_mask.shape)
                perturbations = perturbations.at[0].set(False)  # Keep 0
                current_mask = jnp.where(perturbations, 1 - current_mask, current_mask)
                current_loss = searcher._objective_fn(current_mask)
                plateau_counter = 0
        else:
            plateau_counter = 0
        prev_loss = current_loss

    final_set = np.where(current_mask)[0]
    return current_loss, final_set


# EVOLVE-BLOCK-END