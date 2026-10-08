# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 210
    num_restarts: int = 5
    num_search_steps: int = 800
    initial_temperature: float = 0.021


class C6Searcher:
    """
    Searches for a set U by running the search in pure Python for correctness.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)

    @staticmethod
    def _objective_fn(u_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculates the C6 lower bound using jnp.unique with size optimization."""
        U = jnp.where(u_mask)[0]
        set_size = U.shape[0]

        # Tuned early exit with loss penalty for oversized/undersized sets
        if set_size > 58:
            return 1.0 + (set_size - 58) * 0.09
        if set_size > 48:
            return 1.0 + (set_size - 48) * 0.045
        if set_size < 26:
            return 1.0 + (26 - set_size) * 0.09
        if set_size < 32:
            return 1.0 + (32 - set_size) * 0.035

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
        """Performs one step of Simulated Annealing with structured mutations."""
        key1, key2, key3, key4, key5, key6, key7, key8 = jax.random.split(key, 8)
        
        # Start with current mask
        neighbor_mask = current_mask
        
        # Skew index selection toward larger indices (they have less impact on sumset growth)
        u = jax.random.uniform(key1)
        # Tuned skewed distribution: stronger preference for higher indices
        skewed_idx = 1 + jnp.floor((len(current_mask) - 1) * (u ** 0.19)).astype(jnp.int32)
        neighbor_mask = neighbor_mask.at[skewed_idx].set(1 - neighbor_mask[skewed_idx])
        
        # Second mutation (46% chance): flip another index for larger jumps
        if jax.random.uniform(key2) < 0.46:
            u2 = jax.random.uniform(key3)
            idx2 = 1 + jnp.floor((len(current_mask) - 1) * (u2 ** 0.23)).astype(jnp.int32)
            neighbor_mask = neighbor_mask.at[idx2].set(1 - neighbor_mask[idx2])
        
        # Third mutation (30% chance): toggle the max integer - improves denominator
        if jax.random.uniform(key4) < 0.30:
            neighbor_mask = neighbor_mask.at[self.hypers.max_integer].set(
                1 - neighbor_mask[self.hypers.max_integer]
            )
        
        # Fourth mutation (18% chance): also toggle a high-mid range integer
        if jax.random.uniform(key5) < 0.18:
            mid_high = int(self.hypers.max_integer * 0.80) + jax.random.randint(key6, (), 0, int(self.hypers.max_integer * 0.20))
            mid_high = jnp.minimum(mid_high, self.hypers.max_integer)
            mid_high = jnp.maximum(mid_high, 1)
            neighbor_mask = neighbor_mask.at[mid_high].set(1 - neighbor_mask[mid_high])
        
        # Fifth mutation (10% chance): toggle small integer to adjust sumset growth carefully
        if jax.random.uniform(key7) < 0.10:
            small_idx = jax.random.randint(key8, (), 1, 14)
            neighbor_mask = neighbor_mask.at[small_idx].set(1 - neighbor_mask[small_idx])

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
    # Initialize with structured distribution: moderate sparsity with geometric bias
    key, subkey = jax.random.split(key)
    
    # Create probabilities that increase with index (favor larger integers initially)
    # Larger integers contribute less to sumset growth relative to diffset growth
    base_probs = jnp.arange(hypers.max_integer + 1, dtype=jnp.float32) + 1.0
    base_probs = base_probs ** 0.82  # Even stronger bias toward larger integers
    base_probs = base_probs / jnp.sum(base_probs) * (hypers.max_integer + 1) * 0.190  # Slightly higher density
    
    # Generate random uniform and compare against probs
    random_vals = jax.random.uniform(subkey, shape=(hypers.max_integer + 1,))
    u_mask = random_vals < base_probs
    
    # Ensure 0 is included (required constraint)
    u_mask = u_mask.at[0].set(True)
    
    # Include max_integer initially to improve denominator potential
    u_mask = u_mask.at[hypers.max_integer].set(True)
    
    # Include small integers to build up sum/diff sets efficiently
    for small_idx in range(1, min(8, hypers.max_integer + 1)):
        u_mask = u_mask.at[small_idx].set(True)
    
    # Include strategic high range points for good diffset/sumset ratio
    for high_idx in [int(hypers.max_integer * f) for f in [0.37, 0.47, 0.57, 0.67, 0.75, 0.82, 0.89, 0.95]]:
        if high_idx <= hypers.max_integer and high_idx > 0:
            u_mask = u_mask.at[high_idx].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Tuned exponential cooling: faster convergence to avoid timeout while maintaining quality
        current_temp = hypers.initial_temperature * (0.9995 ** step)
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-9), current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set


# EVOLVE-BLOCK-END