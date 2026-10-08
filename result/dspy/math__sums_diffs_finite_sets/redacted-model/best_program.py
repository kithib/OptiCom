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
    cool_restart_temp: float = 0.02
    cool_restart_steps: int = 500


@jit
def objective_fn_jit(u_mask: jnp.ndarray) -> jnp.ndarray:
    """JIT-compiled C6 objective calculation for speed."""
    U = jnp.where(u_mask, size=jnp.sum(u_mask))[0]
    
    sums = U[:, None] + U[None, :]
    diffs = U[:, None] - U[None, :]
    
    size_U_plus_U = jnp.unique(sums).shape[0]
    size_U_minus_U = jnp.unique(diffs).shape[0]
    max_U = jnp.max(U)
    
    ratio = size_U_minus_U / size_U_plus_U
    c6_bound = 1 + jnp.log(ratio) / jnp.log(2 * max_U + 1)
    
    return jnp.where(max_U == 0, -1.0, -c6_bound)


@jit
def flip_single_bit(mask, idx):
    return mask.at[idx].set(1 - mask[idx])


class C6Searcher:
    """
    Searches for a set U with optimized JIT-compiled operations.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.allowed_values = jnp.array((-1, 0, 1), dtype=jnp.int32)

    @staticmethod
    def _objective_fn(u_mask: jnp.ndarray) -> jnp.ndarray:
        """Calculates the C6 lower bound - uses JIT version internally."""
        return objective_fn_jit(u_mask)

    def anneal_step(self, key, temp, current_mask, current_loss):
        """Performs one step of Simulated Annealing with optimized operations."""
        # Propose a random mutation (exclude 0 index)
        idx_to_flip = jax.random.randint(key, (), 1, len(current_mask))
        neighbor_mask = flip_single_bit(current_mask, idx_to_flip)
        
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

    def generate_geometric_set(self):
        """Generate a set inspired by geometric progressions (known to give good C6 values)."""
        mask = np.zeros(self.hypers.max_integer + 1, dtype=bool)
        mask[0] = True
        
        # Try base-3 like constructions (similar to the Cantor set construction)
        base = 3
        max_val = self.hypers.max_integer
        val = 1
        while val <= max_val:
            if val <= max_val:
                mask[val] = True
            val *= base
        
        # Add some extra values around powers
        for p in range(1, 10):
            v = (2 ** p)
            if v <= max_val:
                mask[v] = True
                if v + 1 <= max_val:
                    mask[v + 1] = True
        
        return jnp.array(mask)

    def greedy_local_search(self, mask, key, steps=200):
        """Perform greedy hill climbing after SA to polish."""
        current_mask = mask
        current_loss = self._objective_fn(current_mask)
        
        for _ in range(steps):
            improved = False
            # Try to flip each bit greedily
            indices = jax.random.permutation(key, jnp.arange(1, len(current_mask)))
            for idx in indices[:50]:  # Limit to random 50 for speed
                test_mask = current_mask.at[idx].set(1 - current_mask[idx])
                test_loss = self._objective_fn(test_mask)
                if test_loss < current_loss:
                    current_mask, current_loss = test_mask, test_loss
                    improved = True
            if not improved:
                break
        return current_mask, current_loss


def run():
    hypers = Hyperparameters()
    main_key = jax.random.PRNGKey(42)

    best_loss = float("inf")
    best_set_np = None

    for i in range(hypers.num_restarts):
        print(f"\n{'='*20} Restart {i+1}/{hypers.num_restarts} {'='*20}")
        restart_key, main_key = jax.random.split(main_key)
        loss, u_set_np = run_single_trial(hypers, restart_key, i)

        if loss < best_loss:
            print(f"New best C6 bound found: {-loss:.8f}")
            best_loss = loss
            best_set_np = u_set_np

    c6_bound = -best_loss
    print(f"\nSearch complete. Best C6 lower bound found: {c6_bound:.8f}")
    return best_set_np, c6_bound


def run_single_trial(hypers, key, restart_idx):
    searcher = C6Searcher(hypers)
    
    # Mix initialization strategies
    if restart_idx == 0:
        # First restart: use geometrically-inspired set
        u_mask = searcher.generate_geometric_set()
    elif restart_idx < 3:
        # Medium density restarts
        sparsity = 0.85
        u_mask = jax.random.bernoulli(key, p=(1 - sparsity), shape=(hypers.max_integer + 1,))
        u_mask = u_mask.at[0].set(True)
    else:
        # Sparse restarts
        sparsity = 0.92
        u_mask = jax.random.bernoulli(key, p=(1 - sparsity), shape=(hypers.max_integer + 1,))
        u_mask = u_mask.at[0].set(True)

    current_loss = searcher._objective_fn(u_mask)
    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    
    # Simulated annealing loop
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Exponential cooling schedule for better late-stage exploitation
        decay_rate = 4.0 / hypers.num_search_steps
        current_temp = hypers.initial_temperature * jnp.exp(-decay_rate * step)
        current_mask, current_loss = searcher.anneal_step(
            subkey, jnp.maximum(current_temp, 1e-7), current_mask, current_loss
        )
    
    # Greedy polish phase
    print(f"After SA: C6 = {-current_loss:.6f}, applying greedy polish...")
    key, subkey = jax.random.split(key)
    current_mask, current_loss = searcher.greedy_local_search(current_mask, subkey)
    print(f"After polish: C6 = {-current_loss:.6f}")

    final_set = np.array(jnp.where(current_mask)[0])
    return current_loss, final_set


# EVOLVE-BLOCK-END