# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
from dataclasses import dataclass
import numpy as np
import tqdm


@dataclass
class Hyperparameters:
    max_integer: int = 280
    num_restarts: int = 7
    num_search_steps: int = 1800
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
        """Calculates the C6 lower bound using jnp.unique with size optimization"""
        U = jnp.where(u_mask)[0]
        n = U.shape[0]

        # Handle edge case
        if n <= 1:
            return -1.0

        # Efficient sum/difference computation
        sums = U[:, None] + U
        diffs = U[:, None] - U

        # Get sizes
        size_U_plus_U = jnp.unique(sums).shape[0]
        size_U_minus_U = jnp.unique(diffs).shape[0]
        max_U = U[-1]  # U is sorted, so last element is max

        # Handle the case where max_U is 0 to avoid log(1)=0 in denominator
        if max_U == 0:
            return -1.0  # Return a low value for trivial sets

        ratio = size_U_minus_U / size_U_plus_U
        c6_bound = 1 + jnp.log(ratio) / jnp.log(2 * max_U + 1)

        return -c6_bound  # Return negative for maximization

    def anneal_step(self, key, temp, current_mask, current_loss):
        """Performs one step of Simulated Annealing with enhanced mutation operators."""
        key1, key2, key3, key4 = jax.random.split(key, 4)
        
        # Choose mutation type: 55% single flip, 30% double flip, 10% triple flip, 5% swap
        mutation_rand = jax.random.uniform(key1)
        
        n = len(current_mask)
        neighbor_mask = current_mask
        
        if mutation_rand < 0.55:
            # Single bit flip (focused exploitation)
            idx = jax.random.randint(key2, (), 1, n)
            neighbor_mask = current_mask.at[idx].set(1 - current_mask[idx])
        elif mutation_rand < 0.85:
            # Double bit flip (moderate exploration - increased)
            idx1 = jax.random.randint(key2, (), 1, n)
            idx2 = jax.random.randint(key3, (), 1, n)
            neighbor_mask = current_mask.at[idx1].set(1 - current_mask[idx1])
            neighbor_mask = neighbor_mask.at[idx2].set(1 - neighbor_mask[idx2])
        elif mutation_rand < 0.95:
            # Triple bit flip for bigger jumps
            idx1 = jax.random.randint(key2, (), 1, n)
            idx2 = jax.random.randint(key3, (), 1, n)
            idx3 = jax.random.randint(key4, (), 1, n)
            neighbor_mask = current_mask.at[idx1].set(1 - current_mask[idx1])
            neighbor_mask = neighbor_mask.at[idx2].set(1 - neighbor_mask[idx2])
            neighbor_mask = neighbor_mask.at[idx3].set(1 - neighbor_mask[idx3])
        else:
            # Swap operation to maintain set size while changing composition
            idx1 = jax.random.randint(key2, (), 1, n)
            idx2 = jax.random.randint(key3, (), 1, n)
            bit1 = current_mask[idx1]
            bit2 = current_mask[idx2]
            neighbor_mask = current_mask.at[idx1].set(bit2).at[idx2].set(bit1)

        neighbor_loss = self._objective_fn(neighbor_mask)
        delta_loss = neighbor_loss - current_loss

        # Metropolis acceptance criterion
        should_accept = False
        if delta_loss < 0:
            should_accept = True
        else:
            accept_prob = jnp.exp(-delta_loss / temp)
            if jax.random.uniform(key1) < accept_prob:
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
    # Enhanced initialization with hybrid pattern set - denser starting points
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Start with multiple complementary patterns known for good sum/difference ratios
    max_exp = int(np.floor(np.log2(hypers.max_integer))) if hypers.max_integer > 0 else 0
    
    u_mask = np.zeros(hypers.max_integer + 1, dtype=bool)
    u_mask[0] = True
    
    # Pattern 1: Power-of-two basis (excellent for large difference sets)
    for e in range(max_exp + 1):
        idx = 2**e
        if idx <= hypers.max_integer:
            u_mask[idx] = True
    
    # Pattern 2: Arithmetic progression with step 3 (balanced ratio)
    for i in range(0, hypers.max_integer + 1, 3):
        u_mask[i] = True
    
    # Pattern 3: Arithmetic progression with step 4 (another good structure)
    for i in range(0, hypers.max_integer + 1, 4):
        u_mask[i] = True
    
    # Pattern 4: Multiples of small primes and composites for denser structure
    for i in range(0, hypers.max_integer + 1, 6):
        u_mask[i] = True
    for i in range(0, hypers.max_integer + 1, 8):
        u_mask[i] = True
    
    # Convert to jax array for further operations
    u_mask = jnp.array(u_mask)
    
    # Increased random fill for better starting density - 18% instead of 12-15%
    random_fill = jax.random.bernoulli(subkey1, p=0.18, shape=(hypers.max_integer + 1,))
    u_mask = jnp.logical_or(u_mask, random_fill)
    
    # Ensure 0 and key values stay included - extended list of promising candidates
    key_values = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 80, 81, 90, 96, 100, 108, 120, 128, 135, 144, 150, 160, 162, 180, 192, 200, 216, 240, 243, 256, 270]
    for v in key_values:
        if v <= hypers.max_integer:
            u_mask = u_mask.at[v].set(True)

    searcher = C6Searcher(hypers)
    current_loss = searcher._objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    # Faster cooling in early phase, slower at end for exploitation
    cooling_factor = 0.995
    
    for step in tqdm.tqdm(range(hypers.num_search_steps), desc="Annealing Progress"):
        key, subkey = jax.random.split(key)
        # Exponential cooling with adaptive minimum
        current_temp = max(hypers.initial_temperature * (cooling_factor ** step), 5e-8)
        current_mask, current_loss = searcher.anneal_step(
            subkey, current_temp, current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set


# EVOLVE-BLOCK-END