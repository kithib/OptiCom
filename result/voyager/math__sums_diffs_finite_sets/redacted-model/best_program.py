import numpy as np
from numba import jit
from dataclasses import dataclass
import random
import math


@dataclass
class Hyperparameters:
    max_integer: int = 100
    num_restarts: int = 2
    num_search_steps: int = 150
    initial_temperature: float = 0.012


@jit(nopython=True)
def _objective_fn(u_mask):
    """Calculates the C6 lower bound - JIT-compiled for speed."""
    U = np.where(u_mask)[0]
    n = len(U)
    
    sums = np.empty(n * n, dtype=np.int32)
    diffs = np.empty(n * n, dtype=np.int32)
    
    idx = 0
    for i in range(n):
        for j in range(n):
            sums[idx] = U[i] + U[j]
            diffs[idx] = U[i] - U[j]
            idx += 1

    size_U_plus_U = len(np.unique(sums))
    size_U_minus_U = len(np.unique(diffs))
    max_U = U[-1]

    if max_U == 0:
        return -1.0

    ratio = size_U_minus_U / size_U_plus_U
    c6_bound = 1 + math.log(ratio) / math.log(2 * max_U + 1)

    return -c6_bound


def anneal_step(key, temp, current_mask, current_loss):
    num_flips = 1 if random.random() < 0.8 else 2
    neighbor_mask = current_mask.copy()
    
    for _ in range(num_flips):
        if random.random() < 0.85:
            idx_to_flip = random.randrange(1, min(len(current_mask), 60))
        else:
            idx_to_flip = random.randrange(max(1, len(current_mask) // 3), len(current_mask))
        neighbor_mask[idx_to_flip] = 1 - neighbor_mask[idx_to_flip]

    if neighbor_mask.sum() < 3:
        return current_mask, current_loss

    neighbor_loss = _objective_fn(neighbor_mask)
    delta_loss = neighbor_loss - current_loss

    should_accept = False
    if delta_loss < 0:
        should_accept = True
    else:
        accept_prob = math.exp(-delta_loss / temp)
        if random.random() < accept_prob:
            should_accept = True

    if should_accept:
        return neighbor_mask, neighbor_loss
    else:
        return current_mask, current_loss


def run():
    hypers = Hyperparameters()
    random.seed(42)

    best_loss = float("inf")
    best_set_np = None

    for i in range(hypers.num_restarts):
        print(f"\n{'='*20} Restart {i+1}/{hypers.num_restarts} {'='*20}")
        loss, u_set_np = run_single_trial(hypers, i)

        if loss < best_loss:
            print(f"New best C6 bound found: {-loss:.8f}")
            best_loss = loss
            best_set_np = u_set_np

    c6_bound = -best_loss
    print(f"\nSearch complete. Best C6 lower bound found: {c6_bound:.8f}")
    return best_set_np, c6_bound


def run_single_trial(hypers, seed):
    random.seed(seed + 100)
    sparsity = 0.925
    u_mask = np.array([random.random() < (1 - sparsity) for _ in range(hypers.max_integer + 1)], dtype=bool)
    u_mask[0] = True
    while u_mask.sum() < 4:
        u_mask[random.randrange(1, hypers.max_integer + 1)] = True

    current_loss = _objective_fn(u_mask)

    print(f"Starting SA search. Initial C6 bound: {-current_loss:.6f}")

    current_mask = u_mask
    for step in range(hypers.num_search_steps):
        current_temp = hypers.initial_temperature * (1 - step / hypers.num_search_steps)
        current_mask, current_loss = anneal_step(
            step, max(current_temp, 1e-6), current_mask, current_loss
        )

    final_set = np.where(current_mask)[0]
    return current_loss, final_set