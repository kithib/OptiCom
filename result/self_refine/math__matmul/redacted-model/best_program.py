# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"

# Fixed parameters
n, m, p = 2, 4, 5

# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
import tqdm
import time


# --- Straight-Through Estimator for Rounding ---
@jax.custom_vjp
def round_to_half_ste(x):
    """Forward pass: snaps values to the nearest half-integer."""
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    """Standard forward pass and identity for backward pass."""
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    """Backward pass: Identity function, passes gradient straight through with clipping."""
    return (jnp.clip(g, -1.0, 1.0),)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)
# --- End of STE definition ---


# --- Loss Functions ---
def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 100.0, 1.0)
    return jnp.mean(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


# --- Hyperparameters ---
@dataclass
class Hyperparameters:
    rank: int = 40
    # Phase 1: Continuous Search
    num_restarts: int = 20
    phase1_steps: int = 50000
    phase1_lr: float = 0.02
    init_scale: float = 0.2
    l1_strength: float = 2e-7
    clamp_range: float = 2.5
    # Phase 2: Discrete Fine-tuning
    phase2_steps: int = 40000
    phase2_lr: float = 3e-4
    # Gradient clipping
    max_grad_norm: float = 1.0
    # Rank pruning
    pruning_threshold: float = 1e-6


# --- Helper to prune zero columns from decomposition ---
def prune_zero_columns(decomposition, hypers):
    """Remove columns where all three matrices have near-zero values."""
    U, V, W = decomposition
    col_norms_U = jnp.sum(U ** 2, axis=0)
    col_norms_V = jnp.sum(V ** 2, axis=0)
    col_norms_W = jnp.sum(W ** 2, axis=0)
    combined_col_norms = col_norms_U + col_norms_V + col_norms_W
    keep_columns_mask = combined_col_norms > hypers.pruning_threshold
    U_pruned = U[:, keep_columns_mask]
    V_pruned = V[:, keep_columns_mask]
    W_pruned = W[:, keep_columns_mask]
    new_rank = U_pruned.shape[1]
    return (U_pruned, V_pruned, W_pruned), new_rank


# --- Optimizer Classes ---
class ContinuousOptimizer:
    """Finds a high-quality approximate continuous solution."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.adamw(hypers.phase1_lr, weight_decay=1e-5)
        )

    def _get_constrained_decomposition(self, latent_decomposition: tuple) -> tuple:
        """Applies a scaled tanh to map latent parameters to the desired range."""
        return jax.tree_util.tree_map(
            lambda x: self.hypers.clamp_range * jnp.tanh(x), latent_decomposition
        )

    def _loss_fn(self, latent_decomposition: tuple) -> jnp.ndarray:
        constrained = self._get_constrained_decomposition(latent_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained, optimize=True)
        recon_loss = weighted_l2_loss(reconstructed, self.target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + self.hypers.l1_strength * l1_penalty


class DiscreteOptimizer:
    """Refines a continuous solution into an exact discrete one using an STE."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.adamw(hypers.phase2_lr, weight_decay=1e-6)
        )

    def _loss_fn(self, continuous_decomposition: tuple) -> jnp.ndarray:
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition, optimize=True)
        return l2_loss_real(reconstructed, self.target_tensor)


# --- JIT-compatible Train Step ---
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, n * p))
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1)
    return T


def run():
    start_time = time.time()
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)

    # --- PHASE 1: CONTINUOUS EXPLORATION ---
    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    best_loss_phase1 = float("inf")
    best_latent_decomp = None

    continuous_optimizer = ContinuousOptimizer(target_tensor, hypers)
    jit_train_step_continuous = jax.jit(train_step, static_argnums=(2, 3))

    for i in range(hypers.num_restarts):
        print(f"\n--- Restart {i+1}/{hypers.num_restarts} ---")
        main_key, restart_key = jax.random.split(main_key)
        init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale)
        latent_decomp = (
            init_fn(restart_key, (n * m, hypers.rank)),
            init_fn(restart_key, (m * p, hypers.rank)),
            init_fn(restart_key, (n * p, hypers.rank)),
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        for _ in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp,
                opt_state,
                continuous_optimizer.opt,
                continuous_optimizer._loss_fn,
            )

        final_loss = l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *continuous_optimizer._get_constrained_decomposition(latent_decomp),
                optimize=True
            ),
        )
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
            print(f"** New best! **")

    # --- PHASE 1.5: PRUNE ZERO COLUMNS ---
    print(f"\n{'='*20} PHASE 1.5: Rank Pruning {'='*20}")
    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)
    pruned_continuous_params, effective_rank = prune_zero_columns(continuous_params, hypers)
    print(f"Pruned zero columns: rank reduced from {hypers.rank} → {effective_rank}")

    # --- PHASE 2: DISCRETE FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: Discrete Fine-tuning (STE) {'='*20}")
    print(f"Starting with pruned continuous solution (rank: {effective_rank}, loss: {best_loss_phase1:.8f})")

    discrete_optimizer = DiscreteOptimizer(target_tensor, hypers)
    opt_state = discrete_optimizer.opt.init(pruned_continuous_params)

    jit_train_step_discrete = jax.jit(train_step, static_argnums=(2, 3))

    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Discrete Fine-tuning"):
        pruned_continuous_params, opt_state, loss = jit_train_step_discrete(
            pruned_continuous_params, opt_state, discrete_optimizer.opt, discrete_optimizer._loss_fn
        )
        if (step + 1) % 5000 == 0:
            tqdm.tqdm.write(f"Step {step+1} | Discrete Loss: {loss:.10f}")
        if loss < 1e-12:
            tqdm.tqdm.write(f"\n*** Found EXACT solution at step {step+1}! ***")
            break

    final_discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, pruned_continuous_params)
    final_loss = l2_loss_real(
        target_tensor, jnp.einsum("ir,jr,kr->ijk", *final_discrete_decomposition, optimize=True)
    )
    final_rank = final_discrete_decomposition[0].shape[1]
    elapsed_time = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Search complete. Final discrete loss: {final_loss:.12f}")
    print(f"Total evaluation time: {elapsed_time:.2f}s")
    print(f"Rank achieved: {final_rank}")
    if final_loss < 1e-9:
        print(f"SUCCESS: Exact decomposition found!")
        combined_score = 32.0 / final_rank
        print(f"Combined score: {combined_score:.4f}")
    print(f"{'='*50}\n")

    final_decomposition_np = jax.tree_util.tree_map(np.array, final_discrete_decomposition)
    return final_decomposition_np, n, m, p, float(final_loss), final_rank


# EVOLVE-BLOCK-END