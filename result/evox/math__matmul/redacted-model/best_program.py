# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"
# Force x64 support to avoid dtype truncation drift
os.environ["JAX_ENABLE_X64"] = "1"

# Fixed parameters
n, m, p = 2, 4, 5

# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
import tqdm

jax.config.update("jax_enable_x64", True)
DTYPE = jnp.float64


# --- Straight-Through Estimator for Rounding ---
@jax.custom_vjp
def round_to_int_ste(x):
    """Forward pass: snaps values to nearest integer (allows 0, ±1, ±2...)"""
    return jnp.round(x)


def round_int_ste_fwd(x):
    return round_to_int_ste(x), None


def round_int_ste_bwd(res, g):
    return (g,)


round_to_int_ste.defvjp(round_int_ste_fwd, round_int_ste_bwd)
# --- End of STE definition ---


# --- Loss Functions ---
def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 1000.0, 1.0)
    return jnp.sum(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.sum((x - y) ** 2)


# --- Hyperparameters ---
@dataclass
class Hyperparameters:
    rank: int = 55
    # Phase 1: Continuous Search
    num_restarts: int = 12
    phase1_steps: int = 120000
    phase1_lr: float = 0.015
    init_scale: float = 0.3
    l1_strength: float = 1e-5
    clamp_range: float = 3.0
    # Phase 2: Exact Fine-tuning
    phase2_steps: int = 40000
    phase2_lr: float = 2e-5
    # Phase 3: Gradient-free rank reduction
    phase3_steps: int = 8000
    shrink_tolerance: float = 1e-8


# --- Target Tensor (in evaluator canonical form) ---
def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, p * n), dtype=DTYPE)
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1.0)
    return T


# --- Optimizer Classes ---
class ContinuousOptimizer:
    """Finds a high-quality approximate continuous solution."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(hypers.phase1_lr),
        )

    def _get_constrained_decomposition(self, latent_decomposition: tuple) -> tuple:
        return jax.tree_util.tree_map(
            lambda x: self.hypers.clamp_range * jnp.tanh(x), latent_decomposition
        )

    def _loss_fn(self, latent_decomposition: tuple) -> jnp.ndarray:
        constrained = self._get_constrained_decomposition(latent_decomposition)
        reconstructed = jnp.einsum(
            "ir,jr,kr->ijk", *constrained, precision=jax.lax.Precision.HIGH
        )
        recon_loss = weighted_l2_loss(reconstructed, self.target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + self.hypers.l1_strength * l1_penalty


class DiscreteOptimizer:
    """Refines a continuous solution into an exact discrete integer one using an STE."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(0.5),
            optax.adam(hypers.phase2_lr),
        )

    def _loss_fn(self, continuous_decomposition: tuple) -> jnp.ndarray:
        discrete_decomposition = jax.tree_util.tree_map(
            round_to_int_ste, continuous_decomposition
        )
        reconstructed = jnp.einsum(
            "ir,jr,kr->ijk",
            *discrete_decomposition,
            precision=jax.lax.Precision.HIGHEST,
        )
        return l2_loss_real(reconstructed, self.target_tensor)


# --- JIT-compatible Train Step ---
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def _recon_loss_int(target, u, v, w):
    recon = jnp.einsum(
        "ir,jr,kr->ijk", u, v, w, precision=jax.lax.Precision.HIGHEST
    )
    return l2_loss_real(target, recon)


def _exact_check(target_tensor, u, v, w):
    return float(_recon_loss_int(target_tensor, u, v, w)) < 1e-12


def run():
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
            init_fn(restart_key, (n * m, hypers.rank), dtype=DTYPE),
            init_fn(restart_key, (m * p, hypers.rank), dtype=DTYPE),
            init_fn(restart_key, (n * p, hypers.rank), dtype=DTYPE),
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        for step in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp,
                opt_state,
                continuous_optimizer.opt,
                continuous_optimizer._loss_fn,
            )
            if (step + 1) % 20000 == 0:
                print(f"  Step {step+1} | Loss: {loss:.6f}")

        final_loss = float(
            l2_loss_real(
                target_tensor,
                jnp.einsum(
                    "ir,jr,kr->ijk",
                    *continuous_optimizer._get_constrained_decomposition(latent_decomp),
                    precision=jax.lax.Precision.HIGH,
                ),
            )
        )
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
            if final_loss < 1e-6:
                print("Excellent approximate solution found, moving to fine-tuning.")
                break

    # --- PHASE 2: EXACT INTEGER FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: Exact Integer Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")

    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)
    continuous_params = jax.tree_util.tree_map(
        lambda x: jnp.asarray(x, dtype=DTYPE), continuous_params
    )

    discrete_optimizer = DiscreteOptimizer(target_tensor, hypers)
    opt_state = discrete_optimizer.opt.init(continuous_params)
    jit_train_step_discrete = jax.jit(train_step, static_argnums=(2, 3))

    found_exact = False
    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Discrete Fine-tuning"):
        continuous_params, opt_state, loss = jit_train_step_discrete(
            continuous_params, opt_state, discrete_optimizer.opt, discrete_optimizer._loss_fn
        )
        if (step + 1) % 2000 == 0:
            print(f"Step {step+1} | Discrete Loss: {loss:.8f}")
        if loss < 1e-12:
            print("\nFound an exact integer solution!")
            found_exact = True
            break

    # Round to exact 64-bit integers (x64 is enabled, so int64 is supported)
    final_discrete_decomposition = jax.tree_util.tree_map(
        lambda x: jnp.asarray(jnp.round(x), dtype=jnp.int64), continuous_params
    )

    # --- PHASE 3: RANK REDUCTION ---
    if found_exact:
        print(f"\n{'='*20} PHASE 3: Rank Reduction {'='*20}")
        u, v, w = final_discrete_decomposition
        current_rank = u.shape[1]
        print(f"Starting rank reduction from {current_rank} ...")
        main_key, shrink_key = jax.random.split(main_key)

        improved = True
        while improved:
            improved = False
            # Random column order for robust local search
            cols = jax.random.permutation(shrink_key, jnp.arange(current_rank))
            for c in cols:
                if current_rank <= 1:
                    break
                mask = jnp.ones(current_rank, dtype=bool).at[c].set(False)
                u2 = u[:, mask]
                v2 = v[:, mask]
                w2 = w[:, mask]
                if _exact_check(target_tensor, u2, v2, w2):
                    u, v, w = u2, v2, w2
                    current_rank -= 1
                    improved = True
                    print(f"  Reduced rank by 1 -> new rank {current_rank}")
                    main_key, shrink_key = jax.random.split(main_key)
                    break
        final_discrete_decomposition = (u, v, w)
        hypers.rank = current_rank

    recon = jnp.einsum(
        "ir,jr,kr->ijk",
        *final_discrete_decomposition,
        precision=jax.lax.Precision.HIGHEST,
    )
    max_error = float(jnp.max(jnp.abs(recon - target_tensor)))
    final_loss = float(l2_loss_real(target_tensor, recon))
    print(
        f"Search complete. Rank: {hypers.rank}, Max abs error: {max_error:.2e}, Loss: {final_loss:.8f}"
    )

    final_decomposition_np = jax.tree_util.tree_map(
        lambda x: np.asarray(np.rint(np.asarray(x, dtype=np.float64)), dtype=np.int64),
        final_discrete_decomposition,
    )
    return final_decomposition_np, n, m, p, float(final_loss), hypers.rank


# EVOLVE-BLOCK-END