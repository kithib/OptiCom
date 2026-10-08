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
import time


# --- Straight-Through Estimator for Rounding ---
@jax.custom_vjp
def round_to_half_ste(x):
    """Forward pass: snaps values to the nearest half-integer."""
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)
# --- End of STE definition ---


# --- Precompute tensor shape and target for speed and stability ---
_SHAPE = (n * m, m * p, n * p)
_NP_TARGET = None


def _get_numpy_target():
    global _NP_TARGET
    if _NP_TARGET is None:
        T = np.zeros(_SHAPE, dtype=np.float64)
        for i, j, k in np.ndindex(n, m, p):
            T[i * m + j, j * p + k, k * n + i] = 1.0
        _NP_TARGET = T
    return _NP_TARGET


def reconstruction_error_numpy(uvw):
    """Use max-absolute-error as ground-truth validation metric (matching evaluator's test)."""
    T = _get_numpy_target()
    u, v, w = uvw
    recon = np.zeros_like(T)
    # Fused, memory-light reconstruction using small per-rank outer products
    for r in range(u.shape[1]):
        ucol = u[:, r].reshape(_SHAPE[0], 1, 1)
        vcol = v[:, r].reshape(1, _SHAPE[1], 1)
        wcol = w[:, r].reshape(1, 1, _SHAPE[2])
        recon += (ucol * vcol * wcol)
    return float(np.max(np.abs(recon - T)))


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
    num_restarts: int = 16
    phase1_steps: int = 80000
    phase1_lr: float = 0.012
    init_scale: float = 0.14
    l1_strength: float = 1.5e-7
    clamp_range: float = 4.0
    # Phase 2: Discrete Fine-tuning
    phase2_steps: int = 30000
    phase2_lr: float = 2e-4
    # Phase 3: Exact Grid Search
    grid_window: float = 0.5
    early_stop_loss: float = 1e-9
    # Global runtime budget (to stay under evaluator limits)
    runtime_budget_s: float = 250.0


# --- Optimizer Classes ---
class ContinuousOptimizer:
    """Finds a high-quality approximate continuous solution."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(4.5),
            optax.adamw(hypers.phase1_lr, weight_decay=1e-5),
        )

    def _get_constrained_decomposition(self, latent_decomposition: tuple) -> tuple:
        return jax.tree_util.tree_map(
            lambda x: self.hypers.clamp_range * jnp.tanh(x), latent_decomposition
        )

    def _loss_fn(self, latent_decomposition: tuple) -> jnp.ndarray:
        constrained = self._get_constrained_decomposition(latent_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        recon_loss = weighted_l2_loss(reconstructed, self.target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + self.hypers.l1_strength * l1_penalty


class DiscreteOptimizer:
    """Refines a continuous solution into an exact discrete one using an STE."""

    def __init__(self, target_tensor: jnp.ndarray, hypers: Hyperparameters):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(1.5),
            optax.adamw(hypers.phase2_lr, weight_decay=1e-6),
        )

    def _loss_fn(self, continuous_decomposition: tuple) -> jnp.ndarray:
        # Snap the continuous parameters to the discrete grid
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        # Compute the loss using only these exact half-integer values
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        return l2_loss_real(reconstructed, self.target_tensor)


# --- JIT-compatible Train Step ---
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def get_matrix_multiplication_tensor(n, m, p):
    T = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i, j, k in np.ndindex(n, m, p):
        T[i * m + j, j * p + k, k * n + i] = 1.0
    return jnp.asarray(T)


def run():
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)
    start_time = time.time()
    # Precompute target once
    _get_numpy_target()

    def _over_budget():
        return (time.time() - start_time) > hypers.runtime_budget_s

    # --- PHASE 1: CONTINUOUS EXPLORATION ---
    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    best_loss_phase1 = float("inf")
    best_latent_decomp = None

    continuous_optimizer = ContinuousOptimizer(target_tensor, hypers)

    # Properly close over loss_fn in the JIT to avoid stale closures
    def continuous_loss_fn(latent):
        return continuous_optimizer._loss_fn(latent)

    jit_train_step_continuous = jax.jit(
        lambda params, opt_state: train_step(params, opt_state, continuous_optimizer.opt, continuous_loss_fn)
    )

    for i in range(hypers.num_restarts):
        if _over_budget():
            print("Runtime budget reached, skipping additional restarts.")
            break

        print(f"\n--- Restart {i+1}/{hypers.num_restarts} ---")
        main_key, restart_key = jax.random.split(main_key)
        # Use unique keys per matrix to avoid correlated initialization
        keys = jax.random.split(restart_key, 3)
        latent_decomp = (
            jax.random.normal(keys[0], (n * m, hypers.rank)) * hypers.init_scale,
            jax.random.normal(keys[1], (m * p, hypers.rank)) * hypers.init_scale,
            jax.random.normal(keys[2], (n * p, hypers.rank)) * hypers.init_scale,
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        phase_start = time.time()
        for step in range(hypers.phase1_steps):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp, opt_state
            )
            if (step + 1) % 20000 == 0:
                print(f"Step {step+1} | Loss: {loss:.8f}")
            if step % 10000 == 9999 and _over_budget():
                print(f"Phase 1 trial cut short at step {step+1} for runtime budget.")
                break

        phase_elapsed = time.time() - phase_start

        constrained = continuous_optimizer._get_constrained_decomposition(latent_decomp)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        final_loss = float(l2_loss_real(target_tensor, reconstructed))
        print(f"End of Trial | Final continuous loss: {final_loss:.8f} | Time: {phase_elapsed:.1f}s")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
            if final_loss < 2e-5:
                print("High-quality continuous solution found, moving on to discrete phase early.")
                break

    # Fallback guard for best_latent_decomp
    if best_latent_decomp is None:
        keys = jax.random.split(main_key, 3)
        best_latent_decomp = (
            jax.random.normal(keys[0], (n * m, hypers.rank)) * hypers.init_scale,
            jax.random.normal(keys[1], (m * p, hypers.rank)) * hypers.init_scale,
            jax.random.normal(keys[2], (n * p, hypers.rank)) * hypers.init_scale,
        )

    # --- PHASE 2: DISCRETE FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: Discrete Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")

    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)

    discrete_optimizer = DiscreteOptimizer(target_tensor, hypers)

    def discrete_loss_fn(params):
        return discrete_optimizer._loss_fn(params)

    jit_train_step_discrete = jax.jit(
        lambda params, opt_state: train_step(params, opt_state, discrete_optimizer.opt, discrete_loss_fn)
    )

    opt_state = discrete_optimizer.opt.init(continuous_params)

    best_discrete_loss = float("inf")
    best_discrete_params = jax.tree_util.tree_map(lambda x: x.copy(), continuous_params)

    for step in range(hypers.phase2_steps):
        continuous_params, opt_state, loss = jit_train_step_discrete(
            continuous_params, opt_state
        )
        if loss < best_discrete_loss:
            best_discrete_loss = float(loss)
            best_discrete_params = jax.tree_util.tree_map(lambda x: x.copy(), continuous_params)
        if (step + 1) % 5000 == 0:
            print(f"Step {step+1} | Discrete Loss: {loss:.8f} | Best: {best_discrete_loss:.8f}")
        if best_discrete_loss < hypers.early_stop_loss:
            print("\nFound near-exact solution during STE phase!")
            break
        if _over_budget():
            print("Runtime budget reached, cutting discrete fine-tuning short.")
            break

    # Use the best-discrete snapshot, not the final-step params
    continuous_params = best_discrete_params
    final_discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_params)
    final_loss = float(l2_loss_real(
        target_tensor, jnp.einsum("ir,jr,kr->ijk", *final_discrete_decomposition)
    ))
    print(f"Phase 2 complete. Best discrete loss: {final_loss:.8f}")

    # --- PHASE 3: EXACT GRID SEARCH CORRECTION ---
    print(f"\n{'='*20} PHASE 3: Exact Grid Search {'='*20}")
    print("Adjusting decomposition to exact half-integer values...")

    base_np = jax.tree_util.tree_map(lambda x: np.asarray(x, dtype=np.float64), final_discrete_decomposition)
    # Enforce half-integer values up front to eliminate off-grid rounding issues
    base_np = [np.round(arr * 2.0, 0) / 2.0 for arr in base_np]

    best_np = [b.copy() for b in base_np]
    best_grid_loss = reconstruction_error_numpy(best_np)
    print(f"Starting max-error: {best_grid_loss:.8f}")

    grid_offsets = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=np.float64)
    # Randomize rank exploration order with fixed seed for reproducibility
    rng = np.random.default_rng(12345)
    test_order = rng.permutation(hypers.rank)

    for r in test_order:
        if best_grid_loss == 0.0:
            break
        if _over_budget():
            print("Runtime budget reached during grid search.")
            break

        base_u_col = best_np[0][:, r].copy()
        base_v_col = best_np[1][:, r].copy()
        base_w_col = best_np[2][:, r].copy()

        for offset_u in grid_offsets:
            if best_grid_loss == 0.0:
                break
            trial_u_col = np.round((base_u_col + offset_u) * 2.0, 0) / 2.0
            for offset_v in grid_offsets:
                if best_grid_loss == 0.0:
                    break
                trial_v_col = np.round((base_v_col + offset_v) * 2.0, 0) / 2.0
                for offset_w in grid_offsets:
                    if offset_u == 0.0 and offset_v == 0.0 and offset_w == 0.0:
                        continue
                    trial = [best_np[0].copy(), best_np[1].copy(), best_np[2].copy()]
                    trial[0][:, r] = trial_u_col
                    trial[1][:, r] = trial_v_col
                    trial[2][:, r] = np.round((base_w_col + offset_w) * 2.0, 0) / 2.0
                    trial_loss = reconstruction_error_numpy(trial)
                    if trial_loss < best_grid_loss:
                        best_grid_loss = trial_loss
                        best_np = trial
                        print(f"  Rank {r:3d} | New max-error: {best_grid_loss:.8f}")
                        if best_grid_loss == 0.0:
                            break

    # Attempt multi-rank swap refinement
    if best_grid_loss != 0.0 and not _over_budget():
        print("Attempting rank-swap refinement...")
        rng2 = np.random.default_rng(98765)
        for _ in range(3000):
            if best_grid_loss == 0.0 or _over_budget():
                break
            r1, r2 = rng2.integers(0, hypers.rank, size=2)
            if r1 == r2:
                continue
            trial = [b.copy() for b in best_np]
            for k in range(3):
                trial[k][:, [r1, r2]] = trial[k][:, [r2, r1]]
            trial_loss = reconstruction_error_numpy(trial)
            if trial_loss < best_grid_loss:
                best_grid_loss = trial_loss
                best_np = trial
                print(f"  Swap {r1}<->{r2} | New max-error: {best_grid_loss:.8f}")

    # Attempt sign flips as final refinement step
    if best_grid_loss != 0.0 and not _over_budget():
        print("Attempting sign-flip refinement...")
        rng3 = np.random.default_rng(4242)
        for _ in range(4000):
            if best_grid_loss == 0.0 or _over_budget():
                break
            r = rng3.integers(0, hypers.rank)
            factor_idx = rng3.integers(0, 3)
            trial = [b.copy() for b in best_np]
            trial[factor_idx][:, r] = -trial[factor_idx][:, r]
            trial_loss = reconstruction_error_numpy(trial)
            if trial_loss < best_grid_loss:
                best_grid_loss = trial_loss
                best_np = trial
                print(f"  Sign-flip F{factor_idx}R{r} | New max-error: {best_grid_loss:.8f}")

    # Final rounding hygiene to guarantee half-integer output
    for i in range(3):
        best_np[i] = np.round(best_np[i] * 2.0, 0) / 2.0

    final_tensor_error = reconstruction_error_numpy(best_np)
    # Convert final loss to an MSE-style number for downstream metrics while keeping tensor exact
    final_loss_mse = float(np.mean((np.einsum("ir,jr,kr->ijk", *best_np) - _get_numpy_target()) ** 2))

    elapsed = time.time() - start_time
    print(f"\n{'='*50}")
    print(f"Final tensor max-error: {final_tensor_error:.15f}")
    print(f"Final MSE-style loss: {final_loss_mse:.15f}")
    print(f"Rank achieved: {hypers.rank}")
    print(f"Total time: {elapsed:.2f} seconds")
    print(f"Combined score (32/R): {32.0 / hypers.rank:.6f}")
    print(f"{'='*50}")

    final_decomposition_np = [np.array(arr, dtype=np.float64) for arr in best_np]
    return final_decomposition_np, n, m, p, float(final_loss_mse), hypers.rank


# EVOLVE-BLOCK-END

if __name__ == "__main__":
    run()