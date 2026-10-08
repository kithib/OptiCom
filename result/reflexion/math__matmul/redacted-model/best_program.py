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

# Enable high precision for critical operations
jax.config.update("jax_enable_x64", True)


# --- Straight-Through Estimator for Rounding ---
@jax.custom_vjp
def round_to_half_ste(x):
    """Forward pass: snaps values to the nearest half-integer."""
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    """Standard forward pass and identity for backward pass."""
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    """Backward pass: Identity function, passes gradient straight through."""
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)
# --- End of STE definition ---


# --- Loss Functions ---
def weighted_l2_loss(reconstructed: jnp.ndarray, target: jnp.ndarray) -> jnp.ndarray:
    error = reconstructed - target
    weights = jnp.where(target != 0, 1000.0, 1.0)
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
    init_scale: float = 0.1
    l1_strength: float = 1e-6
    clamp_range: float = 3.0
    # Phase 2: Discrete Fine-tuning
    phase2_steps: int = 50000
    phase2_lr: float = 1e-4
    # Time budget (seconds)
    time_budget: float = 250.0
    # Phase 3: Local integer search
    int_perturb_steps: int = 500000


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
        """Applies a scaled tanh to map latent parameters to the desired range."""
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
            optax.clip_by_global_norm(1.0),
            optax.adam(hypers.phase2_lr),
        )

    def _loss_fn(self, continuous_decomposition: tuple) -> jnp.ndarray:
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        return l2_loss_real(reconstructed, self.target_tensor)


# --- JIT-compatible Train Step ---
def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, n * p), dtype=jnp.float64)
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1.0)
    return T


def run():
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)
    start_time = time.time()

    def check_timeout():
        if time.time() - start_time > hypers.time_budget:
            print("\n[TIMEOUT GUARD] Approaching time limit, finishing early.")
            return True
        return False

    # --- PHASE 1: CONTINUOUS EXPLORATION ---
    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    best_loss_phase1 = float("inf")
    best_latent_decomp = None

    continuous_optimizer = ContinuousOptimizer(target_tensor, hypers)

    jit_train_step_continuous = jax.jit(train_step, static_argnums=(2, 3))

    for i in range(hypers.num_restarts):
        if check_timeout():
            break
        print(f"\n--- Restart {i+1}/{hypers.num_restarts} (time: {time.time()-start_time:.1f}s) ---")
        main_key, restart_key = jax.random.split(main_key)
        keys = jax.random.split(restart_key, 3)
        init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale, dtype=jnp.float64)
        latent_decomp = (
            init_fn(keys[0], (n * m, hypers.rank)),
            init_fn(keys[1], (m * p, hypers.rank)),
            init_fn(keys[2], (n * p, hypers.rank)),
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        for _ in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp,
                opt_state,
                continuous_optimizer.opt,
                continuous_optimizer._loss_fn,
            )
            if loss < 1e-8:
                break
            if time.time() - start_time > hypers.time_budget * 0.35:
                break

        final_loss = l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *continuous_optimizer._get_constrained_decomposition(latent_decomp),
            ),
        )
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
            print(f"New best continuous loss: {best_loss_phase1:.10f}")
            if final_loss < 1e-8:
                print("Continuous solution is very close, moving to phase 2 early.")
                break

    if best_latent_decomp is None:
        # Fallback: initialize any decomposition to prevent later failures
        key = jax.random.PRNGKey(0)
        keys = jax.random.split(key, 3)
        init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale, dtype=jnp.float64)
        best_latent_decomp = (
            init_fn(keys[0], (n * m, hypers.rank)),
            init_fn(keys[1], (m * p, hypers.rank)),
            init_fn(keys[2], (n * p, hypers.rank)),
        )

    # --- PHASE 2: DISCRETE FINE-TUNING ---
    print(f"\n{'='*20} PHASE 2: Discrete Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")
    print(f"Time so far: {time.time()-start_time:.1f}s")

    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)

    discrete_optimizer = DiscreteOptimizer(target_tensor, hypers)
    opt_state = discrete_optimizer.opt.init(continuous_params)

    jit_train_step_discrete = jax.jit(train_step, static_argnums=(2, 3))

    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Discrete Fine-tuning"):
        continuous_params, opt_state, loss = jit_train_step_discrete(
            continuous_params, opt_state, discrete_optimizer.opt, discrete_optimizer._loss_fn
        )
        if (step + 1) % 5000 == 0:
            print(f"Step {step+1} | Discrete Loss: {loss:.10f}")
        if loss == 0.0:
            print("\nFound a perfect exact solution!")
            break
        if time.time() - start_time > hypers.time_budget * 0.7:
            print("\n[TIMEOUT GUARD] Phase 2 stopping to save time for refinement.")
            break

    def final_round(arr):
        return jnp.round(arr * 2) / 2.0

    final_discrete_decomposition = jax.tree_util.tree_map(final_round, continuous_params)
    final_reconstructed = jnp.einsum("ir,jr,kr->ijk", *final_discrete_decomposition)
    final_loss = l2_loss_real(target_tensor, final_reconstructed)
    print(f"After STE. Final discrete loss: {final_loss:.10f}")

    # --- PHASE 3: LOCAL INTEGER SEARCH ---
    if final_loss > 1e-12:
        print("Solution not exact, running local integer refinement...")
        print(f"Time so far: {time.time()-start_time:.1f}s")
        target_int = (target_tensor * 2).astype(jnp.int64)
        current_int_params = [
            (final_discrete_decomposition[0] * 2).astype(jnp.int64),
            (final_discrete_decomposition[1] * 2).astype(jnp.int64),
            (final_discrete_decomposition[2] * 2).astype(jnp.int64),
        ]
        ri = jnp.einsum("ir,jr,kr->ijk", *current_int_params)
        best_int_loss = jnp.sum((ri - target_int) ** 2).item()
        print(f"Initial integer loss: {best_int_loss}")

        perturb_key = jax.random.PRNGKey(12345)
        int_perturb = 0
        while int_perturb < hypers.int_perturb_steps:
            perturb_key, pkey = jax.random.split(perturb_key)
            fidx = int(jax.random.randint(pkey, (1,), 0, 3)[0])
            shp = current_int_params[fidx].shape
            rr = int(jax.random.randint(pkey, (1,), 0, shp[0])[0])
            cc = int(jax.random.randint(pkey, (1,), 0, shp[1])[0])
            d = int(jax.random.randint(pkey, (1,), -2, 2)[0])
            if d == 0:
                int_perturb += 1
                continue
            old = current_int_params[fidx][rr, cc]
            current_int_params[fidx] = current_int_params[fidx].at[rr, cc].set(old + d)
            ri = jnp.einsum("ir,jr,kr->ijk", *current_int_params)
            new_loss = jnp.sum((ri - target_int) ** 2).item()
            if new_loss < best_int_loss:
                best_int_loss = new_loss
                if best_int_loss == 0:
                    print(f"Found exact integer solution at step {int_perturb+1}")
                    break
            else:
                current_int_params[fidx] = current_int_params[fidx].at[rr, cc].set(old)
            int_perturb += 1
            if int_perturb % 20000 == 0:
                print(f"Refinement step {int_perturb} | int loss: {best_int_loss}")
            if time.time() - start_time > hypers.time_budget * 0.95:
                print("\n[TIMEOUT GUARD] Refinement stopping due to time constraints.")
                break

        exact_match = (best_int_loss == 0)
        print(f"Integer reconstruction exact match: {exact_match}")

        if exact_match:
            final_discrete_decomposition = (
                current_int_params[0].astype(jnp.float64) / 2.0,
                current_int_params[1].astype(jnp.float64) / 2.0,
                current_int_params[2].astype(jnp.float64) / 2.0,
            )
            final_loss = 0.0
            print("Exact decomposition confirmed via integer arithmetic.")
        else:
            # Try smaller targeted perturbations
            print("Attempting targeted smaller perturbations...")
            perturb_key = jax.random.PRNGKey(54321)
            for extra_perturb in range(200000):
                perturb_key, pkey = jax.random.split(perturb_key)
                fidx = int(jax.random.randint(pkey, (1,), 0, 3)[0])
                shp = current_int_params[fidx].shape
                rr = int(jax.random.randint(pkey, (1,), 0, shp[0])[0])
                cc = int(jax.random.randint(pkey, (1,), 0, shp[1])[0])
                old = current_int_params[fidx][rr, cc]
                # Try +/-1 changes only
                for delta in [-1, 1]:
                    current_int_params[fidx] = current_int_params[fidx].at[rr, cc].set(old + delta)
                    ri = jnp.einsum("ir,jr,kr->ijk", *current_int_params)
                    new_loss = jnp.sum((ri - target_int) ** 2).item()
                    if new_loss < best_int_loss:
                        best_int_loss = new_loss
                        if best_int_loss == 0:
                            print(f"Found exact integer solution at extra step {extra_perturb+1}")
                            break
                    else:
                        current_int_params[fidx] = current_int_params[fidx].at[rr, cc].set(old)
                if best_int_loss == 0:
                    break
                if time.time() - start_time > hypers.time_budget * 0.98:
                    print("\n[TIMEOUT GUARD] Extra refinement stopping due to time constraints.")
                    break

            exact_match = (best_int_loss == 0)
            print(f"Integer reconstruction exact match after extra steps: {exact_match}")

            if exact_match:
                final_discrete_decomposition = (
                    current_int_params[0].astype(jnp.float64) / 2.0,
                    current_int_params[1].astype(jnp.float64) / 2.0,
                    current_int_params[2].astype(jnp.float64) / 2.0,
                )
                final_loss = 0.0
                print("Exact decomposition confirmed via integer arithmetic.")
            else:
                final_discrete_decomposition = (
                    current_int_params[0].astype(jnp.float64) / 2.0,
                    current_int_params[1].astype(jnp.float64) / 2.0,
                    current_int_params[2].astype(jnp.float64) / 2.0,
                )
                final_reconstructed = jnp.einsum("ir,jr,kr->ijk", *final_discrete_decomposition)
                final_loss = l2_loss_real(target_tensor, final_reconstructed)
                print(f"Refined decomposition loss: {final_loss:.10f}")

    print(f"Total run time: {time.time()-start_time:.1f}s")
    final_decomposition_np = jax.tree_util.tree_map(np.array, final_discrete_decomposition)
    return final_decomposition_np, n, m, p, float(final_loss), hypers.rank


# EVOLVE-BLOCK-END