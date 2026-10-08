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
    weights = jnp.where(target != 0, 100.0, 1.0)
    return jnp.mean(weights * (error**2))


def l2_loss_real(x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    return jnp.mean((x - y) ** 2)


# --- Hyperparameters ---
@dataclass
class Hyperparameters:
    rank: int = 40
    # Phase 1: Continuous Search
    num_restarts: int = 15
    phase1_steps: int = 100000
    phase1_lr: float = 0.01
    init_scale: float = 0.1
    l1_strength: float = 1e-6
    clamp_range: float = 4.0
    # Phase 2: Discrete Fine-tuning
    phase2_steps: int = 30000
    phase2_lr: float = 1e-4
    # Grid definition for validation
    grid_min: float = -4.0
    grid_max: float = 4.0
    grid_step: float = 0.5


# --- Core Helper Functions ---
def build_decomposition(U, V, W):
    """Reconstruct tensor from U, V, W factors using einsum."""
    return jnp.einsum("ir,jr,kr->ijk", U, V, W)


def build_int_decomposition(U, V, W):
    """Scale half-integer factors by 2 to get integer factors (exact arithmetic)."""
    return (
        jnp.asarray(jnp.round(U * 2), dtype=jnp.int32),
        jnp.asarray(jnp.round(V * 2), dtype=jnp.int32),
        jnp.asarray(jnp.round(W * 2), dtype=jnp.int32),
    )


def get_matrix_multiplication_tensor(n, m, p):
    """Build exact ground-truth matrix multiplication tensor as float32."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.float32)
    for i, j, k in np.ndindex(n, m, p):
        T[i * m + j, j * p + k, k * n + i] = 1.0
    return jnp.asarray(T)


def get_int_target_tensor(n, m, p):
    """Build exact ground-truth matrix multiplication tensor as int8 (for scaled validation)."""
    T = np.zeros((n * m, m * p, n * p), dtype=np.int64)
    for i, j, k in np.ndindex(n, m, p):
        T[i * m + j, j * p + k, k * n + i] = 8  # = 2^3 since 3 factors scaled by 2 each
    return T


def check_exact_decomposition(U_int, V_int, W_int, target_int):
    """Verify integer-scaled decomposition exactly matches integer target (no floating point)."""
    reconstructed_int = np.einsum("ir,jr,kr->ijk",
                                  np.asarray(U_int, dtype=np.int64),
                                  np.asarray(V_int, dtype=np.int64),
                                  np.asarray(W_int, dtype=np.int64))
    diff = np.max(np.abs(reconstructed_int - np.asarray(target_int, dtype=np.int64)))
    return (diff == 0), diff


# --- Optimizer Logic ---
def phase1_loss_fn(latent_decomposition, target_tensor, hypers):
    """Phase 1 loss: continuous weighted L2 on clamped parameters + L1 regularization."""
    Uc, Vc, Wc = [hypers.clamp_range * jnp.tanh(x) for x in latent_decomposition]
    reconstructed = jnp.einsum("ir,jr,kr->ijk", Uc, Vc, Wc)
    error = reconstructed - target_tensor
    weights = jnp.where(target_tensor != 0, 100.0, 1.0)
    recon_loss = jnp.mean(weights * (error**2))
    l1_penalty = sum(jnp.mean(jnp.abs(x)) for x in (Uc, Vc, Wc))
    return recon_loss + hypers.l1_strength * l1_penalty


def phase2_loss_fn(continuous_params, target_tensor):
    """Phase 2 loss: L2 on STE-rounded half-integer parameters."""
    Ud, Vd, Wd = [round_to_half_ste(x) for x in continuous_params]
    reconstructed = jnp.einsum("ir,jr,kr->ijk", Ud, Vd, Wd)
    return jnp.mean((reconstructed - target_tensor) ** 2)


@jax.jit
def continuous_train_step(params, opt_state, target_tensor, phase1_lr, l1_strength, clamp_range):
    """JIT-compatible Phase 1 step: no function objects passed as arguments."""
    hypers_snapshot = Hyperparameters(phase1_lr=phase1_lr, l1_strength=l1_strength, clamp_range=clamp_range)
    loss, grads = jax.value_and_grad(phase1_loss_fn)(params, target_tensor, hypers_snapshot)
    opt = optax.adam(phase1_lr)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


@jax.jit
def discrete_train_step(params, opt_state, target_tensor, phase2_lr):
    """JIT-compatible Phase 2 step: STE-based discrete fine-tuning."""
    loss, grads = jax.value_and_grad(phase2_loss_fn)(params, target_tensor)
    opt = optax.adam(phase2_lr)
    updates, opt_state = opt.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def run():
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    target_int8 = get_int_target_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)

    # --- PHASE 1: CONTINUOUS EXPLORATION ACROSS RESTARTS ---
    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    best_loss_phase1 = float("inf")
    best_continuous_params = None
    init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale)

    for i in range(hypers.num_restarts):
        print(f"\n--- Restart {i+1}/{hypers.num_restarts} (rank={hypers.rank}) ---")
        main_key, restart_key = jax.random.split(main_key)
        latent_decomp = (
            init_fn(restart_key, (n * m, hypers.rank)),
            init_fn(restart_key, (m * p, hypers.rank)),
            init_fn(restart_key, (n * p, hypers.rank)),
        )
        opt = optax.adam(hypers.phase1_lr)
        opt_state = opt.init(latent_decomp)

        for _ in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = continuous_train_step(
                latent_decomp,
                opt_state,
                target_tensor,
                hypers.phase1_lr,
                hypers.l1_strength,
                hypers.clamp_range,
            )

        # Evaluate final clamped parameters
        Uc, Vc, Wc = [hypers.clamp_range * np.tanh(np.asarray(x)) for x in latent_decomp]
        final_loss = float(l2_loss_real(target_tensor, jnp.einsum("ir,jr,kr->ijk", Uc, Vc, Wc)))
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_continuous_params = (jnp.asarray(Uc), jnp.asarray(Vc), jnp.asarray(Wc))
            print(f"  * New best continuous solution (loss={final_loss:.8f})")

    # --- PHASE 2: DISCRETE FINE-TUNING WITH STE ---
    print(f"\n{'='*20} PHASE 2: Discrete Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")
    continuous_params = [jnp.asarray(p) for p in best_continuous_params]
    opt = optax.adam(hypers.phase2_lr)
    opt_state = opt.init(continuous_params)

    found_exact = False
    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Discrete Fine-tuning"):
        continuous_params, opt_state, loss = discrete_train_step(
            continuous_params, opt_state, target_tensor, hypers.phase2_lr
        )
        if (step + 1) % 2000 == 0:
            print(f"Step {step+1} | Discrete Loss: {loss:.8f}")

        if loss < 1e-7:
            # Verify exact integer match
            U, V, W = [round_to_half_ste(p) for p in continuous_params]
            U2, V2, W2 = build_int_decomposition(U, V, W)
            matches, _ = check_exact_decomposition(U2, V2, W2, target_int8)
            if bool(matches):
                found_exact = True
                print("\nFound exact solution!")
                break

    # --- FINAL EXACT VALIDATION ---
    if not found_exact:
        U, V, W = [round_to_half_ste(p) for p in continuous_params]
        U2, V2, W2 = build_int_decomposition(U, V, W)
        matches, _ = check_exact_decomposition(U2, V2, W2, target_int8)
        if bool(matches):
            found_exact = True
            print("Hard-rounding yields exact match!")

    # --- OUTPUT FORMATTING ---
    if found_exact:
        U_out = np.asarray(np.asarray(U2, dtype=np.float32) / 2.0, dtype=np.float32)
        V_out = np.asarray(np.asarray(V2, dtype=np.float32) / 2.0, dtype=np.float32)
        W_out = np.asarray(np.asarray(W2, dtype=np.float32) / 2.0, dtype=np.float32)
        final_decomposition_np = (U_out, V_out, W_out)
        final_loss = 0.0
    else:
        U, V, W = [round_to_half_ste(p) for p in continuous_params]
        final_decomposition_np = (
            np.asarray(U, dtype=np.float32),
            np.asarray(V, dtype=np.float32),
            np.asarray(W, dtype=np.float32),
        )
        final_loss = float(l2_loss_real(
            target_tensor,
            jnp.einsum("ir,jr,kr->ijk", *final_decomposition_np),
        ))
        print(f"Warning: No exact decomposition found. Final discrete loss: {final_loss:.8f}")

    print(f"Search complete. Final loss: {final_loss:.8f} | Rank: {hypers.rank}")
    return final_decomposition_np, n, m, p, float(final_loss), hypers.rank


# EVOLVE-BLOCK-END