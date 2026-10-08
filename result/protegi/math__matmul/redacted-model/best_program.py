import os
os.environ["JAX_ENABLE_X64"] = "1"
os.environ["TQDM_DISABLE"] = "1"

n, m, p = 2, 4, 5

# EVOLVE-BLOCK-START
import numpy as np
import jax
import jax.numpy as jnp
import optax
from dataclasses import dataclass
import time
import tqdm

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platform_name", "cpu")


@jax.custom_vjp
def round_to_half_ste(x):
    return jnp.round(x * 2) / 2


def round_ste_fwd(x):
    return round_to_half_ste(x), None


def round_ste_bwd(res, g):
    return (g,)


round_to_half_ste.defvjp(round_ste_fwd, round_ste_bwd)


@jax.custom_vjp
def round_integer_ste(x):
    return jnp.round(x)


def round_int_fwd(x):
    return round_integer_ste(x), None


def round_int_bwd(res, g):
    return (g,)


round_integer_ste.defvjp(round_int_fwd, round_int_bwd)


def weighted_l2_loss(reconstructed, target):
    error = reconstructed - target
    weights = jnp.where(target != 0, 200.0, 1.0)
    return jnp.mean(weights * (error ** 2))


def l2_loss_real(x, y):
    return jnp.mean((x - y) ** 2)


def max_abs_error(x, y):
    return jnp.max(jnp.abs(x - y))


@dataclass
class Hyperparameters:
    rank: int = 40
    num_restarts: int = 15
    phase1_steps: int = 90000
    phase1_lr: float = 0.01
    init_scale: float = 0.08
    l1_strength: float = 1e-6
    clamp_range: float = 4.0
    phase2_steps: int = 40000
    phase2_lr: float = 1e-4
    phase3_steps: int = 30000
    phase3_lr: float = 5e-6
    success_threshold: float = 1e-6
    validation_interval: int = 2000
    max_grad_norm: float = 1.0


class ContinuousOptimizer:
    def __init__(self, target_tensor, hypers):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.adam(hypers.phase1_lr),
        )

    def _get_constrained_decomposition(self, latent_decomposition):
        return jax.tree_util.tree_map(
            lambda x: self.hypers.clamp_range * jnp.tanh(x), latent_decomposition
        )

    def _loss_fn(self, latent_decomposition):
        constrained = self._get_constrained_decomposition(latent_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *constrained)
        recon_loss = weighted_l2_loss(reconstructed, self.target_tensor)
        l1_penalty = sum(jnp.mean(jnp.abs(arr)) for arr in constrained)
        return recon_loss + self.hypers.l1_strength * l1_penalty


class HalfIntegerOptimizer:
    def __init__(self, target_tensor, hypers):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.adam(hypers.phase2_lr),
        )

    def _loss_fn(self, continuous_decomposition):
        discrete_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_decomposition)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *discrete_decomposition)
        diff = reconstructed - self.target_tensor
        return jnp.mean(jnp.where(jnp.abs(diff) > 0.25, diff**2 * 100.0, diff**2))


class IntegerCorrector:
    def __init__(self, target_tensor, hypers):
        self.target_tensor = target_tensor
        self.hypers = hypers
        self.opt = optax.chain(
            optax.clip_by_global_norm(hypers.max_grad_norm),
            optax.adam(hypers.phase3_lr),
        )

    def _loss_fn(self, residuals, base_halfint):
        adjusted = jax.tree_util.tree_map(
            lambda base, res: base + 0.49 * jnp.tanh(res), base_halfint, residuals
        )
        integer_decomposition = jax.tree_util.tree_map(round_integer_ste, adjusted)
        reconstructed = jnp.einsum("ir,jr,kr->ijk", *integer_decomposition)
        diff = reconstructed - self.target_tensor
        return jnp.mean(jnp.where(jnp.abs(diff) > 0.25, diff**2 * 100.0, diff**2))


def train_step(params, opt_state, optimizer, loss_fn):
    loss, grads = jax.value_and_grad(loss_fn)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def train_step_phase3(params, opt_state, optimizer, loss_fn, base_halfint):
    loss, grads = jax.value_and_grad(loss_fn)(params, base_halfint)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss


def get_matrix_multiplication_tensor(n, m, p):
    T = jnp.zeros((n * m, m * p, n * p), dtype=jnp.float64)
    for i, j, k in np.ndindex(n, m, p):
        T = T.at[i * m + j, j * p + k, k * n + i].set(1.0)
    return T


def run():
    overall_start = time.time()
    hypers = Hyperparameters()
    target_tensor = get_matrix_multiplication_tensor(n, m, p)
    main_key = jax.random.PRNGKey(42)
    np.random.seed(42)

    print(f"\n{'='*20} PHASE 1: Continuous Exploration {'='*20}")
    print(f"Target rank R = {hypers.rank}")
    best_loss_phase1 = float("inf")
    best_latent_decomp = None

    continuous_optimizer = ContinuousOptimizer(target_tensor, hypers)
    jit_train_step_continuous = jax.jit(train_step, static_argnums=(2, 3))

    for i in range(hypers.num_restarts):
        print(f"\n--- Restart {i+1}/{hypers.num_restarts} ---")
        main_key, restart_key = jax.random.split(main_key)
        init_fn = jax.nn.initializers.normal(stddev=hypers.init_scale, dtype=jnp.float64)
        latent_decomp = (
            init_fn(restart_key, (n * m, hypers.rank)),
            init_fn(restart_key, (m * p, hypers.rank)),
            init_fn(restart_key, (n * p, hypers.rank)),
        )
        opt_state = continuous_optimizer.opt.init(latent_decomp)

        for _ in tqdm.tqdm(range(hypers.phase1_steps), desc="Continuous Search"):
            latent_decomp, opt_state, loss = jit_train_step_continuous(
                latent_decomp, opt_state, continuous_optimizer.opt, continuous_optimizer._loss_fn
            )

        final_loss = float(l2_loss_real(
            target_tensor,
            jnp.einsum(
                "ir,jr,kr->ijk",
                *continuous_optimizer._get_constrained_decomposition(latent_decomp),
            ),
        ))
        print(f"End of Trial | Final continuous loss: {final_loss:.8f}")

        if final_loss < best_loss_phase1:
            best_loss_phase1 = final_loss
            best_latent_decomp = latent_decomp
            if final_loss < 1e-7:
                print("Excellent continuous solution - moving on.")
                break

    print(f"\n{'='*20} PHASE 2: Half-integer Fine-tuning (STE) {'='*20}")
    print(f"Starting with best continuous solution (loss: {best_loss_phase1:.8f})")

    continuous_params = continuous_optimizer._get_constrained_decomposition(best_latent_decomp)
    halfint_optimizer = HalfIntegerOptimizer(target_tensor, hypers)
    opt_state = halfint_optimizer.opt.init(continuous_params)
    jit_train_step_halfint = jax.jit(train_step, static_argnums=(2, 3))

    best_discrete_loss = float("inf")
    best_discrete_params = None

    for step in tqdm.tqdm(range(hypers.phase2_steps), desc="Half-int Fine-tuning"):
        continuous_params, opt_state, loss = jit_train_step_halfint(
            continuous_params, opt_state, halfint_optimizer.opt, halfint_optimizer._loss_fn
        )
        if (step + 1) % hypers.validation_interval == 0:
            d = jax.tree_util.tree_map(round_to_half_ste, continuous_params)
            recon = jnp.einsum("ir,jr,kr->ijk", *d)
            exact_loss = float(l2_loss_real(target_tensor, recon))
            print(f"Step {step+1} | Exact discrete loss: {exact_loss:.10f}")
            if exact_loss < best_discrete_loss:
                best_discrete_loss = exact_loss
                best_discrete_params = jax.tree_util.tree_map(
                    lambda x: jnp.array(x, dtype=jnp.float64), continuous_params
                )
            if exact_loss < 1e-9:
                print("\nFound a near-perfect half-integer solution!")
                break

    if best_discrete_params is not None:
        continuous_params = best_discrete_params

    halfint_decomposition = jax.tree_util.tree_map(round_to_half_ste, continuous_params)
    halfint_loss = float(l2_loss_real(
        target_tensor, jnp.einsum("ir,jr,kr->ijk", *halfint_decomposition)
    ))
    print(f"Best half-integer reconstruction loss: {halfint_loss:.12f}")

    print(f"\n{'='*20} PHASE 3: Integer Correction (STE) {'='*20}")

    residuals = (
        jnp.zeros_like(halfint_decomposition[0], dtype=jnp.float64),
        jnp.zeros_like(halfint_decomposition[1], dtype=jnp.float64),
        jnp.zeros_like(halfint_decomposition[2], dtype=jnp.float64),
    )
    integer_corrector = IntegerCorrector(target_tensor, hypers)
    opt_state = integer_corrector.opt.init(residuals)
    jit_train_step_phase3 = jax.jit(train_step_phase3, static_argnums=(2, 3))

    for step in tqdm.tqdm(range(hypers.phase3_steps), desc="Integer Correction"):
        residuals, opt_state, loss = jit_train_step_phase3(
            residuals, opt_state, integer_corrector.opt, integer_corrector._loss_fn, halfint_decomposition
        )
        if (step + 1) % 2000 == 0:
            adjusted = jax.tree_util.tree_map(
                lambda base, res: base + 0.49 * jnp.tanh(res), halfint_decomposition, residuals
            )
            rounded = jax.tree_util.tree_map(round_integer_ste, adjusted)
            recon = jnp.einsum("ir,jr,kr->ijk", *rounded)
            err = float(max_abs_error(target_tensor, recon))
            print(f"Step {step+1} | Max-err: {err:.10f}")
        if loss < 1e-10:
            print("\nFound a perfect integer solution!")
            break

    adjusted = jax.tree_util.tree_map(
        lambda base, res: base + 0.49 * jnp.tanh(res), halfint_decomposition, residuals
    )
    U_adj, V_adj, W_adj = [np.asarray(x, dtype=np.float64) for x in adjusted]
    U_np = np.rint(U_adj).astype(np.int64)
    V_np = np.rint(V_adj).astype(np.int64)
    W_np = np.rint(W_adj).astype(np.int64)

    reconstructed_int = np.zeros((n * m, m * p, n * p), dtype=np.int64)
    for r in range(hypers.rank):
        reconstructed_int += (
            U_np[:, r, None, None].astype(np.int64) *
            V_np[None, :, None, r].astype(np.int64) *
            W_np[None, None, :, r].astype(np.int64)
        )
    target_np = np.zeros_like(reconstructed_int)
    for i, j, k in np.ndindex(n, m, p):
        target_np[i * m + j, j * p + k, k * n + i] = 1

    final_loss = float(np.mean(
        (reconstructed_int.astype(np.float64) - target_np.astype(np.float64)) ** 2
    ))
    max_diff = int(np.max(np.abs(reconstructed_int - target_np)))

    if max_diff != 0:
        print(f"\nInteger decomposition not exact (max_diff={max_diff}). Attempting local repair...")
        current_u, current_v, current_w = U_np.copy(), V_np.copy(), W_np.copy()
        best_err = max_diff
        np.random.seed(123)
        for repair_iter in range(5000):
            improved = False
            candidates = []
            for _ in range(50):
                arr_choice = np.random.choice(["U", "V", "W"])
                if arr_choice == "U":
                    shape = current_u.shape
                    idx = np.random.randint(0, shape[0])
                    jdx = np.random.randint(0, shape[1])
                    delta = np.random.choice([-3, -2, -1, 1, 2, 3])
                    new_u = current_u.copy(); new_u[idx, jdx] += delta
                    u_t, v_t, w_t = new_u, current_v, current_w
                elif arr_choice == "V":
                    shape = current_v.shape
                    idx = np.random.randint(0, shape[0])
                    jdx = np.random.randint(0, shape[1])
                    delta = np.random.choice([-3, -2, -1, 1, 2, 3])
                    new_v = current_v.copy(); new_v[idx, jdx] += delta
                    u_t, v_t, w_t = current_u, new_v, current_w
                else:
                    shape = current_w.shape
                    idx = np.random.randint(0, shape[0])
                    jdx = np.random.randint(0, shape[1])
                    delta = np.random.choice([-3, -2, -1, 1, 2, 3])
                    new_w = current_w.copy(); new_w[idx, jdx] += delta
                    u_t, v_t, w_t = current_u, current_v, new_w

                rec_test = np.zeros_like(reconstructed_int)
                for r in range(hypers.rank):
                    rec_test += (
                        u_t[:, r, None, None].astype(np.int64) *
                        v_t[None, :, None, r].astype(np.int64) *
                        w_t[None, None, :, r].astype(np.int64)
                    )
                err = int(np.max(np.abs(rec_test - target_np)))
                if err < best_err:
                    candidates.append((arr_choice, idx, jdx, delta, err))
            if candidates:
                best_choice = min(candidates, key=lambda c: c[4])
                arr_choice, idx, jdx, delta, best_err = best_choice
                if arr_choice == "U":
                    current_u[idx, jdx] += delta
                elif arr_choice == "V":
                    current_v[idx, jdx] += delta
                else:
                    current_w[idx, jdx] += delta
                improved = True
                if best_err == 0:
                    print(f"  Repair succeeded on iteration {repair_iter+1}!")
                    break
            if not improved:
                break
        U_np, V_np, W_np = current_u, current_v, current_w
        reconstructed_int = np.zeros_like(target_np)
        for r in range(hypers.rank):
            reconstructed_int += (
                U_np[:, r, None, None].astype(np.int64) *
                V_np[None, :, None, r].astype(np.int64) *
                W_np[None, None, :, r].astype(np.int64)
            )
        final_loss = float(np.mean(
            (reconstructed_int.astype(np.float64) - target_np.astype(np.float64)) ** 2
        ))
        max_diff = int(np.max(np.abs(reconstructed_int - target_np)))

    eval_time = time.time() - overall_start

    print(f"\n{'='*50} SEARCH COMPLETE {'='*50}")
    print(f"Rank R = {hypers.rank}")
    print(f"Final integer reconstruction loss (L2): {final_loss:.12e}")
    print(f"Max element-wise difference vs target: {max_diff}")
    print(f"Combined Score (32/R): {32 / hypers.rank:.4f}")
    print(f"Evaluation Time: {eval_time:.2f}s")
    success = (max_diff == 0) and (final_loss < hypers.success_threshold)
    print(f"Validation Status: {'PASS' if success else 'FAIL'} (threshold={hypers.success_threshold})")

    final_decomposition_np = [
        U_np.astype(np.float64),
        V_np.astype(np.float64),
        W_np.astype(np.float64),
    ]
    return final_decomposition_np, n, m, p, float(final_loss), hypers.rank


# EVOLVE-BLOCK-END
if __name__ == "__main__":
    run()