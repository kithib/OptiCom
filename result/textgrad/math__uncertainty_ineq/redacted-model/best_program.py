# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"

# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
from scipy.special import hermite
import tqdm


@dataclass
class Hyperparameters:
    learning_rate: float = 0.001
    num_steps: int = 55000
    num_restarts: int = 9
    num_hermite_coeffs: int = 4


class UncertaintyOptimizer:
    """
    Finds coefficients for a generalized Hermite polynomial P(x) that minimize
    the largest positive root, providing an upper bound for C4.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.degrees = [4 * k for k in range(hypers.num_hermite_coeffs)]
        max_degree = self.degrees[-1]
        hermite_polys = [hermite(d) for d in self.degrees]

        basis = []
        for poly in hermite_polys:
            pad_amount = max_degree - poly.order
            basis.append(jnp.array(np.pad(poly.coef, (pad_amount, 0))))
        self.hermite_basis = jnp.stack(basis)

        self.H_vals_at_zero = jnp.array([p(0) for p in hermite_polys])
        # Focused grid with balance between critical resolution and speed
        self.x_grid = jnp.concatenate([
            jnp.linspace(0.0, 1.39, 80),       # Coarse early region
            jnp.linspace(1.39, 1.405, 250),     # Pre-critical
            jnp.linspace(1.405, 1.42, 5200),    # Critical root zone - resolution balanced
            jnp.linspace(1.42, 1.44, 150),      # Post-critical buffer
            jnp.linspace(1.44, 2.0, 50)          # Final tail
        ])
        # Stable learning rate schedule
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=1e-5,
            peak_value=self.hypers.learning_rate,
            warmup_steps=600,
            decay_steps=self.hypers.num_steps - 600,
            end_value=1e-7
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(1.0),
            optax.adam(lr_schedule)
        )

    @staticmethod
    def _objective_fn(params: jnp.ndarray, hermite_basis, H_vals_at_zero, x_grid):
        """Balanced negative penalty with sharp root region focus."""
        c_others, log_c_last = params[:-1], params[-1]
        c_last = jnp.exp(log_c_last)

        # Enforce P(0) = 0
        c0 = (
            -(jnp.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1])
            / H_vals_at_zero[0]
        )
        hermite_coeffs = jnp.concatenate([jnp.array([c0]), c_others, jnp.array([c_last])])

        poly_coeffs_std = jnp.sum(hermite_coeffs[:, None] * hermite_basis, axis=0)
        p_values = jnp.polyval(poly_coeffs_std, x_grid)

        # Sharp focus on root zone + heavy tail penalty for violations after expected root
        weights = (1.0 + 25.0 * jnp.exp(-((x_grid - 1.413) / 0.008) ** 6) +
                   6.0 * jnp.where(x_grid > 1.40, (x_grid / 1.6) ** 16, 0.0))
        
        neg_penalty = jax.nn.relu(-p_values)
        # Balanced polynomial penalty for good gradient signal without over-squashing
        loss = jnp.sum(weights * (neg_penalty + 5.0 * neg_penalty ** 2.5 + 15.0 * neg_penalty ** 5))
        
        # Light regularization
        reg = 1e-7 * (jnp.sum(c_others ** 2) + log_c_last ** 2)
        return loss + reg

    @staticmethod
    def train_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer,
        hermite_basis,
        H_vals_at_zero,
        x_grid,
    ):
        loss, grads = jax.value_and_grad(UncertaintyOptimizer._objective_fn)(
            params, hermite_basis, H_vals_at_zero, x_grid
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


def run_single_trial(optimizer: UncertaintyOptimizer, key: jax.random.PRNGKey):
    """Runs one optimization with early stopping to prevent timeouts."""
    num_params_to_opt = optimizer.hypers.num_hermite_coeffs - 1
    assert num_params_to_opt == 3, "This initialization assumes num_hermite_coeffs == 4."

    # Refined baseline initialization
    base_c1 = -0.01178
    base_c2 = -9.02e-05
    base_log_c_last = np.log(1.205e-6)

    base = jnp.array([base_c1, base_c2, base_log_c_last], dtype=jnp.float32)
    # Moderate exploration noise
    noise = jax.random.normal(key, (num_params_to_opt,)) * jnp.array([2e-4, 5e-6, 0.04])
    params = base + noise

    opt_state = optimizer.optimizer.init(params)
    jit_train_step = jax.jit(UncertaintyOptimizer.train_step, static_argnums=(2,))

    best_loss = float("inf")
    best_params = params
    patience_counter = 0
    # Early stopping prevents timeout while exploring thoroughly
    early_stopping_patience = 4000

    for _ in range(optimizer.hypers.num_steps):
        params, opt_state, loss = jit_train_step(
            params,
            opt_state,
            optimizer.optimizer,
            optimizer.hermite_basis,
            optimizer.H_vals_at_zero,
            optimizer.x_grid,
        )

        if loss < best_loss:
            best_loss = loss
            best_params = params
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= early_stopping_patience:
            break

    return best_params


def _build_P_from_hermite_coeffs(hermite_coeffs: np.ndarray, degrees: list[int]) -> np.poly1d:
    """Build monomial-basis polynomial P from Hermite-basis coefficients."""
    max_degree = degrees[-1]
    hermite_polys = [hermite(d) for d in degrees]

    P_poly_coeffs = np.zeros(max_degree + 1)
    for i, c in enumerate(hermite_coeffs):
        poly = hermite_polys[i]
        pad_amount = max_degree - poly.order
        P_poly_coeffs[pad_amount:] += c * poly.coef

    if P_poly_coeffs[0] < 0:
        P_poly_coeffs = -P_poly_coeffs
        hermite_coeffs[:] = -hermite_coeffs
    return np.poly1d(P_poly_coeffs)


def _c4_from_hermite_coeffs(hermite_coeffs: np.ndarray, num_hermite_coeffs: int):
    """Compute r_max and C4 from full Hermite coefficient vector."""
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    # Divide by x^2
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-10:
        return None, None

    roots = Q.r
    real_pos = roots[(np.isreal(roots)) & (roots.real > 0)].real
    if real_pos.size == 0:
        return None, None

    # Tiny sign-change check around candidates
    r_candidates = np.sort(real_pos)
    r_max = None
    for r in r_candidates:
        eps = 1e-10 * max(1.0, abs(r))
        left = np.polyval(Q, r - eps)
        right = np.polyval(Q, r + eps)
        if left * right < 0:
            r_max = float(r)
    if r_max is None:
        r_max = float(r_candidates[-1])

    c4 = (r_max**2) / (2 * np.pi)
    return c4, r_max


def get_c4_from_params(params: np.ndarray, hypers: Hyperparameters):
    """Calculates the precise C4 bound from a final set of parameters."""
    c_others, log_c_last = params[:-1], params[-1]
    c_last = np.exp(log_c_last)

    degrees = [4 * k for k in range(hypers.num_hermite_coeffs)]
    hermite_polys = [hermite(d) for d in degrees]
    H_vals_at_zero = np.array([p(0) for p in hermite_polys])

    # Enforce P(0) = 0
    c0 = (
        -(np.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
    )
    hermite_coeffs = np.concatenate([[c0], np.array(c_others), [c_last]])

    c4, rmax = _c4_from_hermite_coeffs(hermite_coeffs, hypers.num_hermite_coeffs)
    if c4 is None:
        return None, None, None
    return hermite_coeffs, c4, rmax


def run():
    hypers = Hyperparameters()
    optimizer = UncertaintyOptimizer(hypers)
    main_key = jax.random.PRNGKey(42)
    best_c4_bound = float("inf")
    best_coeffs, best_r_max = None, None

    print(f"Running {hypers.num_restarts} trials to find the best C4 upper bound...")
    for _ in tqdm.tqdm(range(hypers.num_restarts), desc="Searching", disable=True):
        main_key, restart_key = jax.random.split(main_key)
        final_params = run_single_trial(optimizer, restart_key)

        coeffs, c4_bound, r_max = get_c4_from_params(np.array(final_params), hypers)

        if c4_bound is not None and c4_bound < best_c4_bound:
            best_c4_bound = c4_bound
            best_coeffs = coeffs
            best_r_max = r_max

    if best_coeffs is None:
        raise RuntimeError("Failed to find a valid solution in any restart.")

    print("\nSearch complete.")
    print(f"Best Hermite coeffs: {best_coeffs}")
    print(f"Best largest positive root r_max: {best_r_max:.8f}")
    print(f"Resulting best C4 upper bound: {best_c4_bound:.8f}")

    return best_coeffs, best_c4_bound, best_r_max


# EVOLVE-BLOCK-END