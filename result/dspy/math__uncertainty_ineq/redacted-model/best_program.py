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
    learning_rate: float = 0.000215
    num_steps: int = 88000
    num_restarts: int = 42
    num_hermite_coeffs: int = 4  # uses H0, H4, H8, H12
    x_grid_end: float = 5.3
    x_grid_points: int = 2400
    weight_exponent: float = 3.85
    l2_reg: float = 5.5e-7
    plateau_patience: int = 6800
    plateau_tolerance: float = 3.5e-9
    init_noise_scale: float = 3.4e-4


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
        self.x_grid = jnp.linspace(0.0, hypers.x_grid_end, hypers.x_grid_points)
        
        # Cosine decay learning rate schedule with warmup
        lr_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=hypers.learning_rate,
            warmup_steps=4500,
            decay_steps=hypers.num_steps,
            end_value=hypers.learning_rate * 0.075
        )
        self.optimizer = optax.adamw(
            learning_rate=lr_schedule,
            weight_decay=hypers.l2_reg
        )

    @staticmethod
    def _objective_fn(params: jnp.ndarray, hermite_basis, H_vals_at_zero, x_grid, weight_exponent, l2_reg):
        """Penalize negative values of P(x) on [0, Xmax]; strong weighting to emphasize larger x with regularization."""
        c_others, log_c_last = params[:-1], params[-1]
        c_last = jnp.exp(log_c_last)

        # Enforce P(0) = 0 constraint
        c0 = -(jnp.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
        hermite_coeffs = jnp.concatenate([jnp.array([c0]), c_others, jnp.array([c_last])])

        poly_coeffs_std = jnp.sum(hermite_coeffs[:, None] * hermite_basis, axis=0)
        p_values = jnp.polyval(poly_coeffs_std, x_grid)

        # Strongly increasing weight toward larger x values with smoother transition
        weights = jnp.power((x_grid / (x_grid[-1] + 1e-12)), weight_exponent)
        # Add small constant to prevent zero weight at x=0
        weights = weights + 1e-4
        
        loss = jnp.sum(weights * jax.nn.relu(-p_values))
        
        # L2 regularization to prevent coefficient explosion
        reg_loss = l2_reg * jnp.sum(jnp.square(hermite_coeffs))
        
        # Additional penalty for very small leading coefficient to ensure numerical stability
        loss += 4e-7 * jnp.reciprocal(jnp.abs(c_last) + 1e-10)
        
        return loss + reg_loss

    @staticmethod
    def train_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer,
        hermite_basis,
        H_vals_at_zero,
        x_grid,
        weight_exponent,
        l2_reg,
    ):
        loss, grads = jax.value_and_grad(UncertaintyOptimizer._objective_fn)(
            params, hermite_basis, H_vals_at_zero, x_grid, weight_exponent, l2_reg
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


def run_single_trial(optimizer: UncertaintyOptimizer, key: jax.random.PRNGKey):
    """Runs one full optimization with adaptive early stopping and improved initialization."""
    num_params_to_opt = optimizer.hypers.num_hermite_coeffs - 1

    # Improved initialization based on refined known solutions
    base_c1 = -0.01235
    base_c2 = -1.01e-04
    base_log_c_last = np.log(1.22e-6)

    base = jnp.array([base_c1, base_c2, base_log_c_last], dtype=jnp.float32)
    noise = jax.random.normal(key, (num_params_to_opt,)) * optimizer.hypers.init_noise_scale
    params = base + noise

    opt_state = optimizer.optimizer.init(params)
    jit_train_step = jax.jit(UncertaintyOptimizer.train_step, static_argnums=(2,))

    best_loss = float("inf")
    plateau_count = 0

    for step in range(optimizer.hypers.num_steps):
        params, opt_state, loss = jit_train_step(
            params,
            opt_state,
            optimizer.optimizer,
            optimizer.hermite_basis,
            optimizer.H_vals_at_zero,
            optimizer.x_grid,
            optimizer.hypers.weight_exponent,
            optimizer.hypers.l2_reg,
        )

        # Early stopping on plateau with stricter tolerance
        if loss < best_loss - optimizer.hypers.plateau_tolerance:
            best_loss = loss
            plateau_count = 0
        else:
            plateau_count += 1
            
        if plateau_count >= optimizer.hypers.plateau_patience:
            break

    return params


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
    """Compute r_max and C4 from full Hermite coefficient vector with robust root verification."""
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    # Divide by x^2 to handle P(0)=0 constraint
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-8:
        return None, None

    roots = Q.r
    # Filter real positive roots with small epsilon to avoid numerical zero artifacts
    real_pos = roots[(np.isreal(roots)) & (roots.real > 1e-6)].real
    if real_pos.size == 0:
        return None, None

    # Precise root verification with sign change check and bisection refinement
    r_candidates = np.sort(real_pos)
    r_max = None
    
    for r in r_candidates:
        # Bisection refinement for better root accuracy
        left = r - 1e-8
        right = r + 1e-8
        for _ in range(12):
            mid = (left + right) / 2
            if np.polyval(Q, left) * np.polyval(Q, mid) < 0:
                right = mid
            else:
                left = mid
        refined_r = (left + right) / 2
        
        # Verify sign change at refined root
        eps = 1e-10 * max(1.0, abs(refined_r))
        left_val = np.polyval(Q, refined_r - eps)
        right_val = np.polyval(Q, refined_r + eps)
        if left_val * right_val < 0:
            r_max = float(refined_r)
    
    # Fallback to largest refined candidate if no sign change found
    if r_max is None:
        # Refine largest candidate
        r = r_candidates[-1]
        left = r - 1e-8
        right = r + 1e-8
        for _ in range(12):
            mid = (left + right) / 2
            if np.polyval(Q, left) * np.polyval(Q, mid) < 0:
                right = mid
            else:
                left = mid
        r_max = float((left + right) / 2)

    c4 = (r_max**2) / (2 * np.pi)
    return c4, r_max


def get_c4_from_params(params: np.ndarray, hypers: Hyperparameters):
    """Calculates the precise C4 bound from a final set of parameters."""
    c_others, log_c_last = params[:-1], params[-1]
    c_last = np.exp(log_c_last)

    degrees = [4 * k for k in range(hypers.num_hermite_coeffs)]
    hermite_polys = [hermite(d) for d in degrees]
    H_vals_at_zero = np.array([p(0) for p in hermite_polys])

    # Enforce P(0) = 0 constraint
    c0 = -(np.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
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
    print(f"Best largest positive root r_max: {best_r_max:.12f}")
    print(f"Resulting best C4 upper bound: {best_c4_bound:.12f}")

    return best_coeffs, best_c4_bound, best_r_max


# EVOLVE-BLOCK-END