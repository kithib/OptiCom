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
from scipy.optimize import root_scalar, minimize


@dataclass
class Hyperparameters:
    learning_rate: float = 0.0005
    num_steps: int = 120000
    num_restarts: int = 20
    num_hermite_coeffs: int = 4  # uses H0, H4, H8, H12
    x_grid_end: float = 8.0
    x_grid_points: int = 4000
    weight_power: float = 2.0  # Exponent for distance-based weighting
    regularization: float = 1e-6  # L2 regularization strength
    restart_noise_scale: float = 1e-3  # Scale for initialization noise
    post_tune_iterations: int = 500  # L-BFGS-B post-tuning iterations
    early_stop_patience: int = 5000  # Patience for early stopping
    early_stop_tolerance: float = 1e-8  # Tolerance for early stopping


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
        self.optimizer = optax.adam(self.hypers.learning_rate)

        # Precompute basis polynomial values on grid for faster evaluation
        self.basis_grid_vals = jnp.array([np.polyval(p.coef, self.x_grid) for p in hermite_polys])
        
        # Precompute weighting function with normalization
        self.weights = (self.x_grid / (hypers.x_grid_end + 1e-12)) ** hypers.weight_power
        self.weights = self.weights / jnp.max(self.weights)

    @staticmethod
    def _objective_fn(params: jnp.ndarray, basis_grid_vals, H_vals_at_zero, weights, regularization):
        """Penalize negative values of P(x) on [0, Xmax]; weighted MSE loss with L2 regularization."""
        c_others, log_c_last = params[:-1], params[-1]
        c_last = jnp.exp(log_c_last)

        # Enforce P(0) = 0
        c0 = (
            -(jnp.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1])
            / H_vals_at_zero[0]
        )
        hermite_coeffs = jnp.concatenate([jnp.array([c0]), c_others, jnp.array([c_last])])

        # Calculate polynomial values using precomputed basis values
        p_values = jnp.sum(hermite_coeffs[:, None] * basis_grid_vals, axis=0)

        # Weighted MSE loss for negative values with robust clamping
        neg_loss = jnp.mean(weights * jax.nn.relu(-p_values) ** 2)
        
        # L2 regularization to prevent coefficient explosion
        reg_loss = regularization * jnp.mean(jnp.square(hermite_coeffs))
        return neg_loss + reg_loss

    @staticmethod
    def train_step(
        params: jnp.ndarray,
        opt_state: optax.OptState,
        optimizer,
        basis_grid_vals,
        H_vals_at_zero,
        weights,
        regularization,
    ):
        loss, grads = jax.value_and_grad(UncertaintyOptimizer._objective_fn)(
            params, basis_grid_vals, H_vals_at_zero, weights, regularization
        )
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss


def run_single_trial(optimizer: UncertaintyOptimizer, key: jax.random.PRNGKey):
    """Runs one full optimization with Adam + early stopping + L-BFGS-B post-refinement."""
    num_params_to_opt = optimizer.hypers.num_hermite_coeffs - 1  # = 3 when using H0,H4,H8,H12
    assert num_params_to_opt == 3, "This initialization assumes num_hermite_coeffs == 4."

    base_c1 = -0.01158510802599293
    base_c2 = -8.921606035407065e-05
    base_log_c_last = np.log(1e-6)

    base = jnp.array([base_c1, base_c2, base_log_c_last], dtype=jnp.float32)
    noise = jax.random.normal(key, (num_params_to_opt,)) * optimizer.hypers.restart_noise_scale
    params = base + noise

    opt_state = optimizer.optimizer.init(params)
    jit_train_step = jax.jit(UncertaintyOptimizer.train_step, static_argnums=(2,))

    # Adam optimization with early stopping
    best_loss = float('inf')
    patience_counter = 0
    
    for step in range(optimizer.hypers.num_steps):
        params, opt_state, loss = jit_train_step(
            params,
            opt_state,
            optimizer.optimizer,
            optimizer.basis_grid_vals,
            optimizer.H_vals_at_zero,
            optimizer.weights,
            optimizer.hypers.regularization,
        )
        
        # Early stopping logic
        if loss < best_loss - optimizer.hypers.early_stop_tolerance:
            best_loss = loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if patience_counter >= optimizer.hypers.early_stop_patience:
            break
    
    # Ensure numerical stability by clamping log values
    params = jnp.where(params[-1] < np.log(1e-10), np.log(1e-10), params)
    params = jnp.where(params[-1] > np.log(1e2), np.log(1e2), params)
    
    # Post-optimization refinement with L-BFGS-B
    def objective_wrapper(params_np):
        params_jnp = jnp.array(params_np)
        return float(UncertaintyOptimizer._objective_fn(
            params_jnp,
            optimizer.basis_grid_vals,
            optimizer.H_vals_at_zero,
            optimizer.weights,
            optimizer.hypers.regularization
        ))
    
    result = minimize(
        objective_wrapper,
        np.array(params),
        method='L-BFGS-B',
        options={'maxiter': optimizer.hypers.post_tune_iterations, 'ftol': 1e-10}
    )
    
    return jnp.array(result.x) if result.success else params


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
    """Compute r_max and C4 from full Hermite coefficient vector with precise root verification."""
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    # Divide by x^2 since P(0) = 0 and we have even function (double root at 0)
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-10:
        return None, None

    # Get all roots and filter real positive ones
    roots = Q.r
    real_pos = roots[(np.isreal(roots)) & (roots.real > 1e-10)].real
    if real_pos.size == 0:
        return None, None

    # Sort candidates and find the largest actual root with sign change
    r_candidates = np.sort(real_pos)
    r_max = None
    
    # Check largest candidate first since we care about the maximum root
    for r in reversed(r_candidates):
        # Use bracket around candidate to verify root
        lower = max(r * 0.999, 1e-10)
        upper = r * 1.001
        
        try:
            # Use Brentq method for precise root finding with tight tolerance
            res = root_scalar(
                lambda x: np.polyval(Q, x), 
                bracket=[lower, upper], 
                method='brentq',
                rtol=1e-12
            )
            if res.converged:
                r_max = res.root
                break
        except ValueError:
            # Bracket doesn't contain root, continue to next candidate
            continue
    
    # Fallback to largest candidate if no sign change found
    if r_max is None:
        r_max = r_candidates[-1]

    # Calculate C4 using precise formula (matches evaluator's calculation)
    c4 = (r_max ** 2) / (2 * np.pi)
    return c4, r_max


def get_c4_from_params(params: np.ndarray, hypers: Hyperparameters):
    """Calculates the precise C4 bound from a final set of parameters with validation."""
    c_others, log_c_last = params[:-1], params[-1]
    c_last = np.exp(log_c_last)

    # Prevent overflow/underflow in coefficient calculation
    if np.isinf(c_last) or c_last < 1e-12:
        return None, None, None

    degrees = [4 * k for k in range(hypers.num_hermite_coeffs)]
    hermite_polys = [hermite(d) for d in degrees]
    H_vals_at_zero = np.array([p(0) for p in hermite_polys])

    # Enforce P(0) = 0
    c0 = -(np.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
    hermite_coeffs = np.concatenate([[c0], np.array(c_others), [c_last]])

    # Sanity check: ensure no NaNs/Infs in coefficients
    if np.any(np.isnan(hermite_coeffs)) or np.any(np.isinf(hermite_coeffs)):
        return None, None, None

    c4, rmax = _c4_from_hermite_coeffs(hermite_coeffs, hypers.num_hermite_coeffs)
    if c4 is None or np.isnan(c4) or np.isinf(c4):
        return None, None, None
    
    # Ensure C4 is within physically plausible bounds
    if c4 < 0 or c4 > 5.0:
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
    print(f"Combined score: {0.3215872333529007 / best_c4_bound:.12f}")

    return best_coeffs, best_c4_bound, best_r_max


# EVOLVE-BLOCK-END