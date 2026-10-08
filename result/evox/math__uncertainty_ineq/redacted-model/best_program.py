# Disable progress bar for cleaner output logs
import os

os.environ["TQDM_DISABLE"] = "1"

# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from scipy.special import hermite
import tqdm
from scipy.optimize import root_scalar


@dataclass(frozen=True)
class Hyperparameters:
    learning_rate: float = 0.0007
    num_steps: int = 68000
    num_restarts: int = 24
    num_hermite_coeffs: int = 4  # uses H0, H4, H8, H12
    x_grid_max: float = 8.8
    x_grid_points: int = 2800
    weight_exponent: float = 3.7
    reg_strength: float = 3.0e-7
    noise_scale: float = 4.0e-4
    beta1: float = 0.9
    beta2: float = 0.999
    epsilon: float = 1e-8
    root_tolerance: float = 1e-12
    plateau_threshold: float = 8e-7
    plateau_patience: int = 3800
    lr_decay_factor: float = 0.55
    lr_decay_patience: int = 2800
    root_refine_iterations: int = 3


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
        self.x_grid = jnp.linspace(0.0, hypers.x_grid_max, hypers.x_grid_points)
        
        # Precompute polynomial derivatives for root refinement
        self.hermite_basis_deriv = jnp.stack([
            jnp.array(np.pad(np.polyder(poly).coef, (max_degree - np.polyder(poly).order, 0)))
            for poly in hermite_polys
        ])

    @staticmethod
    def _objective_fn(params: jnp.ndarray, hermite_basis, H_vals_at_zero, x_grid, weight_exponent, reg_strength):
        """Penalize negative values of P(x) on [0, Xmax]; strong weighting for larger x with regularization."""
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

        # Weight larger x values more heavily to push roots leftward
        weights = jnp.power((x_grid / (x_grid[-1] + 1e-12)), weight_exponent)
        loss = jnp.sum(weights * jax.nn.relu(-p_values))
        
        # Regularization to prevent extreme coefficients and numerical instability
        reg_loss = reg_strength * jnp.sum(jnp.square(hermite_coeffs))
        return loss + reg_loss

    @staticmethod
    def train_step(
        params: jnp.ndarray,
        opt_state: tuple[jnp.ndarray, jnp.ndarray],
        beta1: float,
        beta2: float,
        epsilon: float,
        learning_rate: float,
        hermite_basis,
        H_vals_at_zero,
        x_grid,
        weight_exponent,
        reg_strength,
    ):
        loss, grads = jax.value_and_grad(UncertaintyOptimizer._objective_fn)(
            params, hermite_basis, H_vals_at_zero, x_grid, weight_exponent, reg_strength
        )
        
        # Manual Adam optimization to avoid optax dependency issues
        m, v = opt_state
        m = beta1 * m + (1.0 - beta1) * grads
        v = beta2 * v + (1.0 - beta2) * jnp.square(grads)
        m_hat = m / (1.0 - beta1)
        v_hat = v / (1.0 - beta2)
        updates = -learning_rate * m_hat / (jnp.sqrt(v_hat) + epsilon)
        params = params + updates
        
        return params, (m, v), loss


def run_single_trial(optimizer: UncertaintyOptimizer, key: jax.random.PRNGKey):
    """Runs one full optimization from a near-good starting point with small noise, with learning rate decay and early stopping."""
    num_params_to_opt = optimizer.hypers.num_hermite_coeffs - 1
    assert num_params_to_opt == 3, "This initialization assumes num_hermite_coeffs == 4."

    base_c1 = -0.01158510802599293
    base_c2 = -8.921606035407065e-05
    base_log_c_last = np.log(1e-6)

    base = jnp.array([base_c1, base_c2, base_log_c_last], dtype=jnp.float32)
    noise = jax.random.normal(key, (num_params_to_opt,)) * optimizer.hypers.noise_scale
    params = base + noise

    # Initialize Adam state
    m = jnp.zeros_like(params)
    v = jnp.zeros_like(params)
    opt_state = (m, v)
    
    # JIT compile with static parameters marked correctly
    jit_train_step = jax.jit(
        UncertaintyOptimizer.train_step,
        static_argnums=(2, 3, 4, 5)
    )

    best_loss = float('inf')
    plateau_count = 0
    decay_count = 0
    current_lr = optimizer.hypers.learning_rate

    for step in range(optimizer.hypers.num_steps):
        params, opt_state, loss = jit_train_step(
            params,
            opt_state,
            optimizer.hypers.beta1,
            optimizer.hypers.beta2,
            optimizer.hypers.epsilon,
            current_lr,
            optimizer.hermite_basis,
            optimizer.H_vals_at_zero,
            optimizer.x_grid,
            optimizer.hypers.weight_exponent,
            optimizer.hypers.reg_strength,
        )
        
        # Early stopping based on loss plateau
        if loss < best_loss - optimizer.hypers.plateau_threshold:
            best_loss = loss
            plateau_count = 0
            decay_count = 0
        else:
            plateau_count += 1
            decay_count += 1
            
        # Learning rate decay
        if decay_count >= optimizer.hypers.lr_decay_patience:
            current_lr *= optimizer.hypers.lr_decay_factor
            decay_count = 0
            
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

    # Ensure highest-degree coefficient is positive (matches evaluator expectation)
    if P_poly_coeffs[0] < 0:
        P_poly_coeffs = -P_poly_coeffs
        hermite_coeffs[:] = -hermite_coeffs
    return np.poly1d(P_poly_coeffs)


def _refine_root(Q: np.poly1d, initial_r: float, tolerance: float, max_iter: int) -> float:
    """Refine root using Newton-Raphson method for better accuracy."""
    Q_deriv = np.polyder(Q)
    r = initial_r
    
    for _ in range(max_iter):
        f_val = np.polyval(Q, r)
        f_deriv_val = np.polyval(Q_deriv, r)
        
        if abs(f_deriv_val) < 1e-15:
            break
            
        delta = f_val / f_deriv_val
        r -= delta
        
        if abs(delta) < tolerance:
            break
            
    return r


def _c4_from_hermite_coeffs(hermite_coeffs: np.ndarray, num_hermite_coeffs: int, root_tolerance: float, refine_iterations: int):
    """Compute r_max and C4 from full Hermite coefficient vector with improved root verification and refinement."""
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    # Divide by x^2 since P(0)=0 and even function implies x^2 factor
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > root_tolerance:
        return None, None

    roots = Q.r
    real_pos = roots[(np.isreal(roots)) & (roots.real > root_tolerance)].real
    if real_pos.size == 0:
        return None, None

    # Sort candidates and verify actual roots via sign change with multiple tolerance checks
    r_candidates = np.sort(real_pos)
    r_max = None
    
    # Check each candidate with increasing precision
    for r in reversed(r_candidates):
        # First check with small epsilon
        eps_small = 1e-10 * max(1.0, abs(r))
        left_small = np.polyval(Q, r - eps_small)
        right_small = np.polyval(Q, r + eps_small)
        
        # Verify with larger epsilon to avoid numerical artifacts
        eps_large = 1e-8 * max(1.0, abs(r))
        left_large = np.polyval(Q, r - eps_large)
        right_large = np.polyval(Q, r + eps_large)
        
        # Only consider valid roots with consistent sign changes
        if (left_small * right_small < 0) and (left_large * right_large < 0):
            # Refine root for better accuracy
            refined_r = _refine_root(Q, r, root_tolerance, refine_iterations)
            r_max = float(refined_r)
            break
    
    # Fallback to largest candidate if no valid sign changes found
    if r_max is None:
        r_max = float(r_candidates[-1])
        # Refine fallback root as well
        r_max = float(_refine_root(Q, r_max, root_tolerance, refine_iterations))

    # Correct C4 calculation: (r_max^2)/(2π) per problem statement
    c4 = (r_max**2) / (2 * np.pi)
    return c4, r_max


def get_c4_from_params(params: np.ndarray, hypers: Hyperparameters):
    """Calculates the precise C4 bound from a final set of parameters."""
    c_others, log_c_last = params[:-1], params[-1]
    c_last = np.exp(log_c_last)

    degrees = [4 * k for k in range(hypers.num_hermite_coeffs)]
    hermite_polys = [hermite(d) for d in degrees]
    H_vals_at_zero = np.array([p(0) for p in hermite_polys])

    # Enforce P(0) = 0 constraint correctly
    c0 = (
        -(np.sum(c_others * H_vals_at_zero[1:-1]) + c_last * H_vals_at_zero[-1]) / H_vals_at_zero[0]
    )
    hermite_coeffs = np.concatenate([[c0], np.array(c_others), [c_last]])

    c4, rmax = _c4_from_hermite_coeffs(hermite_coeffs, hypers.num_hermite_coeffs, hypers.root_tolerance, hypers.root_refine_iterations)
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