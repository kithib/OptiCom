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
from scipy.optimize import minimize


@dataclass
class Hyperparameters:
    learning_rate: float = 0.0005
    num_steps: int = 50000
    num_restarts: int = 20
    num_hermite_coeffs: int = 4  # uses H0, H4, H8, H12
    post_opt_steps: int = 1500
    x_grid_end: float = 6.0
    x_grid_points: int = 5000


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

    def _build_poly_coeffs(self, params):
        c_others, log_c_last = params[:-1], params[-1]
        c_last = jnp.exp(log_c_last)
        
        # Enforce P(0) = 0 constraint
        c0 = -(jnp.sum(c_others * self.H_vals_at_zero[1:-1]) + c_last * self.H_vals_at_zero[-1]) / self.H_vals_at_zero[0]
        hermite_coeffs = jnp.concatenate([jnp.array([c0]), c_others, jnp.array([c_last])])
        return jnp.sum(hermite_coeffs[:, None] * self.hermite_basis, axis=0)

    def _objective_fn(self, params):
        poly_coeffs = self._build_poly_coeffs(params)
        p_values = jnp.polyval(poly_coeffs, self.x_grid)
        
        # Weighted loss that emphasizes regions near potential roots and larger x values
        weights = 1.0 + jnp.exp(5.0 * (self.x_grid / self.x_grid[-1] - 0.8))
        neg_loss = jnp.sum(weights * jax.nn.relu(-p_values))
        
        # Add regularization to prevent coefficient explosion
        reg_loss = 1e-6 * jnp.sum(jnp.square(params[:-1])) + 1e-8 * jnp.square(params[-1])
        
        return neg_loss + reg_loss

    def train_step(self, params, opt_state):
        loss, grads = jax.value_and_grad(self._objective_fn)(params)
        updates, opt_state = self.optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def post_optimize(self, params):
        """Refine optimization with L-BFGS for better convergence"""
        def numpy_obj_fn(np_params):
            jax_params = jnp.array(np_params)
            loss = self._objective_fn(jax_params)
            return float(loss)

        def numpy_grad_fn(np_params):
            jax_params = jnp.array(np_params)
            grads = jax.grad(self._objective_fn)(jax_params)
            return np.array(grads)

        result = minimize(
            numpy_obj_fn,
            np.array(params),
            jac=numpy_grad_fn,
            method='L-BFGS-B',
            options={'maxiter': self.hypers.post_opt_steps, 'disp': False}
        )
        return jnp.array(result.x)


def run_single_trial(optimizer: UncertaintyOptimizer, key: jax.random.PRNGKey):
    """Runs one full optimization with improved initialization and refinement"""
    num_params_to_opt = optimizer.hypers.num_hermite_coeffs - 1
    
    # Expanded set of starting points covering broader parameter space
    init_options = [
        jnp.array([-0.0115, -8.9e-5, np.log(1e-6)]),
        jnp.array([-0.010, -7.0e-5, np.log(5e-7)]),
        jnp.array([-0.013, -1.0e-4, np.log(2e-6)]),
        jnp.array([-0.012, -9.2e-5, np.log(1.2e-6)]),
        jnp.array([-0.011, -8.7e-5, np.log(8e-7)]),
        jnp.array([-0.0095, -6.5e-5, np.log(3e-7)]),
        jnp.array([-0.0135, -1.1e-4, np.log(2.5e-6)]),
        jnp.array([-0.0105, -7.8e-5, np.log(6e-7)]),
    ]
    
    base = init_options[jax.random.randint(key, (), 0, len(init_options))]
    noise = jax.random.normal(key, (num_params_to_opt,)) * 5e-4
    params = base + noise

    opt_state = optimizer.optimizer.init(params)
    jit_train_step = jax.jit(optimizer.train_step)

    # Main ADAM optimization
    for _ in range(optimizer.hypers.num_steps):
        params, opt_state, _ = jit_train_step(params, opt_state)
    
    # Refine with L-BFGS
    params = optimizer.post_optimize(params)
    
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
    """Compute r_max and C4 from full Hermite coefficient vector with improved root verification."""
    degrees = [4 * k for k in range(num_hermite_coeffs)]
    P = _build_P_from_hermite_coeffs(hermite_coeffs.copy(), degrees)

    # Divide by x^2 (since P(0)=0 and even function, x^2 divides P(x))
    Q, R = np.polydiv(P, np.poly1d([1.0, 0.0, 0.0]))
    if np.max(np.abs(R.c)) > 1e-8:
        return None, None

    roots = Q.r
    real_pos = roots[(np.isreal(roots)) & (roots.real > 1e-6)].real
    if real_pos.size == 0:
        return None, None

    # Verify sign changes to confirm actual roots
    r_max = None
    sorted_roots = np.sort(real_pos)
    
    # Check polynomial behavior at large x to ensure we're capturing the rightmost root
    large_x = 2.0 * sorted_roots[-1] if sorted_roots.size > 0 else 10.0
    large_x_val = np.polyval(Q, large_x)
    
    # Find the rightmost root where sign changes from negative to positive
    for r in reversed(sorted_roots):
        eps = 1e-9 * max(1.0, abs(r))
        left = np.polyval(Q, r - eps)
        right = np.polyval(Q, r + eps)
        
        if left * right < 0:
            # Ensure this is the rightmost valid root
            if large_x_val > 0 and left < 0:
                r_max = float(r)
                break
    
    if r_max is None:
        # Fallback to largest real root with sanity check
        r_max = float(sorted_roots[-1])
        # Verify polynomial is positive at 2*r_max
        if np.polyval(Q, 2*r_max) <= 0:
            return None, None

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
        try:
            final_params = run_single_trial(optimizer, restart_key)
            coeffs, c4_bound, r_max = get_c4_from_params(np.array(final_params), hypers)

            if c4_bound is not None and c4_bound < best_c4_bound:
                best_c4_bound = c4_bound
                best_coeffs = coeffs
                best_r_max = r_max
        except Exception as e:
            print(f"Trial failed with error: {str(e)}")
            continue

    if best_coeffs is None:
        raise RuntimeError("Failed to find a valid solution in any restart.")

    print("\nSearch complete.")
    print(f"Best Hermite coeffs: {best_coeffs}")
    print(f"Best largest positive root r_max: {best_r_max:.12f}")
    print(f"Resulting best C4 upper bound: {best_c4_bound:.12f}")

    return best_coeffs, best_c4_bound, best_r_max


# EVOLVE-BLOCK-END