# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 600
    learning_rate: float = 0.005
    end_lr_factor: float = 1e-4
    num_steps: int = 40000
    warmup_steps: int = 2000


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes the objective function, which is the C1 ratio.
        We minimize this ratio to find a tight upper bound.
        """
        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx

        eps = 1e-9
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        fft_conv = fft_f * fft_f
        conv_f_f = jnp.fft.ifft(fft_conv).real

        # Scale by dx.
        scaled_conv_f_f = conv_f_f * self.dx

        max_conv = jnp.max(scaled_conv_f_f)
        c1_ratio = max_conv / (integral_f_safe**2)

        # Return the value to be MINIMIZED.
        return c1_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        """Performs a single training step."""
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)

        return f_values, opt_state, loss

    def run_optimization(self):
        """Sets up and runs the full optimization process."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.adam(learning_rate=schedule)

        key = jax.random.PRNGKey(42)
        N = self.hypers.num_intervals
        f_values = jnp.zeros((N,))
        start_idx, end_idx = N // 4, 3 * N // 4
        f_values = f_values.at[start_idx:end_idx].set(1.0)
        f_values += 0.05 * jax.random.uniform(key, (N,))

        opt_state = self.optimizer.init(f_values)

        print(
            f"Number of intervals (N): {self.hypers.num_intervals}, Steps: {self.hypers.num_steps}"
        )

        train_step_jit = jax.jit(self.train_step)

        loss = jnp.inf  # Initialize loss
        for step in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if step % 2000 == 0 or step == self.hypers.num_steps - 1:
                # CORRECTED PRINTING: Show the positive loss value directly.
                print(f"Step {step:5d} | C1 ≈ {loss:.8f}")

        print(f"Final C1 found: {loss:.8f}")

        return jax.nn.relu(f_values), loss


def run():
    """Entry point for running the optimization and returning results."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    final_c1 = float(final_loss_val)

    f_values_np = np.array(optimized_f)

    return f_values_np, final_c1, final_loss_val, hypers.num_intervals


# EVOLVE-BLOCK-END
# EVOLVE-BLOCK-START
import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    """Hyperparameters for the optimization process."""

    num_intervals: int = 800
    learning_rate: float = 0.003
    end_lr_factor: float = 1e-5
    num_steps: int = 80000
    warmup_steps: int = 4000


class AutocorrelationOptimizer:
    """
    Optimizes a discretized function to find the minimal C1 constant.
    Uses multi-start optimization with refinement.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5  # [-1/4, 1/4]
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """
        Computes C1 = max_t (f★f)(t) / (∫f)².
        Uses smooth max over autoconvolution values for better gradient flow.
        """
        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx
        eps = 1e-10
        integral_f_safe = jnp.maximum(integral_f, eps)

        N = self.hypers.num_intervals
        # Pad to 2N for linear convolution (autoconvolution length = 2N-1)
        padded_f = jnp.pad(f_non_negative, (0, N))

        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real * self.dx

        # Smooth max (log-sum-exp) with adaptive sharpness
        sharpness = 200.0
        max_conv_est = jax.scipy.special.logsumexp(sharpness * conv_f_f) / sharpness
        # Ensure we don't underestimate the true max (for validity of upper bound)
        true_max = jnp.max(conv_f_f)
        max_conv = jnp.maximum(max_conv_est, true_max)

        return max_conv / (integral_f_safe ** 2)

    def train_step(self, state: tuple) -> tuple:
        f_values, opt_state = state
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return (f_values, opt_state), loss

    def _optimize_from(self, init_f: jnp.ndarray) -> tuple:
        """Run optimization from a given initial guess."""
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * self.hypers.end_lr_factor,
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        opt_state = self.optimizer.init(init_f)

        train_step_jit = jax.jit(self.train_step)
        state = (init_f, opt_state)

        best_f = init_f
        best_loss = jnp.inf

        for step in range(self.hypers.num_steps):
            state, loss = train_step_jit(state)
            if loss < best_loss:
                best_loss = loss
                best_f = state[0]
            if step % 4000 == 0 or step == self.hypers.num_steps - 1:
                print(f"  Step {step:6d} | C1 ≈ {float(loss):.8f} | Best: {float(best_loss):.8f}")

        return jax.nn.relu(best_f), best_loss

    def run_optimization(self):
        """Multi-start optimization with diverse initial guesses."""
        N = self.hypers.num_intervals
        x = jnp.linspace(-0.25, 0.25, N, endpoint=False)
        key = jax.random.PRNGKey(123)

        initial_guesses = []

        # 1. Smooth cosine bump (centered)
        guess1 = jnp.cos(jnp.pi * x / 0.25) ** 4
        guess1 = guess1 * (jnp.abs(x) <= 0.25)
        initial_guesses.append(guess1)

        # 2. Two-peak symmetric structure (often optimal for this problem)
        peak1 = jnp.exp(-((x - 0.12) ** 2) / (2 * 0.04 ** 2))
        peak2 = jnp.exp(-((x + 0.12) ** 2) / (2 * 0.04 ** 2))
        guess2 = peak1 + peak2
        initial_guesses.append(guess2)

        # 3. Flat center + decaying edges
        guess3 = jnp.ones(N) * 0.5
        guess3 = guess3 * (jnp.abs(x) <= 0.15) + 0.1 * jnp.exp(-(jnp.abs(x) - 0.15) ** 2 / 0.01) * (jnp.abs(x) > 0.15)
        initial_guesses.append(guess3)

        # 4. Trapezoidal-like
        ramp_up = jnp.clip((x + 0.2) / 0.1, 0.0, 1.0)
        ramp_down = jnp.clip((0.2 - x) / 0.1, 0.0, 1.0)
        flat = (jnp.abs(x) <= 0.1).astype(float)
        guess4 = flat + ramp_up * (x < 0) * (x > -0.2) + ramp_down * (x > 0) * (x < 0.2)
        initial_guesses.append(guess4)

        # 5. Random perturbation around uniform
        for _ in range(2):
            key, subkey = jax.random.split(key)
            guess5 = 1.0 + 0.3 * jax.random.normal(subkey, (N,))
            guess5 = jax.nn.relu(guess5)
            initial_guesses.append(guess5)

        print(f"N={N}, Steps={self.hypers.num_steps}, {len(initial_guesses)} starting points")

        overall_best_f = initial_guesses[0]
        overall_best_loss = jnp.inf

        for i, init_guess in enumerate(initial_guesses):
            print(f"\n=== Run {i+1}/{len(initial_guesses)} ===")
            # Normalize initial guess to have integral ~1
            init_norm = init_guess / (jnp.sum(init_guess) * self.dx + 1e-12)
            # Add small randomization
            key, subkey = jax.random.split(key)
            init_norm = init_norm * (1.0 + 0.01 * jax.random.normal(subkey, init_norm.shape))
            init_norm = jax.nn.relu(init_norm)

            f_opt, loss_opt = self._optimize_from(init_norm)
            c1_actual = float(self._compute_exact_c1(f_opt))
            print(f"  Run {i+1} finished. Exact C1: {c1_actual:.10f}")

            if c1_actual < float(overall_best_loss):
                overall_best_loss = c1_actual
                overall_best_f = f_opt
                print(f"  ** New global best: {overall_best_loss:.10f}")

        print(f"\nFinal best C1 across all runs: {overall_best_loss:.10f}")
        return overall_best_f, jnp.array(overall_best_loss)

    def _compute_exact_c1(self, f_values: jnp.ndarray) -> jnp.ndarray:
        """Compute the exact C1 using true max (for final evaluation)."""
        f_non_negative = jax.nn.relu(f_values)
        integral_f = jnp.sum(f_non_negative) * self.dx
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_non_negative, (0, N))
        conv = jnp.fft.ifft(jnp.fft.fft(padded_f) ** 2).real * self.dx
        return jnp.max(conv) / (integral_f ** 2)


def run():
    """Entry point."""
    hypers = Hyperparameters()
    optimizer = AutocorrelationOptimizer(hypers)

    optimized_f, final_loss_val = optimizer.run_optimization()

    # Final numerical validation using higher-resolution convolution
    f_np = np.array(optimized_f)
    dx = optimizer.dx
    integral = np.sum(f_np) * dx
    if integral <= 0:
        f_np = np.ones_like(f_np)
        integral = np.sum(f_np) * dx

    # High-accuracy autoconvolution via FFT
    N = len(f_np)
    padded = np.pad(f_np, (0, N))
    conv = np.fft.ifft(np.fft.fft(padded) ** 2).real * dx
    max_conv = np.max(conv)
    final_c1 = float(max_conv / (integral ** 2))

    print(f"\n=== VALIDATION ===")
    print(f"Integral of f: {integral:.6f}")
    print(f"Max autoconvolution: {max_conv:.6f}")
    print(f"FINAL C1 UPPER BOUND: {final_c1:.12f}")
    print(f"Improvement over 1.5052939684401607: {1.5052939684401607 - final_c1:.12f}")

    return f_np, final_c1, float(final_loss_val), hypers.num_intervals


# EVOLVE-BLOCK-END