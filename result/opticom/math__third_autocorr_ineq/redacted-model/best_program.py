import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass


@dataclass
class Hyperparameters:
    num_intervals: int = 1800
    learning_rate: float = 0.030
    num_steps: int = 200000
    warmup_steps: int = 10000


class C3Optimizer:
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        integral_f = jnp.sum(f_values) * self.dx
        integral_f_sq_safe = jnp.maximum(integral_f ** 2, 1e-12)
        n = self.hypers.num_intervals
        padded_f = jnp.pad(f_values, (n // 2, n // 2))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real * self.dx
        return jnp.max(jnp.abs(conv_f_f)) / integral_f_sq_safe

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def _exact_c3_np(self, f_values_np: np.ndarray) -> float:
        integral = float(np.sum(f_values_np) * self.dx)
        if abs(integral) < 1e-9:
            return float("inf")
        conv = np.convolve(f_values_np, f_values_np, mode="full") * self.dx
        return float(np.max(np.abs(conv)) / (integral * integral))

    def run_optimization(self):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 6e-6,
        )
        self.optimizer = optax.chain(
            optax.clip_by_global_norm(0.55),
            optax.adamw(learning_rate=schedule, b1=0.91, b2=0.998, weight_decay=1.8e-6),
        )

        key = jax.random.PRNGKey(42)
        x = jnp.linspace(-1.0, 1.0, self.hypers.num_intervals)
        components = [
            jnp.sin(3.0 * jnp.pi * x) * jnp.exp(-3.45 * x ** 2),
            0.465 * jnp.sin(7.0 * jnp.pi * x) * jnp.exp(-4.35 * x ** 2),
            0.23 * jnp.cos(1.0 * jnp.pi * x) * jnp.exp(-1.85 * x ** 2),
            -0.185 * jnp.sin(5.0 * jnp.pi * x) * jnp.exp(-3.78 * x ** 2),
            0.108 * jnp.sin(9.0 * jnp.pi * x) * jnp.exp(-4.95 * x ** 2),
            -0.063 * jnp.cos(11.0 * jnp.pi * x) * jnp.exp(-5.55 * x ** 2),
            0.038 * jnp.sin(13.0 * jnp.pi * x) * jnp.exp(-6.15 * x ** 2),
            -0.022 * jnp.cos(15.0 * jnp.pi * x) * jnp.exp(-6.78 * x ** 2),
            0.011 * jnp.sin(17.0 * jnp.pi * x) * jnp.exp(-7.45 * x ** 2),
            -0.0048 * jnp.cos(19.0 * jnp.pi * x) * jnp.exp(-8.15 * x ** 2),
            0.088 * jnp.cos(2.0 * jnp.pi * x) * jnp.exp(-2.48 * x ** 2),
            0.0018 * jnp.sin(21.0 * jnp.pi * x) * jnp.exp(-8.85 * x ** 2),
            -0.0009 * jnp.cos(23.0 * jnp.pi * x) * jnp.exp(-9.55 * x ** 2),
        ]
        f_values = sum(components) + jax.random.normal(key, (self.hypers.num_intervals,)) * 0.011

        opt_state = self.optimizer.init(f_values)
        train_step_jit = jax.jit(self.train_step)
        best_loss = jnp.inf
        best_f = f_values
        for _ in range(self.hypers.num_steps):
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if loss < best_loss:
                best_loss = loss
                best_f = f_values

        # Two-step polishing: re-run with halved learning rate using current best as warm start.
        polish_schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate * 0.5,
            warmup_steps=2000,
            decay_steps=40000,
            end_value=self.hypers.learning_rate * 1e-6,
        )
        polish_optimizer = optax.chain(
            optax.clip_by_global_norm(0.4),
            optax.adamw(learning_rate=polish_schedule, b1=0.92, b2=0.9985, weight_decay=1.2e-6),
        )
        f_values = best_f
        opt_state = polish_optimizer.init(f_values)

        def polish_step(f_vals, opt_st):
            l, g = jax.value_and_grad(self._objective_fn)(f_vals)
            u, opt_st = polish_optimizer.update(g, opt_st, f_vals)
            f_vals = optax.apply_updates(f_vals, u)
            return f_vals, opt_st, l

        polish_jit = jax.jit(polish_step)
        for _ in range(42000):
            f_values, opt_state, loss = polish_jit(f_values, opt_state)
            if loss < best_loss:
                best_loss = loss
                best_f = f_values

        best_np = np.array(best_f)
        exact_c3 = self._exact_c3_np(best_np)
        return best_np, exact_c3


def run():
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()
    final_c3_val = optimizer._exact_c3_np(np.array(optimized_f))
    return np.array(optimized_f), float(final_c3_val), float(final_c3_val), hypers.num_intervals
