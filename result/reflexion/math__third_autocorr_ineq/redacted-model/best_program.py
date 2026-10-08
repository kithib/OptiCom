import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import time


@dataclass
class Hyperparameters:
    num_intervals: int = 4800
    learning_rate: float = 0.018
    num_steps: int = 480000
    warmup_steps: int = 48000
    timeout_s: int = 250


class C3Optimizer:
    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 0.5
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, f_values: jnp.ndarray) -> jnp.ndarray:
        integral_f = jnp.sum(f_values) * self.dx
        eps = 1e-20
        integral_f_sq_safe = jnp.maximum(integral_f**2, eps)
        N = self.hypers.num_intervals
        padded_f = jnp.pad(f_values, (0, N))
        fft_f = jnp.fft.fft(padded_f)
        conv_f_f = jnp.fft.ifft(fft_f * fft_f).real
        scaled_conv_f_f = conv_f_f * self.dx
        max_abs_conv = jnp.max(jnp.abs(scaled_conv_f_f))
        c3_ratio = max_abs_conv / integral_f_sq_safe
        return c3_ratio

    def train_step(self, f_values: jnp.ndarray, opt_state: optax.OptState) -> tuple:
        loss, grads = jax.value_and_grad(self._objective_fn)(f_values)
        updates, opt_state = self.optimizer.update(grads, opt_state, f_values)
        f_values = optax.apply_updates(f_values, updates)
        return f_values, opt_state, loss

    def run_optimization(self):
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            decay_steps=self.hypers.num_steps - self.hypers.warmup_steps,
            end_value=self.hypers.learning_rate * 1e-9,
        )
        self.optimizer = optax.adam(learning_rate=schedule)
        key = jax.random.PRNGKey(42)
        f_values = jax.random.normal(key, (self.hypers.num_intervals,)) * 0.1
        opt_state = self.optimizer.init(f_values)
        print(f"N={self.hypers.num_intervals}, Steps={self.hypers.num_steps}, Timeout={self.hypers.timeout_s}s")
        train_step_jit = jax.jit(self.train_step)
        best_f = f_values
        best_loss = jnp.inf
        loss = jnp.inf
        start_time = time.time()
        for step in range(self.hypers.num_steps):
            if (step % 8000 == 0) and (time.time() - start_time > self.hypers.timeout_s):
                print(f"[Stop] Approaching time limit after {time.time()-start_time:.1f}s at step {step}")
                break
            f_values, opt_state, loss = train_step_jit(f_values, opt_state)
            if loss < best_loss:
                best_loss = loss
                best_f = f_values
            if step % 40000 == 0 or step == self.hypers.num_steps - 1:
                print(f"Step {step:7d} | C3≈{loss:.18f} | Best≈{best_loss:.18f} | T={time.time()-start_time:.0f}s")
        return best_f, best_loss


def run():
    hypers = Hyperparameters()
    optimizer = C3Optimizer(hypers)
    optimized_f, final_c3_val = optimizer.run_optimization()
    loss_val = final_c3_val
    f_values_np = np.array(optimized_f)
    return f_values_np, float(final_c3_val), float(loss_val), hypers.num_intervals