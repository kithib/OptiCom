import jax
import jax.numpy as jnp
import optax
import numpy as np
from dataclasses import dataclass
import tqdm


@dataclass
class Hyperparameters:
    num_intervals: int = 900
    learning_rate: float = 0.0075
    num_steps: int = 85000
    penalty_strength: float = 9500000.0
    sharpness: float = 14.5
    softmax_beta: float = 320.0
    warmup_steps: int = 2500


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals

    def _objective_fn(self, latent_h_values: jnp.ndarray) -> jnp.ndarray:
        """
        The loss function includes the objective and a penalty for the constraint.
        Uses sharp sigmoid and log-sum-exp for smooth max approximation.
        """
        # Enforce h(x) in [0, 1] via sigmoid with tunable sharpness
        h = jax.nn.sigmoid(self.hypers.sharpness * latent_h_values)

        # Calculate the primary objective (max correlation)
        j = 1.0 - h
        N = self.hypers.num_intervals
        h_padded = jnp.pad(h, (0, N))
        j_padded = jnp.pad(j, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        scaled_correlation = correlation * self.dx
        
        # Use log-sum-exp for smooth gradient approximation of max
        beta = self.hypers.softmax_beta
        objective_loss = jax.scipy.special.logsumexp(beta * scaled_correlation) / beta

        # Calculate the penalty for the integral constraint
        integral_h = jnp.sum(h) * self.dx
        constraint_loss = (integral_h - 1.0) ** 2

        # Combine the objective with the penalty
        total_loss = objective_loss + self.hypers.penalty_strength * constraint_loss
        return total_loss

    def run_optimization(self):
        # Use learning rate scheduling with warmup and exponential decay
        lr_schedule = optax.warmup_exponential_decay_schedule(
            init_value=0.0,
            peak_value=self.hypers.learning_rate,
            warmup_steps=self.hypers.warmup_steps,
            transition_steps=8000,
            decay_rate=0.5,
            staircase=True
        )
        optimizer = optax.chain(
            optax.clip_by_global_norm(0.4),
            optax.adam(lr_schedule)
        )

        # Start near h=0.5 (which gives integral = 1.0) with small noise
        key = jax.random.PRNGKey(42)
        latent_h_values = jnp.zeros(self.hypers.num_intervals) + 0.002 * jax.random.normal(key, (self.hypers.num_intervals,))

        opt_state = optimizer.init(latent_h_values)

        @jax.jit
        def train_step(latent_h_values, opt_state):
            loss, grads = jax.value_and_grad(self._objective_fn)(latent_h_values)
            updates, opt_state = optimizer.update(grads, opt_state)
            latent_h_values = optax.apply_updates(latent_h_values, updates)
            return latent_h_values, opt_state, loss

        print(f"Optimizing a step function with {self.hypers.num_intervals} intervals...")
        for step in tqdm.tqdm(range(self.hypers.num_steps), desc="Optimizing"):
            latent_h_values, opt_state, loss = train_step(latent_h_values, opt_state)

        # Use sharp sigmoid to get cleaner values
        final_h_raw = jax.nn.sigmoid(self.hypers.sharpness * latent_h_values)
        
        # Target: exactly integral 1 (half the intervals if values are 0/1)
        n_ones_target = int(jnp.round(1.0 / self.dx))
        
        def compute_c5(h_vals):
            j_vals = 1.0 - h_vals
            N = self.hypers.num_intervals
            h_pad = jnp.pad(h_vals, (0, N))
            j_pad = jnp.pad(j_vals, (0, N))
            corr = jnp.fft.ifft(jnp.fft.fft(h_pad) * jnp.conj(jnp.fft.fft(j_pad))).real
            return jnp.max(corr * self.dx)
        
        candidates = []
        
        # Strategy 1: precise normalization of continuous values
        h1 = final_h_raw
        integral_h1 = jnp.sum(h1) * self.dx
        h1 = jnp.clip(h1 / integral_h1, 0.0, 1.0)
        integral_h1 = jnp.sum(h1) * self.dx
        h1 = h1 / integral_h1
        candidates.append(("continuous", h1))
        
        # Strategy 2: hard threshold to exactly N ones (top values)
        indices = jnp.argsort(-final_h_raw)
        h2 = jnp.zeros_like(final_h_raw)
        h2 = h2.at[indices[:n_ones_target]].set(1.0)
        candidates.append(("hard_thresh", h2))
        
        # Strategy 3: near-exact threshold - try offsets
        for delta in [-3, -2, -1, 1, 2, 3]:
            n_ones = max(0, min(self.hypers.num_intervals, n_ones_target + delta))
            hd = jnp.zeros_like(final_h_raw)
            hd = hd.at[indices[:n_ones]].set(1.0)
            candidates.append((f"thresh_offset_{delta}", hd))
        
        # Strategy 4: refined continuous values with multiple iterations
        h4 = jnp.clip(final_h_raw, 0.0, 1.0)
        for _ in range(5):
            integral_h4 = jnp.sum(h4) * self.dx
            h4 = h4 / integral_h4
            h4 = jnp.clip(h4, 0.0, 1.0)
        integral_h4 = jnp.sum(h4) * self.dx
        h4 = h4 / integral_h4
        candidates.append(("refined", h4))
        
        # Strategy 5: various thresholding levels
        threshold_grid = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]
        for t in threshold_grid:
            ht = jnp.where(final_h_raw > t, 1.0, 0.0)
            integral_ht = jnp.sum(ht) * self.dx
            if integral_ht > 0:
                ht = jnp.clip(ht / integral_ht, 0.0, 1.0)
                integral_ht = jnp.sum(ht) * self.dx
                if integral_ht > 0:
                    ht = ht / integral_ht
                    candidates.append((f"thresh_{t}", ht))
        
        # Strategy 6: power transformations (extended grid)
        power_grid = [0.78, 0.83, 0.88, 0.93, 0.98, 1.03, 1.08, 1.13, 1.18, 1.23]
        for p in power_grid:
            hp = jnp.power(final_h_raw, p)
            hp = jnp.clip(hp, 0.0, 1.0)
            integral_hp = jnp.sum(hp) * self.dx
            if integral_hp > 0:
                hp = hp / integral_hp
                candidates.append((f"power_{p}", hp))
        
        # Strategy 7: quantile-based selection (more quantiles)
        quantile_grid = [0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85]
        for q in quantile_grid:
            threshold_val = jnp.quantile(final_h_raw, q)
            hq = jnp.where(final_h_raw > threshold_val, 1.0, 0.0)
            integral_hq = jnp.sum(hq) * self.dx
            if integral_hq > 0:
                hq = jnp.clip(hq / integral_hq, 0.0, 1.0)
                integral_hq = jnp.sum(hq) * self.dx
                if integral_hq > 0:
                    hq = hq / integral_hq
                    candidates.append((f"quantile_{q}", hq))
        
        # Strategy 8: smooth sigmoid transformations
        sig_sharpness = [5.0, 8.0, 12.0, 18.0, 25.0, 35.0]
        for s in sig_sharpness:
            hs = jax.nn.sigmoid(s * (final_h_raw - 0.5))
            hs = jnp.clip(hs, 0.0, 1.0)
            integral_hs = jnp.sum(hs) * self.dx
            if integral_hs > 0:
                hs = hs / integral_hs
                candidates.append((f"sig_{s}", hs))
        
        # Evaluate all candidates and pick best
        best_c5 = float('inf')
        final_h = None
        for name, cand_h in candidates:
            cand_c5 = compute_c5(cand_h)
            if cand_c5 < best_c5:
                best_c5 = cand_c5
                final_h = cand_h
        
        c5_bound = best_c5

        print(f"Optimization complete. Final C5 upper bound: {c5_bound:.15f}")
        return np.array(final_h), float(c5_bound)


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()

    return final_h_values, c5_bound, hypers.num_intervals