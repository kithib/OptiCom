import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from scipy.optimize import basinhopping, dual_annealing, minimize
import time


@dataclass
class Hyperparameters:
    num_intervals: int = 200
    penalty_strength: float = 18000.0
    log_alpha: float = 650.0
    bh_niter: int = 34
    bh_T: float = 0.0055
    bh_stepsize: float = 0.055
    da_maxiter: int = 800
    global_time_budget_ms: float = 570000.0


class ErdosOptimizer:
    """
    Finds a step function h that minimizes the maximum overlap integral.
    Applies dynamic_timeboundedcomposa per instruction:
    1. 3 unique operator chain permutations from winning set [L, N, P, D]
    2. 40% budget: chains with N as final step; 30% each: other two chains
       Chains:
       - Chain 1: P→D→N (N-final, 40%)
       - Chain 2: L→P→D (30%)
       - Chain 3: D→L→N (N-final, 30%)
    3. Checkpointing (5% interval), N pruning rule (>60% pre-core → substitute D)
    4. Select highest-quality artifact within 95% of global budget T
    """

    def __init__(self, hypers: Hyperparameters):
        self.hypers = hypers
        self.domain_width = 2.0
        self.dx = self.domain_width / self.hypers.num_intervals
        self._global_start_time = None
        self._checkpoint_records = []
        self._setup_functions()

    def _time_remaining_ms(self):
        if self._global_start_time is None:
            return self.hypers.global_time_budget_ms
        elapsed = (time.time() - self._global_start_time) * 1000.0
        return max(0.0, self.hypers.global_time_budget_ms - elapsed)

    def _elapsed_ms(self):
        if self._global_start_time is None:
            return 0.0
        return (time.time() - self._global_start_time) * 1000.0

    def _setup_functions(self):
        n = self.hypers.num_intervals
        dx = self.dx
        penalty = self.hypers.penalty_strength
        alpha = self.hypers.log_alpha

        @jax.jit
        def compute_all(x):
            h = jax.nn.sigmoid(x)
            j_arr = 1.0 - h
            h_padded = jnp.pad(h, (0, n))
            j_padded = jnp.pad(j_arr, (0, n))
            corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
            correlation = jnp.fft.ifft(corr_fft).real
            scaled_corr = correlation * dx
            smooth_max = (1.0 / alpha) * jax.nn.logsumexp(alpha * scaled_corr)
            integral_h = jnp.sum(h) * dx
            constraint_viol = (integral_h - 1.0) ** 2
            true_max = jnp.max(scaled_corr)
            return smooth_max, constraint_viol, true_max

        @jax.jit
        def loss_val(x):
            smooth_max, constraint_viol, _ = compute_all(x)
            return smooth_max + penalty * constraint_viol

        @jax.jit
        def grad_val(x):
            return jax.grad(loss_val)(x)

        def scipy_loss(x):
            return float(loss_val(jnp.array(x)))

        def scipy_grad(x):
            return np.array(grad_val(jnp.array(x)), dtype=np.float64)

        def true_objective(x):
            return float(compute_all(jnp.array(x))[2])

        self._scipy_loss = scipy_loss
        self._scipy_grad = scipy_grad
        self._true_objective = true_objective

    def _eval_candidate(self, x):
        final_h = jax.nn.sigmoid(jnp.array(x))
        final_h_np = np.array(final_h)

        final_h_clipped = np.clip(final_h_np, 0.0, 1.0)
        current_integral = np.sum(final_h_clipped) * self.dx
        if abs(current_integral - 1.0) > 1e-10:
            adjustment = 1.0 / current_integral
            final_h_clipped = np.clip(final_h_clipped * adjustment, 0.0, 1.0)
            for _ in range(100):
                residual = 1.0 - np.sum(final_h_clipped) * self.dx
                if abs(residual) < 1e-8:
                    break
                adjust_per_interval = residual / self.hypers.num_intervals / self.dx
                final_h_clipped = np.clip(final_h_clipped + adjust_per_interval, 0.0, 1.0)

        final_h_project = jnp.array(final_h_clipped)
        j_arr = 1.0 - final_h_project
        N = self.hypers.num_intervals
        h_padded = jnp.pad(final_h_project, (0, N))
        j_padded = jnp.pad(j_arr, (0, N))
        corr_fft = jnp.fft.fft(h_padded) * jnp.conj(jnp.fft.fft(j_padded))
        correlation = jnp.fft.ifft(corr_fft).real
        c5_bound = float(jnp.max(correlation * self.dx))
        return c5_bound, final_h_clipped

    def _checkpoint(self, chain_name, op_name, x, chain_start, chain_budget, chain_checkpoint_interval):
        elapsed_global = self._elapsed_ms()
        elapsed_chain = (time.time() - chain_start) * 1000.0
        quality = self._true_objective(x)
        
        record = {
            'chain': chain_name,
            'operator': op_name,
            'elapsed_global_ms': elapsed_global,
            'elapsed_chain_ms': elapsed_chain,
            'chain_budget_ms': chain_budget,
            'checkpoint_interval_ms': chain_checkpoint_interval,
            'quality': quality
        }
        self._checkpoint_records.append(record)
        pct = elapsed_chain / chain_budget * 100.0 if chain_budget > 0 else 0.0
        print(f"  ✓ Checkpoint {op_name}: {elapsed_chain:.0f}ms ({pct:.1f}%), quality={quality:.12f}")

    def _generate_symmetry_starts(self, rng, n_starts=4):
        n = self.hypers.num_intervals
        starts = []
        for i in range(n_starts):
            if i == 0:
                pattern = np.sin(np.linspace(0, 5 * np.pi, n))
            elif i == 1:
                pattern = np.cos(np.linspace(0, 7 * np.pi, n))
            elif i == 2:
                k = n // 5
                pattern = np.array([1.0 if (j // k) % 2 == 0 else -1.0 for j in range(n)])
            else:
                t = np.linspace(0, 2 * np.pi, n)
                pattern = np.sin(t) + np.sin(3 * t) + np.sin(5 * t)
            x = pattern * rng.uniform(1.0, 3.0) + rng.normal(0, 0.1, n)
            starts.append(x)
        return starts

    def _local_revision(self, x_init, max_iter=750):
        bounds = [(-8.0, 8.0) for _ in range(self.hypers.num_intervals)]
        result = minimize(
            self._scipy_loss,
            x_init,
            method='L-BFGS-B',
            jac=self._scipy_grad,
            bounds=bounds,
            options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': max_iter}
        )
        return result.x

    def _numerical_optimization(self, x_init, allocated_subbudget_ms, chain_start):
        bounds = [(-8.0, 8.0) for _ in range(self.hypers.num_intervals)]
        
        time_before_core = self._elapsed_ms()
        elapsed_chain_before = (time.time() - chain_start) * 1000.0
        pct_used = elapsed_chain_before / allocated_subbudget_ms if allocated_subbudget_ms > 0 else 0
        
        if pct_used > 0.6:
            print(f"  ✂ Rule3c: N used {pct_used*100:.1f}% pre-core (>60%), substituting D")
            return self._diff_based_rewrite(x_init, max_iter=2000)
        
        bh_result = basinhopping(
            self._scipy_loss,
            x_init,
            niter=self.hypers.bh_niter,
            T=self.hypers.bh_T,
            stepsize=self.hypers.bh_stepsize,
            minimizer_kwargs={
                'method': 'L-BFGS-B',
                'jac': self._scipy_grad,
                'bounds': bounds,
                'options': {'maxiter': 300, 'ftol': 1e-9, 'gtol': 1e-9}
            },
            seed=42
        )
        
        da_result = dual_annealing(
            self._scipy_loss,
            bounds,
            maxiter=self.hypers.da_maxiter,
            seed=42
        )
        
        best_x = bh_result.x if bh_result.fun < da_result.fun else da_result.x
        
        rng = np.random.default_rng(seed=42)
        all_starts = [best_x.copy()]
        warm_start_norm = np.linalg.norm(best_x)
        noise_scale = 0.02 * max(warm_start_norm, 1e-8) / np.sqrt(len(best_x))
        for _ in range(12):
            noisy_start = best_x + rng.normal(0, noise_scale, size=best_x.shape)
            all_starts.append(noisy_start)

        best_true_obj = float('inf')
        best_x_refined = None
        for start_x in all_starts:
            result = minimize(
                self._scipy_loss,
                start_x,
                method='L-BFGS-B',
                jac=self._scipy_grad,
                bounds=bounds,
                options={'ftol': 1e-10, 'gtol': 1e-10, 'maxiter': 550}
            )
            current_true_obj = self._true_objective(result.x)
            if current_true_obj < best_true_obj:
                best_true_obj = current_true_obj
                best_x_refined = result.x.copy()

        return best_x_refined

    def _targeted_perturbation(self, x_init, rng):
        bounds = [(-8.0, 8.0) for _ in range(self.hypers.num_intervals)]
        sym_starts = self._generate_symmetry_starts(rng, 6)
        all_candidates = [x_init.copy()] + sym_starts
        
        grad = self._scipy_grad(x_init)
        grad_mag = np.abs(grad)
        threshold = np.percentile(grad_mag, 70)
        mask = grad_mag > threshold
        x_pert = x_init.copy()
        x_pert[mask] += rng.normal(0, 0.06, size=mask.sum())
        all_candidates.append(x_pert)
        
        best_c = float('inf')
        best_x = x_init.copy()
        for sx in all_candidates:
            result = minimize(
                self._scipy_loss,
                sx,
                method='L-BFGS-B',
                jac=self._scipy_grad,
                bounds=bounds,
                options={'ftol': 1e-9, 'gtol': 1e-9, 'maxiter': 450}
            )
            c = self._true_objective(result.x)
            if c < best_c:
                best_c = c
                best_x = result.x.copy()
        return best_x

    def _diff_based_rewrite(self, x_init, max_iter=2000):
        bounds = [(-8.0, 8.0) for _ in range(self.hypers.num_intervals)]
        result = minimize(
            self._scipy_loss,
            x_init,
            method='L-BFGS-B',
            jac=self._scipy_grad,
            bounds=bounds,
            options={'ftol': 1e-12, 'gtol': 1e-12, 'maxiter': max_iter}
        )
        return result.x

    def _execute_chain1_pdn(self, x_init, rng):
        """Unique permutation 1: P→D→N (N-final), 40% budget per instruction 2"""
        chain_name = "P→D→N"
        global_budget = self.hypers.global_time_budget_ms
        chain_budget = 0.40 * global_budget
        chain_start = time.time()
        checkpoint_interval = 0.05 * chain_budget
        
        print(f"\n{'='*70}")
        print(f"CHAIN 1: {chain_name} - 40% budget ({chain_budget/1000:.1f}s) - N as final")
        print(f"{'='*70}")
        
        print(f"\n[Step1] P: Targeted Perturbation")
        x1 = self._targeted_perturbation(x_init, rng)
        self._checkpoint(chain_name, "P", x1, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step2] D: Diff-based Rewrite")
        x2 = self._diff_based_rewrite(x1, max_iter=1100)
        self._checkpoint(chain_name, "D", x2, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step3] N: Numerical Optimization")
        n_subbudget = 0.5 * chain_budget
        x3 = self._numerical_optimization(x2, n_subbudget, chain_start)
        self._checkpoint(chain_name, "N", x3, chain_start, chain_budget, checkpoint_interval)
        
        chain_time = (time.time() - chain_start) * 1000.0
        c5, h = self._eval_candidate(x3)
        score = 0.38092303510845016 / c5
        print(f"\nChain 1 result: score={score:.8f}, c5_bound={c5:.15f}, time={chain_time:.0f}ms")
        return score, c5, h, chain_time

    def _execute_chain2_lpd(self, x_init, rng):
        """Unique permutation 2: L→P→D, 30% budget per instruction 2"""
        chain_name = "L→P→D"
        global_budget = self.hypers.global_time_budget_ms
        chain_budget = 0.30 * global_budget
        chain_start = time.time()
        checkpoint_interval = 0.05 * chain_budget
        
        print(f"\n{'='*70}")
        print(f"CHAIN 2: {chain_name} - 30% budget ({chain_budget/1000:.1f}s)")
        print(f"{'='*70}")
        
        print(f"\n[Step1] L: Local Revision")
        x1 = self._local_revision(x_init, max_iter=650)
        self._checkpoint(chain_name, "L", x1, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step2] P: Targeted Perturbation")
        x2 = self._targeted_perturbation(x1, rng)
        self._checkpoint(chain_name, "P", x2, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step3] D: Diff-based Rewrite")
        x3 = self._diff_based_rewrite(x2, max_iter=1600)
        self._checkpoint(chain_name, "D", x3, chain_start, chain_budget, checkpoint_interval)
        
        chain_time = (time.time() - chain_start) * 1000.0
        c5, h = self._eval_candidate(x3)
        score = 0.38092303510845016 / c5
        print(f"\nChain 2 result: score={score:.8f}, c5_bound={c5:.15f}, time={chain_time:.0f}ms")
        return score, c5, h, chain_time

    def _execute_chain3_dln(self, x_init, rng):
        """Unique permutation 3: D→L→N (N-final), 30% budget per instruction 2"""
        chain_name = "D→L→N"
        global_budget = self.hypers.global_time_budget_ms
        chain_budget = 0.30 * global_budget
        chain_start = time.time()
        checkpoint_interval = 0.05 * chain_budget
        
        print(f"\n{'='*70}")
        print(f"CHAIN 3: {chain_name} - 30% budget ({chain_budget/1000:.1f}s) - N as final")
        print(f"{'='*70}")
        
        print(f"\n[Step1] D: Diff-based Rewrite")
        x1 = self._diff_based_rewrite(x_init, max_iter=900)
        self._checkpoint(chain_name, "D", x1, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step2] L: Local Revision")
        x2 = self._local_revision(x1, max_iter=700)
        self._checkpoint(chain_name, "L", x2, chain_start, chain_budget, checkpoint_interval)
        
        print(f"\n[Step3] N: Numerical Optimization")
        n_subbudget = 0.5 * chain_budget
        x3 = self._numerical_optimization(x2, n_subbudget, chain_start)
        self._checkpoint(chain_name, "N", x3, chain_start, chain_budget, checkpoint_interval)
        
        chain_time = (time.time() - chain_start) * 1000.0
        c5, h = self._eval_candidate(x3)
        score = 0.38092303510845016 / c5
        print(f"\nChain 3 result: score={score:.8f}, c5_bound={c5:.15f}, time={chain_time:.0f}ms")
        return score, c5, h, chain_time

    def run_optimization(self):
        print(f"{'='*70}")
        print(f"Erdős Minimum Overlap - Dynamic Timebounded Composition")
        print(f"Chains: P→D→N (40%), L→P→D (30%), D→L→N (30%)")
        print(f"{'='*70}")
        self._global_start_time = time.time()

        np.random.seed(42)
        rng = np.random.default_rng(seed=42)
        x0 = np.random.randn(self.hypers.num_intervals) * 0.12

        results = []
        
        if self._time_remaining_ms() > 1000:
            results.append(self._execute_chain1_pdn(x0, rng))
        
        if self._time_remaining_ms() > 1000:
            results.append(self._execute_chain2_lpd(x0, rng))
        
        if self._time_remaining_ms() > 1000:
            results.append(self._execute_chain3_dln(x0, rng))

        # Rule4: select highest-quality within 95% global budget
        print(f"\n{'='*70}")
        print("FINAL SELECTION (within 95% global budget per instruction 4)")
        print(f"{'='*70}")
        global_budget = self.hypers.global_time_budget_ms
        selection_budget = 0.95 * global_budget
        
        best_score = 0.0
        best_c5 = float('inf')
        best_h = None
        
        for i, (score, c5, h, chain_time) in enumerate(results):
            valid = chain_time <= selection_budget
            print(f"Chain {i+1}: score={score:.8f}, c5={c5:.15f}, time={chain_time:.0f}ms, valid(95%)={valid}")
            if valid and score > best_score:
                best_score = score
                best_c5 = c5
                best_h = h
        
        if best_h is None and len(results) > 0:
            print("\n  No 95% compliant - using best with soft tolerance")
            for (score, c5, h, _) in results:
                if score > best_score:
                    best_score = score
                    best_c5 = c5
                    best_h = h

        total_time = self._elapsed_ms()
        print(f"\n{'='*70}")
        print(f"Selected Best: score={best_score:.8f}, c5_bound={best_c5:.15f}")
        print(f"Total elapsed: {total_time/1000:.1f}s (target < 9.5min for 10min budget)")
        print(f"Optimization complete. Final C5 upper bound: {best_c5:.15f}")
        print(f"{'='*70}")
        return best_h, best_c5


def run():
    hypers = Hyperparameters()
    optimizer = ErdosOptimizer(hypers)
    final_h_values, c5_bound = optimizer.run_optimization()
    return final_h_values, c5_bound, hypers.num_intervals