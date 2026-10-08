GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses multiple greedy seeds, a robust local search, and targeted refinement.
    """
    if gpu_num <= 0:
        raise ValueError("Number of GPUs must be positive")
    if not models:
        return {gpu_id: [] for gpu_id in range(gpu_num)}
    if gpu_num == 1:
        total_size = sum(m.model_size for m in models)
        if total_size >= GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place all models on a single GPU: total size {total_size} GB >= {GPU_MEM_SIZE} GB"
            )
        return {0: list(models)}

    # --- Precompute per-model metrics once for performance + validation ---
    model_metrics = []
    for model in models:
        if model.slo <= 0:
            raise ValueError("Model SLO must be positive")
        if model.model_size <= 0 or model.model_size > GPU_MEM_SIZE:
            raise ValueError(f"Invalid model size: {model.model_size}")
        kv_contrib = model.req_rate / model.slo
        model_metrics.append((kv_contrib, model.model_size, model))

    def eval_max_kvpr(placement):
        """Compute the maximum KVPR across GPUs for a given placement."""
        current_max = 0.0
        for g in range(gpu_num):
            gpus_mods = placement[g]
            if not gpus_mods:
                continue
            total_kv = 0.0
            total_size = 0
            for m in gpus_mods:
                total_kv += m.req_rate / m.slo
                total_size += m.model_size
            remaining = GPU_MEM_SIZE - total_size
            if remaining <= 0:
                return float('inf')
            kvpr = total_kv / remaining
            if kvpr > current_max:
                current_max = kvpr
        return current_max

    def build_greedy_placement(sorted_metrics, prefer_global=True):
        """
        Greedy placement that minimizes the resulting global maximum KVPR.
        If prefer_global is True, we strictly minimize the global max.
        Otherwise we minimize the per-GPU KVPR of the target GPU.
        """
        placement = {g: [] for g in range(gpu_num)}
        remaining_mem = [GPU_MEM_SIZE] * gpu_num
        total_kv = [0.0] * gpu_num
        current_kvprs = [0.0] * gpu_num
        current_global_max = 0.0

        for kv, sz, mdl in sorted_metrics:
            best_idx = None
            best_global_max = float('inf')
            best_post_kvpr = float('inf')

            for g in range(gpu_num):
                post_mem = remaining_mem[g] - sz
                if post_mem <= 0:
                    continue
                post_total_kv = total_kv[g] + kv
                post_kvpr = post_total_kv / post_mem

                # Max KVPR among OTHER GPUs
                if current_kvprs[g] < current_global_max:
                    max_other = current_global_max
                else:
                    max_other = 0.0
                    for o in range(gpu_num):
                        if o != g and current_kvprs[o] > max_other:
                            max_other = current_kvprs[o]

                global_max = post_kvpr if post_kvpr > max_other else max_other

                if prefer_global:
                    if global_max < best_global_max or (
                        global_max == best_global_max and post_kvpr < best_post_kvpr
                    ):
                        best_global_max = global_max
                        best_post_kvpr = post_kvpr
                        best_idx = g
                else:
                    if post_kvpr < best_post_kvpr or (
                        post_kvpr == best_post_kvpr and global_max < best_global_max
                    ):
                        best_post_kvpr = post_kvpr
                        best_global_max = global_max
                        best_idx = g

            if best_idx is None:
                return None, float('inf')

            placement[best_idx].append(mdl)
            total_kv[best_idx] += kv
            remaining_mem[best_idx] -= sz
            current_kvprs[best_idx] = total_kv[best_idx] / remaining_mem[best_idx]
            if current_kvprs[best_idx] > current_global_max:
                current_global_max = current_kvprs[best_idx]

        return placement, eval_max_kvpr(placement)

    def local_search(seed_placement, seed_score):
        """
        Thorough local search: sequential moves, swaps, and re-balancing passes.
        """
        best_placement = {g: list(seed_placement[g]) for g in range(gpu_num)}
        current_score = seed_score

        # --- Move pass ---
        for outer in range(12):
            improved = False

            # Moves: move a single model from one GPU to another
            for g1 in range(gpu_num):
                if not best_placement[g1]:
                    continue
                g1_total = sum(m.model_size for m in best_placement[g1])
                for i1 in range(len(best_placement[g1])):
                    m1 = best_placement[g1][i1]
                    g1_after = g1_total - m1.model_size
                    if g1_after >= GPU_MEM_SIZE:
                        continue
                    for g2 in range(gpu_num):
                        if g2 == g1:
                            continue
                        g2_total = sum(m.model_size for m in best_placement[g2])
                        g2_after = g2_total + m1.model_size
                        if g2_after >= GPU_MEM_SIZE:
                            continue
                        new_placement = {gg: list(ll) for gg, ll in best_placement.items()}
                        new_placement[g1].pop(i1)
                        new_placement[g2].append(m1)
                        new_score = eval_max_kvpr(new_placement)
                        if new_score < current_score - 1e-12:
                            best_placement = new_placement
                            current_score = new_score
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if improved:
                continue

            # Swaps: swap two models between two GPUs
            for g1 in range(gpu_num):
                if not best_placement[g1]:
                    continue
                for g2 in range(g1 + 1, gpu_num):
                    if not best_placement[g2]:
                        continue
                    g1_total = sum(m.model_size for m in best_placement[g1])
                    g2_total = sum(m.model_size for m in best_placement[g2])
                    for i1 in range(len(best_placement[g1])):
                        m1 = best_placement[g1][i1]
                        for i2 in range(len(best_placement[g2])):
                            m2 = best_placement[g2][i2]
                            g1_after = g1_total - m1.model_size + m2.model_size
                            g2_after = g2_total - m2.model_size + m1.model_size
                            if g1_after >= GPU_MEM_SIZE or g2_after >= GPU_MEM_SIZE:
                                continue
                            new_placement = {gg: list(ll) for gg, ll in best_placement.items()}
                            new_placement[g1][i1], new_placement[g2][i2] = (
                                new_placement[g2][i2],
                                new_placement[g1][i1],
                            )
                            new_score = eval_max_kvpr(new_placement)
                            if new_score < current_score - 1e-12:
                                best_placement = new_placement
                                current_score = new_score
                                improved = True
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            if not improved:
                break

        return best_placement, current_score

    # --- Build multiple seeds using different sort orders and strategies ---
    candidates = []
    sort_orders = [
        sorted(model_metrics, key=lambda x: x[0] * x[1], reverse=True),
        sorted(model_metrics, key=lambda x: x[0], reverse=True),
        sorted(model_metrics, key=lambda x: x[1], reverse=True),
        sorted(model_metrics, key=lambda x: (x[0] / x[1]) if x[1] > 0 else 0, reverse=True),
        sorted(model_metrics, key=lambda x: x[0] + x[1], reverse=True),
        sorted(model_metrics, key=lambda x: x[1] / (x[0] + 1e-9), reverse=True),
    ]

    for sm in sort_orders:
        p1, s1 = build_greedy_placement(sm, prefer_global=True)
        if p1 is not None:
            candidates.append((s1, p1))
        p2, s2 = build_greedy_placement(sm, prefer_global=False)
        if p2 is not None:
            candidates.append((s2, p2))

    if not candidates:
        raise ValueError(
            f"Unable to place models on available GPUs. GPU count: {gpu_num}, model sizes: {[m.model_size for m in models]}"
        )

    candidates.sort()
    top_n = min(5, len(candidates))
    refined = []
    for seed_score, seed_place in candidates[:top_n]:
        final_place, final_score = local_search(seed_place, seed_score)
        refined.append((final_score, final_place))

    refined.sort()
    best_score, best_placement = refined[0]

    return best_placement

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    from evaluator import generate_test_gpu_models
    from evaluator import calculate_kvcache_pressure
    from evaluator import safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):
        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")