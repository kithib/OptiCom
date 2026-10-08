GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    def place_with_sort(sorted_models):
        loc_placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        loc_shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        loc_weighted = [0.0 for _ in range(gpu_num)]

        for model in sorted_models:
            best_idx = None
            best_post_kvpr = float('inf')
            model_weight = model.req_rate / model.slo
            model_size = model.model_size

            # First pass: prefer GPUs with remaining memory > 0 after placement
            for gpu_id in range(gpu_num):
                if model_size <= loc_shared_kv[gpu_id]:
                    remaining_after = loc_shared_kv[gpu_id] - model_size
                    if remaining_after > 0:
                        post_kvpr = (loc_weighted[gpu_id] + model_weight) / remaining_after
                        if post_kvpr < best_post_kvpr:
                            best_post_kvpr = post_kvpr
                            best_idx = gpu_id

            # Fallback: allow remaining_after == 0 if no better option
            if best_idx is None:
                fallback_best_idx = None
                fallback_best_ratio = float('inf')
                for gpu_id in range(gpu_num):
                    if model_size <= loc_shared_kv[gpu_id]:
                        remaining_after = loc_shared_kv[gpu_id] - model_size
                        current_ratio = (loc_weighted[gpu_id] + model_weight) / remaining_after if remaining_after > 0 else float('inf')
                        if current_ratio < fallback_best_ratio:
                            fallback_best_ratio = current_ratio
                            fallback_best_idx = gpu_id
                best_idx = fallback_best_idx

            if best_idx is None:
                raise ValueError(
                    f"Unable to place model of size {model.model_size} GB on any GPU."
                )

            loc_placement[best_idx].append(model)
            loc_weighted[best_idx] += model_weight
            loc_shared_kv[best_idx] -= model_size

        return loc_placement

    def compute_max_kvpr(placement):
        max_kvpr = 0.0
        for gpu_id in placement:
            gpu_models = placement[gpu_id]
            if not gpu_models:
                continue
            total_weight = sum(m.req_rate / m.slo for m in gpu_models)
            total_size = sum(m.model_size for m in gpu_models)
            remaining = GPU_MEM_SIZE - total_size
            if remaining <= 0:
                return float('inf')
            kvpr = total_weight / remaining
            if kvpr > max_kvpr:
                max_kvpr = kvpr
        return max_kvpr

    def compute_total_kvpr(placement):
        total = 0.0
        for gpu_id in placement:
            gpu_models = placement[gpu_id]
            if not gpu_models:
                continue
            total_weight = sum(m.req_rate / m.slo for m in gpu_models)
            total_size = sum(m.model_size for m in gpu_models)
            remaining = GPU_MEM_SIZE - total_size
            if remaining <= 0:
                return float('inf')
            total += total_weight / remaining
        return total

    def swap_improve(placement):
        """Try swapping pairs of models between GPUs to reduce max KVPR."""
        improved = True
        current_best = compute_max_kvpr(placement)
        current_total = compute_total_kvpr(placement)
        gpu_ids = list(placement.keys())
        
        while improved:
            improved = False
            for i in range(len(gpu_ids)):
                for j in range(i + 1, len(gpu_ids)):
                    gpu_i = gpu_ids[i]
                    gpu_j = gpu_ids[j]
                    models_i = placement[gpu_i]
                    models_j = placement[gpu_j]
                    
                    # Try all swaps between model from i and model from j
                    for mi_idx in range(len(models_i)):
                        for mj_idx in range(len(models_j)):
                            # Perform swap
                            mi = models_i[mi_idx]
                            mj = models_j[mj_idx]
                            models_i[mi_idx], models_j[mj_idx] = mj, mi
                            
                            # Check memory constraints
                            size_i = sum(m.model_size for m in models_i)
                            size_j = sum(m.model_size for m in models_j)
                            if size_i > GPU_MEM_SIZE or size_j > GPU_MEM_SIZE:
                                # Revert swap
                                models_i[mi_idx], models_j[mj_idx] = mi, mj
                                continue
                            
                            # Check improvement
                            new_max = compute_max_kvpr(placement)
                            new_total = compute_total_kvpr(placement)
                            if (new_max < current_best) or (new_max == current_best and new_total < current_total):
                                current_best = new_max
                                current_total = new_total
                                improved = True
                            else:
                                # Revert swap
                                models_i[mi_idx], models_j[mj_idx] = mi, mj

            # Try moving a single model from one GPU to another
            for src in gpu_ids:
                for dst in gpu_ids:
                    if src == dst:
                        continue
                    src_models = placement[src]
                    dst_models = placement[dst]
                    for m_idx in range(len(src_models)):
                        m = src_models[m_idx]
                        # Try move
                        dst_models.append(m)
                        src_models.pop(m_idx)
                        
                        size_dst = sum(x.model_size for x in dst_models)
                        if size_dst > GPU_MEM_SIZE:
                            # Revert move
                            src_models.insert(m_idx, m)
                            dst_models.pop()
                            continue
                        
                        new_max = compute_max_kvpr(placement)
                        new_total = compute_total_kvpr(placement)
                        if (new_max < current_best) or (new_max == current_best and new_total < current_total):
                            current_best = new_max
                            current_total = new_total
                            improved = True
                            break  # restart iteration after modification
                        else:
                            # Revert move
                            src_models.insert(m_idx, m)
                            dst_models.pop()
                    if improved:
                        break
                if improved:
                    break

        return placement

    # Expanded, high-resolution sorting strategies
    sort_keys = [
        lambda m: (m.req_rate / m.slo) * m.model_size,
        lambda m: m.req_rate / m.slo,
        lambda m: m.model_size,
        lambda m: -(m.req_rate / m.slo) / (m.model_size + 1e-9),
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 0.5),
        lambda m: -m.model_size / (m.req_rate / m.slo + 1e-9),
        lambda m: (m.req_rate / m.slo) / (m.model_size + 1e-9),
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 2),
        lambda m: (m.req_rate / m.slo) ** 1.5 * (m.model_size ** 1.5),
        lambda m: -(m.req_rate / m.slo) ** 2 / (m.model_size + 1e-9),
        lambda m: (m.model_size ** 1.5) / ((m.req_rate / m.slo) + 1e-9),
        lambda m: (m.req_rate / m.slo + 1e-9) * (m.model_size ** 1.2),
        lambda m: -1 / ((m.req_rate / m.slo) * m.model_size + 1e-9),
        lambda m: (m.req_rate / m.slo) ** 2 * m.model_size,
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 1.8),
        lambda m: (m.req_rate / m.slo) ** 0.5 * m.model_size,
        lambda m: (m.req_rate / m.slo) ** 1.5 * m.model_size,
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 2.5),
        lambda m: (m.req_rate / m.slo) ** 2.5 * m.model_size,
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 0.8),
        lambda m: (m.req_rate / m.slo) + (m.model_size * 2),
        lambda m: (m.req_rate / m.slo) * 3 + m.model_size,
        lambda m: m.model_size - (m.req_rate / m.slo),
        lambda m: (m.req_rate / m.slo) / (m.model_size ** 0.5 + 1e-9),
        lambda m: ((m.req_rate / m.slo) ** 2) / (m.model_size ** 0.5 + 1e-9),
        lambda m: (m.req_rate / m.slo) ** 3 * m.model_size,
        lambda m: (m.req_rate / m.slo) * (m.model_size ** 3),
        lambda m: (m.req_rate / m.slo) ** 0.8 * (m.model_size ** 1.2),
        lambda m: (m.req_rate / m.slo) ** 1.2 * (m.model_size ** 0.8),
    ]

    best_placement = None
    best_max_kvpr = float('inf')
    best_total_kvpr = float('inf')
    placed = False

    # Try each sorting key in both descending and ascending order
    for key in sort_keys:
        try:
            sorted_models = sorted(models, key=key, reverse=True)
            candidate = place_with_sort(sorted_models)
            candidate = swap_improve(candidate)
            candidate_max_kvpr = compute_max_kvpr(candidate)
            candidate_total_kvpr = compute_total_kvpr(candidate)
            if (candidate_max_kvpr < best_max_kvpr) or (candidate_max_kvpr == best_max_kvpr and candidate_total_kvpr < best_total_kvpr):
                best_max_kvpr = candidate_max_kvpr
                best_total_kvpr = candidate_total_kvpr
                best_placement = candidate
                placed = True
        except ValueError:
            continue
        try:
            sorted_models_rev = sorted(models, key=key, reverse=False)
            candidate_rev = place_with_sort(sorted_models_rev)
            candidate_rev = swap_improve(candidate_rev)
            candidate_rev_max_kvpr = compute_max_kvpr(candidate_rev)
            candidate_rev_total_kvpr = compute_total_kvpr(candidate_rev)
            if (candidate_rev_max_kvpr < best_max_kvpr) or (candidate_rev_max_kvpr == best_max_kvpr and candidate_rev_total_kvpr < best_total_kvpr):
                best_max_kvpr = candidate_rev_max_kvpr
                best_total_kvpr = candidate_rev_total_kvpr
                best_placement = candidate_rev
                placed = True
        except ValueError:
            continue

    if not placed:
        raise ValueError("Unable to place all models across GPUs with available memory.")

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