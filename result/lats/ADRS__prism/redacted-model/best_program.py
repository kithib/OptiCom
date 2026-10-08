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
    import random
    import itertools

    def assign_models(order, gpu_count):
        """Greedy assignment using projected KVPR to pick target GPU, with tiebreak on remaining memory."""
        placements = {i: [] for i in range(gpu_count)}
        weighted = [0.0] * gpu_count
        remaining = [GPU_MEM_SIZE] * gpu_count

        for m in order:
            best_gpu = None
            best_pressure = float('inf')
            best_remaining = -1.0
            size = m.model_size
            load = m.req_rate / m.slo

            for gpu in range(gpu_count):
                if remaining[gpu] >= size:
                    new_weight = weighted[gpu] + load
                    new_remaining = remaining[gpu] - size
                    if new_remaining <= 0:
                        pressure = float('inf') if new_weight > 0 else 0.0
                    else:
                        pressure = new_weight / new_remaining

                    if pressure < best_pressure:
                        best_pressure = pressure
                        best_gpu = gpu
                        best_remaining = new_remaining
                    elif pressure == best_pressure and new_remaining > best_remaining:
                        best_gpu = gpu
                        best_remaining = new_remaining

            if best_gpu is None:
                return None
            placements[best_gpu].append(m)
            weighted[best_gpu] += load
            remaining[best_gpu] -= size
        return placements

    def compute_max_kvpr(placements):
        if placements is None:
            return float('inf')
        max_pressure = 0.0
        for models_list in placements.values():
            total_load = sum(m.req_rate / m.slo for m in models_list)
            total_size = sum(m.model_size for m in models_list)
            remaining = GPU_MEM_SIZE - total_size
            if remaining <= 0:
                if total_load > 0:
                    return float('inf')
                pressure = 0.0
            else:
                pressure = total_load / remaining
            if pressure > max_pressure:
                max_pressure = pressure
        return max_pressure

    def try_two_opt(order, gpu_count, max_iter=500):
        """2-opt local search on order to reduce final max KVPR."""
        current_best = list(order)
        current_place = assign_models(current_best, gpu_count)
        current_score = compute_max_kvpr(current_place)
        if current_score == float('inf'):
            return current_best
        improved = True
        iterations = 0
        n = len(current_best)
        while improved and iterations < max_iter:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_order = current_best[:i] + current_best[i:j+1][::-1] + current_best[j+1:]
                    new_place = assign_models(new_order, gpu_count)
                    if new_place is None:
                        continue
                    new_score = compute_max_kvpr(new_place)
                    if new_score < current_score - 1e-12:
                        current_best = new_order
                        current_score = new_score
                        improved = True
            iterations += 1
        return current_best

    def try_swap_opt(order, gpu_count, max_iter=300):
        """Pair-swap local search on order to reduce final max KVPR."""
        current_best = list(order)
        current_score = compute_max_kvpr(assign_models(current_best, gpu_count))
        if current_score == float('inf'):
            return current_best
        improved = True
        iterations = 0
        n = len(current_best)
        while improved and iterations < max_iter:
            improved = False
            for i in range(n - 1):
                for j in range(i + 1, n):
                    new_order = list(current_best)
                    new_order[i], new_order[j] = new_order[j], new_order[i]
                    new_place = assign_models(new_order, gpu_count)
                    if new_place is None:
                        continue
                    new_score = compute_max_kvpr(new_place)
                    if new_score < current_score - 1e-12:
                        current_best = new_order
                        current_score = new_score
                        improved = True
            iterations += 1
        return current_best

    def try_placement_swap(placement, gpu_count, max_passes=80):
        """Placement-level local search: move/swaps models between GPUs to reduce max KVPR."""
        current = {g: list(ms) for g, ms in placement.items()}
        current_score = compute_max_kvpr(current)
        if current_score == float('inf'):
            return current

        gpu_ids = list(range(gpu_count))
        for _pass in range(max_passes):
            improved_any = False
            for g1 in range(gpu_count):
                for g2 in range(g1 + 1, gpu_count):
                    # Single move from g1 to g2
                    cur_g1 = list(current[g1])
                    cur_g2 = list(current[g2])
                    for idx1 in range(len(cur_g1)):
                        new_g1 = [m for i, m in enumerate(cur_g1) if i != idx1]
                        new_g2 = list(cur_g2) + [cur_g1[idx1]]
                        size1 = sum(m.model_size for m in new_g1)
                        size2 = sum(m.model_size for m in new_g2)
                        if size1 <= GPU_MEM_SIZE and size2 <= GPU_MEM_SIZE:
                            candidate = {g: list(current[g]) for g in gpu_ids}
                            candidate[g1] = new_g1
                            candidate[g2] = new_g2
                            new_score = compute_max_kvpr(candidate)
                            if new_score < current_score - 1e-12:
                                current = candidate
                                current_score = new_score
                                improved_any = True
                                cur_g1 = list(current[g1])
                                cur_g2 = list(current[g2])
                    # Single move from g2 to g1
                    cur_g1 = list(current[g1])
                    cur_g2 = list(current[g2])
                    for idx2 in range(len(cur_g2)):
                        new_g2 = [m for i, m in enumerate(cur_g2) if i != idx2]
                        new_g1 = list(cur_g1) + [cur_g2[idx2]]
                        size1 = sum(m.model_size for m in new_g1)
                        size2 = sum(m.model_size for m in new_g2)
                        if size1 <= GPU_MEM_SIZE and size2 <= GPU_MEM_SIZE:
                            candidate = {g: list(current[g]) for g in gpu_ids}
                            candidate[g1] = new_g1
                            candidate[g2] = new_g2
                            new_score = compute_max_kvpr(candidate)
                            if new_score < current_score - 1e-12:
                                current = candidate
                                current_score = new_score
                                improved_any = True
                                cur_g1 = list(current[g1])
                                cur_g2 = list(current[g2])
                    # Pair swap between g1 and g2
                    cur_g1 = list(current[g1])
                    cur_g2 = list(current[g2])
                    for idx1 in range(len(cur_g1)):
                        for idx2 in range(len(cur_g2)):
                            m1 = cur_g1[idx1]
                            m2 = cur_g2[idx2]
                            new_g1 = [m for i, m in enumerate(cur_g1) if i != idx1] + [m2]
                            new_g2 = [m for i, m in enumerate(cur_g2) if i != idx2] + [m1]
                            size1 = sum(m.model_size for m in new_g1)
                            size2 = sum(m.model_size for m in new_g2)
                            if size1 <= GPU_MEM_SIZE and size2 <= GPU_MEM_SIZE:
                                candidate = {g: list(current[g]) for g in gpu_ids}
                                candidate[g1] = new_g1
                                candidate[g2] = new_g2
                                new_score = compute_max_kvpr(candidate)
                                if new_score < current_score - 1e-12:
                                    current = candidate
                                    current_score = new_score
                                    improved_any = True
                                    cur_g1 = list(current[g1])
                                    cur_g2 = list(current[g2])
                    # Three-way swaps: move two from g1 to g2
                    cur_g1 = list(current[g1])
                    cur_g2 = list(current[g2])
                    for idx1a in range(len(cur_g1)):
                        for idx1b in range(idx1a + 1, len(cur_g1)):
                            new_g1 = [m for i, m in enumerate(cur_g1) if i != idx1a and i != idx1b]
                            new_g2 = list(cur_g2) + [cur_g1[idx1a], cur_g1[idx1b]]
                            size1 = sum(m.model_size for m in new_g1)
                            size2 = sum(m.model_size for m in new_g2)
                            if size1 <= GPU_MEM_SIZE and size2 <= GPU_MEM_SIZE:
                                candidate = {g: list(current[g]) for g in gpu_ids}
                                candidate[g1] = new_g1
                                candidate[g2] = new_g2
                                new_score = compute_max_kvpr(candidate)
                                if new_score < current_score - 1e-12:
                                    current = candidate
                                    current_score = new_score
                                    improved_any = True
                                    cur_g1 = list(current[g1])
                                    cur_g2 = list(current[g2])
            if not improved_any:
                break
        return current

    # Candidate orderings to try
    candidates = []

    candidates.append(list(models))

    sorted_load = sorted(models, key=lambda m: m.req_rate / m.slo, reverse=True)
    candidates.append(sorted_load)

    sorted_size = sorted(models, key=lambda m: m.model_size, reverse=True)
    candidates.append(sorted_size)

    sorted_size_asc = sorted(models, key=lambda m: m.model_size)
    candidates.append(sorted_size_asc)

    sorted_load_asc = sorted(models, key=lambda m: m.req_rate / m.slo)
    candidates.append(sorted_load_asc)

    sorted_load_then_size = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo, m.model_size),
        reverse=True,
    )
    candidates.append(sorted_load_then_size)

    sorted_product = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo) * m.model_size,
        reverse=True,
    )
    candidates.append(sorted_product)

    sorted_load_sq_product = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo) ** 2 * m.model_size,
        reverse=True,
    )
    candidates.append(sorted_load_sq_product)

    sorted_size_sq_product = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo) * (m.model_size ** 2),
        reverse=True,
    )
    candidates.append(sorted_size_sq_product)

    sorted_load_div_size = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo) / m.model_size,
        reverse=True,
    )
    candidates.append(sorted_load_div_size)

    sorted_size_then_load = sorted(
        models,
        key=lambda m: (m.model_size, m.req_rate / m.slo),
        reverse=True,
    )
    candidates.append(sorted_size_then_load)

    sorted_load_sqrt_size = sorted(
        models,
        key=lambda m: (m.req_rate / m.slo) / (m.model_size ** 0.5),
        reverse=True,
    )
    candidates.append(sorted_load_sqrt_size)

    sorted_sqrt_load_size = sorted(
        models,
        key=lambda m: ((m.req_rate / m.slo) ** 0.5) / m.model_size,
        reverse=True,
    )
    candidates.append(sorted_sqrt_load_size)

    # Add randomized orderings to diversify search
    rng = random.Random(42)
    shuffled = list(models)
    num_random = min(12, max(4, gpu_num))
    for _ in range(num_random):
        rng.shuffle(shuffled)
        candidates.append(list(shuffled))

    # Apply 2-opt and swap-opt improvements on strong heuristic orderings
    strong_bases = [
        sorted_load,
        sorted_product,
        sorted_load_sq_product,
        sorted_size,
        sorted_load_div_size,
        sorted_size_sq_product,
        sorted_load_sqrt_size,
        sorted_sqrt_load_size,
    ]
    for base in strong_bases:
        candidates.append(try_two_opt(base, gpu_num, max_iter=500))
        candidates.append(try_swap_opt(base, gpu_num, max_iter=300))

    best_placement = None
    best_kvpr = float('inf')

    for order in candidates:
        placement = assign_models(order, gpu_num)
        if placement is None:
            continue
        kvpr = compute_max_kvpr(placement)
        if kvpr < best_kvpr:
            best_kvpr = kvpr
            best_placement = placement

    # Apply placement-level swap refinement on best found so far
    if best_placement is not None and best_kvpr < float('inf'):
        swapped = try_placement_swap(best_placement, gpu_num, max_passes=80)
        swapped_kvpr = compute_max_kvpr(swapped)
        if swapped_kvpr < best_kvpr:
            best_kvpr = swapped_kvpr
            best_placement = swapped

    if best_placement is None:
        raise ValueError(
            f"Unable to place models across {gpu_num} GPUs. "
            f"Total model size: {sum(m.model_size for m in models)} GB, "
            f"Total GPU memory: {gpu_num * GPU_MEM_SIZE} GB"
        )

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