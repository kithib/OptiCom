GPU_MEM_SIZE = 80  # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    """

    # Pre-check for trivially impossible placements (total model size > total GPU memory)
    total_model_mem = sum(m.model_size for m in models)
    if total_model_mem > gpu_num * GPU_MEM_SIZE:
        raise ValueError(
            f"Total model size ({total_model_mem} GB) exceeds total available GPU memory "
            f"({gpu_num * GPU_MEM_SIZE} GB). Placement impossible."
        )

    # Edge case: single GPU short-circuit
    if gpu_num == 1:
        if total_model_mem > GPU_MEM_SIZE:
            raise ValueError(
                f"Total model size ({total_model_mem} GB) exceeds single-GPU memory "
                f"({GPU_MEM_SIZE} GB). Placement impossible."
            )
        return {0: list(models)}

    # Precompute per-model pressure and size once to eliminate repeated attribute lookups
    precomputed = []
    for m in models:
        pressure = m.req_rate / m.slo
        size = m.model_size
        precomputed.append((pressure, size, m))

    # Diversified strategies: balanced ordering mix for broader search
    strategy_keys = [
        lambda p: (p[1] * p[0], p[1]),  # composite: size*pressure desc, size desc tiebreak
        lambda p: p[0],                  # raw pressure desc (baseline greedy)
        lambda p: p[1],                  # largest-first (memory heavy)
        lambda p: (p[0] / p[1] if p[1] > 0 else 0.0, p[1]),  # pressure per size desc
        lambda p: (-(p[1]), p[0]),       # largest-first, then highest-pressure tiebreak (balanced)
    ]

    best_global_max = float('inf')
    best_placement = None

    def _compute_kvprs(weighted, rem_mem):
        return [w / r if r > 0 else 0.0 for w, r in zip(weighted, rem_mem)]

    def _try_strategy(strategy_key):
        nonlocal best_global_max, best_placement

        sorted_data = sorted(precomputed, key=strategy_key, reverse=True)

        # Initialize per-GPU state
        placement = {g: [] for g in range(gpu_num)}
        rem_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted = [0.0 for _ in range(gpu_num)]

        # Sort remaining-GPU slots by memory descending for faster pruning of infeasible fits
        gpu_by_mem = sorted(range(gpu_num), key=lambda g: -rem_mem[g])

        for pressure, size, m in sorted_data:
            best_idx = -1
            best_post_max = float('inf')
            best_post_rem = 0

            # Precompute current global max KVPR across all GPUs for fast path
            cur_kvpr = _compute_kvprs(weighted, rem_mem)
            cur_max = 0.0
            cur_max_count = 0
            for k in cur_kvpr:
                if k > cur_max:
                    cur_max = k
                    cur_max_count = 1
                elif k == cur_max:
                    cur_max_count += 1

            # Iterate GPUs ordered by remaining memory; prune when no fit possible
            gpu_iter = sorted(range(gpu_num), key=lambda g: -rem_mem[g])
            for gpu_id in gpu_iter:
                if size > rem_mem[gpu_id]:
                    break  # further GPUs have <= remaining memory; no need to check
                new_weighted = weighted[gpu_id] + pressure
                new_rem = rem_mem[gpu_id] - size
                if new_rem <= 0:
                    continue
                post_ratio = new_weighted / new_rem
                # Fast path: compute resulting global max without full recounts
                if post_ratio >= cur_max:
                    global_max = post_ratio
                else:
                    if cur_kvpr[gpu_id] != cur_max or cur_max_count > 1:
                        global_max = cur_max
                    else:
                        max_others = 0.0
                        for g in range(gpu_num):
                            if g == gpu_id:
                                continue
                            if cur_kvpr[g] > max_others:
                                max_others = cur_kvpr[g]
                        global_max = max(max_others, post_ratio)

                # Minimize global max; tiebreak by larger remaining memory (less fragmentation)
                if global_max < best_post_max or (global_max == best_post_max and new_rem > best_post_rem):
                    best_post_max = global_max
                    best_post_rem = new_rem
                    best_idx = gpu_id

            if best_idx == -1:
                return None  # Infeasible with this strategy; abort early

            placement[best_idx].append(m)
            weighted[best_idx] += pressure
            rem_mem[best_idx] -= size

        # Local search: iterative pairwise swaps plus move operations to reduce global max KVPR
        MAX_PASSES = 16
        gpu_count = gpu_num

        for _pass in range(MAX_PASSES):
            improved = False
            kvpr = _compute_kvprs(weighted, rem_mem)
            current_global_max = max(kvpr) if kvpr else 0.0

            # Prioritize operations involving highest-KVPR GPU first
            max_gpu = kvpr.index(current_global_max) if kvpr else -1
            orderings = []
            if max_gpu >= 0:
                orderings.append([max_gpu] + [g for g in range(gpu_count) if g != max_gpu])
            orderings.append(list(range(gpu_count)))

            # Try SWAPS first (more likely to balance pressure)
            for gpu_order in orderings:
                if improved:
                    break
                for gpu_a in gpu_order:
                    if not placement[gpu_a]:
                        continue
                    for i in range(len(placement[gpu_a])):
                        m_a = placement[gpu_a][i]
                        pa = m_a.req_rate / m_a.slo
                        sa = m_a.model_size
                        for gpu_b in range(gpu_count):
                            if gpu_b == gpu_a or not placement[gpu_b]:
                                continue
                            for j in range(len(placement[gpu_b])):
                                m_b = placement[gpu_b][j]
                                pb = m_b.req_rate / m_b.slo
                                sb = m_b.model_size
                                # Memory feasibility check for swap
                                new_ra = rem_mem[gpu_a] + sa - sb
                                new_rb = rem_mem[gpu_b] + sb - sa
                                if new_ra <= 0 or new_rb <= 0:
                                    continue
                                # New weighted pressures and KVPRs
                                new_wa = weighted[gpu_a] - pa + pb
                                new_wb = weighted[gpu_b] - pb + pa
                                new_ka = new_wa / new_ra
                                new_kb = new_wb / new_rb
                                # Compute new global max
                                new_max = new_ka if new_ka > new_kb else new_kb
                                for g in range(gpu_count):
                                    if g == gpu_a or g == gpu_b:
                                        continue
                                    if kvpr[g] > new_max:
                                        new_max = kvpr[g]
                                if new_max < current_global_max - 1e-12:
                                    # Apply swap in-place
                                    placement[gpu_a][i], placement[gpu_b][j] = placement[gpu_b][j], placement[gpu_a][i]
                                    weighted[gpu_a] = new_wa
                                    weighted[gpu_b] = new_wb
                                    rem_mem[gpu_a] = new_ra
                                    rem_mem[gpu_b] = new_rb
                                    improved = True
                                    current_global_max = new_max
                                    break
                            if improved:
                                break
                        if improved:
                            break
                    if improved:
                        break
                if improved:
                    break

            # If no swap improvements, try MOVING a single model from max-KVPR GPU to another
            if not improved and max_gpu >= 0:
                for i in range(len(placement[max_gpu])):
                    if improved:
                        break
                    m_a = placement[max_gpu][i]
                    pa = m_a.req_rate / m_a.slo
                    sa = m_a.model_size
                    for gpu_b in range(gpu_count):
                        if gpu_b == max_gpu:
                            continue
                        if sa > rem_mem[gpu_b]:
                            continue
                        new_ra = rem_mem[max_gpu] + sa
                        new_rb = rem_mem[gpu_b] - sa
                        if new_ra <= 0 or new_rb <= 0:
                            continue
                        new_wa = weighted[max_gpu] - pa
                        new_wb = weighted[gpu_b] + pa
                        new_ka = new_wa / new_ra if new_ra > 0 else 0.0
                        new_kb = new_wb / new_rb if new_rb > 0 else 0.0
                        new_max = new_ka if new_ka > new_kb else new_kb
                        for g in range(gpu_count):
                            if g == max_gpu or g == gpu_b:
                                continue
                            if kvpr[g] > new_max:
                                new_max = kvpr[g]
                        if new_max < current_global_max - 1e-12:
                            # Apply move
                            placement[max_gpu].pop(i)
                            placement[gpu_b].append(m_a)
                            weighted[max_gpu] = new_wa
                            weighted[gpu_b] = new_wb
                            rem_mem[max_gpu] = new_ra
                            rem_mem[gpu_b] = new_rb
                            improved = True
                            current_global_max = new_max
                            break

            if not improved:
                break

        # Commit best result
        final_max = 0.0
        for g in range(gpu_num):
            if rem_mem[g] > 0:
                r = weighted[g] / rem_mem[g]
                if r > final_max:
                    final_max = r
        if final_max < best_global_max:
            best_global_max = final_max
            best_placement = placement

    # Run all strategies; track best
    for key in strategy_keys:
        _try_strategy(key)

    if best_placement is not None:
        return best_placement

    # Fallback: robust composite-key greedy (guaranteed to work if pre-check passed)
    sorted_data = sorted(precomputed, key=strategy_keys[0], reverse=True)
    placement = {g: [] for g in range(gpu_num)}
    rem_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted = [0.0 for _ in range(gpu_num)]

    for pressure, size, m in sorted_data:
        best_idx = None
        best_ratio = float('inf')
        for gpu_id in range(gpu_num):
            if size <= rem_mem[gpu_id] and rem_mem[gpu_id] > 0:
                r = weighted[gpu_id] / rem_mem[gpu_id]
                if r < best_ratio:
                    best_ratio = r
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {size} GB on any GPU. "
                f"Remaining per-GPU memory: {rem_mem}"
            )
        placement[best_idx].append(m)
        weighted[best_idx] += pressure
        rem_mem[best_idx] -= size

    return placement

# EVOLVE-BLOCK-END


if __name__ == "__main__":
    # Test the algorithm
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