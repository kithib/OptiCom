GPU_MEM_SIZE = 80 # GB

# EVOLVE-BLOCK-START

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses a two-phase approach: initial greedy placement then local search improvement
    to reduce the maximum KVPR further.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    if gpu_num == 0 or not models:
        return {i: [] for i in range(gpu_num)}

    def _gpu_metrics(assign_list):
        """Compute weighted_req_rate, used_mem, kvpr for each GPU given assignment list."""
        w = [0.0] * gpu_num
        used = [0] * gpu_num
        kvpr = [0.0] * gpu_num
        for mid, gid in enumerate(assign_list):
            m = models[mid]
            w[gid] += m.req_rate / m.slo
            used[gid] += m.model_size
        for g in range(gpu_num):
            remaining = GPU_MEM_SIZE - used[g]
            kvpr[g] = w[g] / remaining if remaining > 0 else float('inf')
        return w, used, kvpr

    def _max_kvpr(assign_list):
        _, _, kvpr = _gpu_metrics(assign_list)
        return max(kvpr)

    N = len(models)
    # Phase 1: Greedy by (req_rate/slo) descending, pick GPU minimizing resulting KVPR
    sorted_indices = sorted(range(N), key=lambda i: (models[i].req_rate / models[i].slo, models[i].model_size), reverse=True)

    assign = [-1] * N
    shared_kv = [GPU_MEM_SIZE] * gpu_num
    weighted_req_rate = [0.0] * gpu_num

    for idx in sorted_indices:
        m = models[idx]
        m_weight = m.req_rate / m.slo
        m_size = m.model_size
        best_idx = None
        best_post_kvpr = float('inf')
        for gpu_id in range(gpu_num):
            if m_size <= shared_kv[gpu_id]:
                new_weight = weighted_req_rate[gpu_id] + m_weight
                new_remaining = shared_kv[gpu_id] - m_size
                post_kvpr = new_weight / new_remaining if new_remaining > 0 else float('inf')
                if post_kvpr < best_post_kvpr:
                    best_post_kvpr = post_kvpr
                    best_idx = gpu_id
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {m.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )
        assign[idx] = best_idx
        weighted_req_rate[best_idx] += m_weight
        shared_kv[best_idx] -= m_size

    # Phase 2: Local search via moves and swaps to reduce maximum KVPR
    cur_max = _max_kvpr(assign)

    improved = True
    max_iterations = 50
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Identify highest KVPR GPU
        w, used, kvpr = _gpu_metrics(assign)
        highest_gpu = 0
        for g in range(1, gpu_num):
            if kvpr[g] > kvpr[highest_gpu]:
                highest_gpu = g
        # Identify second-highest (targets for cross-improvement)
        second_gpu = -1
        second_k = -1.0
        for g in range(gpu_num):
            if g == highest_gpu:
                continue
            if kvpr[g] > second_k:
                second_k = kvpr[g]
                second_gpu = g

        # Gather models on highest_gpu, sorted by impact descending (higher weight / size first)
        models_on_highest = [mid for mid in range(N) if assign[mid] == highest_gpu]
        models_on_highest.sort(
            key=lambda mid: -(models[mid].req_rate / models[mid].slo) * models[mid].model_size
        )

        # Try moves: pick move with largest reduction in max KVPR
        best_move = None
        best_move_max = cur_max
        for mid in models_on_highest:
            m = models[mid]
            for target_gpu in range(gpu_num):
                if target_gpu == highest_gpu:
                    continue
                if used[target_gpu] + m.model_size > GPU_MEM_SIZE:
                    continue
                # Only compute new max (lightweight)
                new_h_w = w[highest_gpu] - (m.req_rate / m.slo)
                new_h_used = used[highest_gpu] - m.model_size
                new_h_remaining = GPU_MEM_SIZE - new_h_used
                new_h_kvpr = new_h_w / new_h_remaining if new_h_remaining > 0 else float('inf')
                new_t_w = w[target_gpu] + (m.req_rate / m.slo)
                new_t_used = used[target_gpu] + m.model_size
                new_t_remaining = GPU_MEM_SIZE - new_t_used
                new_t_kvpr = new_t_w / new_t_remaining if new_t_remaining > 0 else float('inf')
                # New maximum is max of the new highest, new target, and all others
                new_max_candidate = new_h_kvpr
                if new_t_kvpr > new_max_candidate:
                    new_max_candidate = new_t_kvpr
                for g in range(gpu_num):
                    if g == highest_gpu or g == target_gpu:
                        continue
                    if kvpr[g] > new_max_candidate:
                        new_max_candidate = kvpr[g]
                if new_max_candidate < best_move_max:
                    best_move_max = new_max_candidate
                    best_move = (mid, target_gpu)
        if best_move is not None:
            assign[best_move[0]] = best_move[1]
            cur_max = best_move_max
            improved = True
            continue

        # Try swaps between highest GPU and other GPUs
        best_swap = None
        best_swap_max = cur_max
        # Sort other GPU models by impact too
        for mid1 in models_on_highest:
            m1 = models[mid1]
            for mid2 in range(N):
                if assign[mid2] == highest_gpu:
                    continue
                m2 = models[mid2]
                g2 = assign[mid2]
                if (used[highest_gpu] - m1.model_size + m2.model_size) > GPU_MEM_SIZE:
                    continue
                if (used[g2] - m2.model_size + m1.model_size) > GPU_MEM_SIZE:
                    continue
                # Compute new KVPR for highest_gpu and g2 only
                new_h_w = w[highest_gpu] - (m1.req_rate / m1.slo) + (m2.req_rate / m2.slo)
                new_h_used = used[highest_gpu] - m1.model_size + m2.model_size
                new_h_remaining = GPU_MEM_SIZE - new_h_used
                new_h_kvpr = new_h_w / new_h_remaining if new_h_remaining > 0 else float('inf')
                new_g2_w = w[g2] - (m2.req_rate / m2.slo) + (m1.req_rate / m1.slo)
                new_g2_used = used[g2] - m2.model_size + m1.model_size
                new_g2_remaining = GPU_MEM_SIZE - new_g2_used
                new_g2_kvpr = new_g2_w / new_g2_remaining if new_g2_remaining > 0 else float('inf')
                new_max_candidate = new_h_kvpr
                if new_g2_kvpr > new_max_candidate:
                    new_max_candidate = new_g2_kvpr
                for g in range(gpu_num):
                    if g == highest_gpu or g == g2:
                        continue
                    if kvpr[g] > new_max_candidate:
                        new_max_candidate = kvpr[g]
                if new_max_candidate < best_swap_max:
                    best_swap_max = new_max_candidate
                    best_swap = (mid1, mid2, g2)
        if best_swap is not None:
            mid1, mid2, g2 = best_swap
            assign[mid1] = g2
            assign[mid2] = highest_gpu
            cur_max = best_swap_max
            improved = True

        # Try cross-GPU swaps between two non-highest GPUs (sometimes reduces global max)
        if not improved and second_gpu != -1:
            best_cross = None
            best_cross_max = cur_max
            models_on_second = [mid for mid in range(N) if assign[mid] == second_gpu]
            for mid1 in models_on_second:
                m1 = models[mid1]
                for mid2 in range(N):
                    if assign[mid2] == second_gpu or assign[mid2] == highest_gpu:
                        continue
                    m2 = models[mid2]
                    g2 = assign[mid2]
                    if (used[second_gpu] - m1.model_size + m2.model_size) > GPU_MEM_SIZE:
                        continue
                    if (used[g2] - m2.model_size + m1.model_size) > GPU_MEM_SIZE:
                        continue
                    new_s_w = w[second_gpu] - (m1.req_rate / m1.slo) + (m2.req_rate / m2.slo)
                    new_s_used = used[second_gpu] - m1.model_size + m2.model_size
                    new_s_remaining = GPU_MEM_SIZE - new_s_used
                    new_s_kvpr = new_s_w / new_s_remaining if new_s_remaining > 0 else float('inf')
                    new_g2_w = w[g2] - (m2.req_rate / m2.slo) + (m1.req_rate / m1.slo)
                    new_g2_used = used[g2] - m2.model_size + m1.model_size
                    new_g2_remaining = GPU_MEM_SIZE - new_g2_used
                    new_g2_kvpr = new_g2_w / new_g2_remaining if new_g2_remaining > 0 else float('inf')
                    new_max_candidate = kvpr[highest_gpu]
                    if new_s_kvpr > new_max_candidate:
                        new_max_candidate = new_s_kvpr
                    if new_g2_kvpr > new_max_candidate:
                        new_max_candidate = new_g2_kvpr
                    for g in range(gpu_num):
                        if g == highest_gpu or g == second_gpu or g == g2:
                            continue
                        if kvpr[g] > new_max_candidate:
                            new_max_candidate = kvpr[g]
                    if new_max_candidate < best_cross_max:
                        best_cross_max = new_max_candidate
                        best_cross = (mid1, mid2, g2, second_gpu)
            if best_cross is not None:
                mid1, mid2, g2, _s = best_cross
                assign[mid1] = g2
                assign[mid2] = second_gpu
                cur_max = best_cross_max
                improved = True

    # Build placement dict
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    for mid in range(N):
        placement[assign[mid]].append(models[mid])
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