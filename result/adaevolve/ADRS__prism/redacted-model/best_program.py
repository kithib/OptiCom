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

    # Handle edge cases: no models or no GPUs
    if not models or gpu_num == 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    # Precompute model metrics: weight (r/s) and size, include original model
    model_metrics = [
        (m.req_rate / m.slo, m.model_size, m) for m in models
    ]
    # Sort by weight descending (place high-impact models first)
    model_metrics.sort(key=lambda x: x[0], reverse=True)

    # Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
    gpu_weight_sum = [0.0 for _ in range(gpu_num)]
    gpu_kvpr = [0.0 for _ in range(gpu_num)]
    gpu_list = list(range(gpu_num))

    # Initial greedy assignment: choose GPU that minimizes *resulting* KVPR (not current)
    for weight, size, model in model_metrics:
        best_idx = -1
        best_kvpr = float('inf')
        for gpu_id in range(gpu_num):
            if size <= remaining_mem[gpu_id]:
                new_weight = gpu_weight_sum[gpu_id] + weight
                new_mem = remaining_mem[gpu_id] - size
                # Skip GPUs that would have zero/negative remaining memory after placement
                if new_mem <= 0:
                    continue
                kvpr = new_weight / new_mem
                if kvpr < best_kvpr:
                    best_kvpr = kvpr
                    best_idx = gpu_id
        if best_idx == -1:
            raise ValueError(
                f"Unable to place model of size {size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining_mem}"
            )
        placement[best_idx].append(model)
        gpu_weight_sum[best_idx] += weight
        remaining_mem[best_idx] -= size
        gpu_kvpr[best_idx] = gpu_weight_sum[best_idx] / remaining_mem[best_idx]

    # Helper to safely compute GPU KVPR
    def get_gpu_kvpr(g):
        if remaining_mem[g] <= 0:
            return float('inf')
        return gpu_weight_sum[g] / remaining_mem[g]

    # Local refinement: iterate until no improvements or max iterations
    improved = True
    max_iterations = 3 * len(models)
    iteration = 0

    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Identify current max KVPR GPU and global max KVPR
        current_max_kvpr = -float('inf')
        current_max_gpu = -1
        second_max_kvpr = -float('inf')
        for g in gpu_list:
            kv = gpu_kvpr[g]
            if kv > current_max_kvpr:
                second_max_kvpr = current_max_kvpr
                current_max_kvpr = kv
                current_max_gpu = g
            elif kv > second_max_kvpr:
                second_max_kvpr = kv

        # Early exit if no valid max GPU found or max GPU is empty (no moves possible)
        if current_max_gpu == -1 or not placement[current_max_gpu]:
            break

        # Compute global non-modified max KVPR once for reuse
        def compute_new_max(kvpr_a, kvpr_b, gpu_a, gpu_b):
            new_max = max(kvpr_a, kvpr_b)
            for g in gpu_list:
                if g == gpu_a or g == gpu_b:
                    continue
                g_kv = gpu_kvpr[g]
                if g_kv > new_max:
                    new_max = g_kv
            return new_max

        # Phase 1: Try moving models from max KVPR GPU to others
        # Iterate over list copy to avoid mutation issues
        move_applied = False
        for model_a in list(placement[current_max_gpu]):
            ma_w = model_a.req_rate / model_a.slo
            ma_s = model_a.model_size
            for gpu_b in gpu_list:
                if gpu_b == current_max_gpu:
                    continue
                if ma_s > remaining_mem[gpu_b]:
                    continue
                # Simulate move from current_max_gpu to gpu_b
                new_weight_a = gpu_weight_sum[current_max_gpu] - ma_w
                new_mem_a = remaining_mem[current_max_gpu] + ma_s
                new_weight_b = gpu_weight_sum[gpu_b] + ma_w
                new_mem_b = remaining_mem[gpu_b] - ma_s
                # Skip invalid target state
                if new_mem_b <= 0:
                    continue
                new_kvpr_a = new_weight_a / new_mem_a if new_mem_a > 0 else float('inf')
                new_kvpr_b = new_weight_b / new_mem_b

                # Compute new system max KVPR
                new_max = compute_new_max(new_kvpr_a, new_kvpr_b, current_max_gpu, gpu_b)

                if new_max < current_max_kvpr - 1e-9:
                    # Apply move
                    placement[current_max_gpu].remove(model_a)
                    placement[gpu_b].append(model_a)
                    gpu_weight_sum[current_max_gpu] = new_weight_a
                    gpu_weight_sum[gpu_b] = new_weight_b
                    remaining_mem[current_max_gpu] = new_mem_a
                    remaining_mem[gpu_b] = new_mem_b
                    gpu_kvpr[current_max_gpu] = new_kvpr_a
                    gpu_kvpr[gpu_b] = new_kvpr_b
                    improved = True
                    move_applied = True
                    break
            if move_applied:
                break

        # Phase 2: Try swapping models if no move worked
        if not improved:
            swap_applied = False
            # Iterate high-impact models first in max KVPR GPU
            sorted_max_models = sorted(
                list(placement[current_max_gpu]),
                key=lambda m: m.req_rate / m.slo,
                reverse=True
            )
            for model_a in sorted_max_models:
                if swap_applied:
                    break
                ma_w = model_a.req_rate / model_a.slo
                ma_s = model_a.model_size
                for gpu_b in gpu_list:
                    if swap_applied:
                        break
                    if gpu_b == current_max_gpu:
                        continue
                    # Iterate low-impact models first in target GPU to increase swap benefit
                    sorted_b_models = sorted(
                        list(placement[gpu_b]),
                        key=lambda m: m.req_rate / m.slo
                    )
                    for model_b in sorted_b_models:
                        mb_w = model_b.req_rate / model_b.slo
                        mb_s = model_b.model_size
                        # Check memory feasibility after swap: net memory change per GPU is ma_s - mb_s
                        if (ma_s - mb_s) > remaining_mem[gpu_b]:
                            continue
                        if (mb_s - ma_s) > remaining_mem[current_max_gpu]:
                            continue
                        # Simulate swap
                        new_weight_a = gpu_weight_sum[current_max_gpu] - ma_w + mb_w
                        new_mem_a = remaining_mem[current_max_gpu] + ma_s - mb_s
                        new_weight_b = gpu_weight_sum[gpu_b] - mb_w + ma_w
                        new_mem_b = remaining_mem[gpu_b] + mb_s - ma_s
                        # Ensure non-zero positive remaining memory
                        if new_mem_a <= 0 or new_mem_b <= 0:
                            continue
                        new_kvpr_a = new_weight_a / new_mem_a
                        new_kvpr_b = new_weight_b / new_mem_b

                        # Compute new system max KVPR
                        new_max = compute_new_max(new_kvpr_a, new_kvpr_b, current_max_gpu, gpu_b)

                        if new_max < current_max_kvpr - 1e-9:
                            # Apply swap
                            placement[current_max_gpu].remove(model_a)
                            placement[gpu_b].remove(model_b)
                            placement[current_max_gpu].append(model_b)
                            placement[gpu_b].append(model_a)
                            gpu_weight_sum[current_max_gpu] = new_weight_a
                            gpu_weight_sum[gpu_b] = new_weight_b
                            remaining_mem[current_max_gpu] = new_mem_a
                            remaining_mem[gpu_b] = new_mem_b
                            gpu_kvpr[current_max_gpu] = new_kvpr_a
                            gpu_kvpr[gpu_b] = new_kvpr_b
                            improved = True
                            swap_applied = True
                            break

        # Phase 3: Optional cross-GPU move refinement if max is still high
        if not improved and gpu_num >= 3 and current_max_kvpr > second_max_kvpr * 1.1:
            # Try moving a small low-weight model from max GPU to any non-target with free space
            cross_applied = False
            for model_a in list(placement[current_max_gpu]):
                if cross_applied:
                    break
                ma_w = model_a.req_rate / model_a.slo
                ma_s = model_a.model_size
                # Prefer lightweight small models
                if ma_w > (gpu_weight_sum[current_max_gpu] * 0.25):
                    continue
                for gpu_b in gpu_list:
                    if gpu_b == current_max_gpu or cross_applied:
                        continue
                    if ma_s > remaining_mem[gpu_b]:
                        continue
                    new_weight_a = gpu_weight_sum[current_max_gpu] - ma_w
                    new_mem_a = remaining_mem[current_max_gpu] + ma_s
                    new_weight_b = gpu_weight_sum[gpu_b] + ma_w
                    new_mem_b = remaining_mem[gpu_b] - ma_s
                    if new_mem_b <= 0:
                        continue
                    new_kvpr_a = new_weight_a / new_mem_a if new_mem_a > 0 else float('inf')
                    new_kvpr_b = new_weight_b / new_mem_b
                    new_max = compute_new_max(new_kvpr_a, new_kvpr_b, current_max_gpu, gpu_b)
                    if new_max < current_max_kvpr - 1e-9:
                        placement[current_max_gpu].remove(model_a)
                        placement[gpu_b].append(model_a)
                        gpu_weight_sum[current_max_gpu] = new_weight_a
                        gpu_weight_sum[gpu_b] = new_weight_b
                        remaining_mem[current_max_gpu] = new_mem_a
                        remaining_mem[gpu_b] = new_mem_b
                        gpu_kvpr[current_max_gpu] = new_kvpr_a
                        gpu_kvpr[gpu_b] = new_kvpr_b
                        improved = True
                        cross_applied = True
                        break

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