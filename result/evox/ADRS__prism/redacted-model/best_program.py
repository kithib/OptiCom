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

    # Early feasibility check to avoid unnecessary computation for impossible cases
    total_model_size = sum(model.model_size for model in models)
    total_gpu_mem = gpu_num * GPU_MEM_SIZE
    if total_model_size > total_gpu_mem:
        raise ValueError(
            f"Total model size {total_model_size} GB exceeds total GPU memory {total_gpu_mem} GB"
        )

    # Precompute model metrics to avoid redundant calculations during placement
    # Priority combines demand rate and size to place highest-impact models first
    precomputed = []
    for model in models:
        rate_ratio = model.req_rate / model.slo
        size = model.model_size
        priority = rate_ratio * size
        precomputed.append((priority, rate_ratio, size, model))
    
    # Sort by priority in descending order
    precomputed.sort(key=lambda x: x[0], reverse=True)

    # Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]    # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]     # sum of r_j / s_j per GPU
    current_max_ratio = 0.0                               # Track current global max KVPR

    # Assign each model to the GPU that minimizes the post-assignment system-wide maximum KVPR
    for priority, m_rate_ratio, m_size, model in precomputed:
        best_idx = None
        best_max_ratio = float('inf')
        best_projected = float('inf')
        best_memory = -1

        for gpu_id in range(gpu_num):
            if m_size > shared_kv[gpu_id]:
                continue
            new_weighted = weighted_req_rate[gpu_id] + m_rate_ratio
            new_memory = shared_kv[gpu_id] - m_size
            if new_memory > 0:
                projected_ratio = new_weighted / new_memory
            else:
                projected_ratio = float('inf') if new_weighted > 0 else 0.0
            
            # The new system-wide maximum KVPR if we place this model here
            new_max_ratio = projected_ratio if projected_ratio > current_max_ratio else current_max_ratio

            # Primary objective: minimize the new system-wide max KVPR
            if new_max_ratio < best_max_ratio:
                best_max_ratio = new_max_ratio
                best_projected = projected_ratio
                best_memory = new_memory
                best_idx = gpu_id
            elif new_max_ratio == best_max_ratio:
                # Tiebreaker 1: prefer the GPU with lower projected KVPR
                if projected_ratio < best_projected:
                    best_projected = projected_ratio
                    best_memory = new_memory
                    best_idx = gpu_id
                elif projected_ratio == best_projected:
                    # Tiebreaker 2: prefer the GPU with more remaining memory
                    if new_memory > best_memory:
                        best_memory = new_memory
                        best_idx = gpu_id
            
            # Early exit: if we found a placement that keeps max_ratio <= current global max, stop searching
            if best_max_ratio <= current_max_ratio:
                break

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += m_rate_ratio
        shared_kv[best_idx] -= m_size

        # Update current global max KVPR
        new_memory_gpu = shared_kv[best_idx]
        new_weighted_gpu = weighted_req_rate[best_idx]
        if new_memory_gpu > 0:
            new_ratio = new_weighted_gpu / new_memory_gpu
        else:
            new_ratio = float('inf') if new_weighted_gpu > 0 else 0.0
        if new_ratio > current_max_ratio:
            current_max_ratio = new_ratio

    # First local search phase: move models from highest-KVPR GPU to lower ones to reduce max
    improved = True
    while improved:
        improved = False

        # Calculate current KVPR for each GPU
        gpu_kvpr = []
        for gpu_id in range(gpu_num):
            if shared_kv[gpu_id] > 0:
                kvpr = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
            else:
                kvpr = float('inf') if weighted_req_rate[gpu_id] > 0 else 0.0
            gpu_kvpr.append(kvpr)

        max_gpu = 0
        max_kvpr = gpu_kvpr[0]
        for gpu_id in range(1, gpu_num):
            if gpu_kvpr[gpu_id] > max_kvpr:
                max_kvpr = gpu_kvpr[gpu_id]
                max_gpu = gpu_id

        # If no GPU has positive KVPR, we are done
        if max_kvpr <= 0 or max_kvpr == float('inf'):
            break

        # Try moving each model from the max GPU to any other GPU that reduces system max
        best_new_max = max_kvpr
        best_move = None

        for model_idx in range(len(placement[max_gpu])):
            model = placement[max_gpu][model_idx]
            m_rate = model.req_rate / model.slo
            m_size = model.model_size

            # Project max GPU's KVPR after removing this model
            new_rate_max = weighted_req_rate[max_gpu] - m_rate
            new_mem_max = shared_kv[max_gpu] + m_size
            if new_mem_max > 0:
                new_kvpr_max = new_rate_max / new_mem_max
            else:
                new_kvpr_max = float('inf') if new_rate_max > 0 else 0.0

            for target_gpu in range(gpu_num):
                if target_gpu == max_gpu:
                    continue
                if m_size > shared_kv[target_gpu]:
                    continue

                # Project target GPU's KVPR after adding this model
                new_rate_target = weighted_req_rate[target_gpu] + m_rate
                new_mem_target = shared_kv[target_gpu] - m_size
                if new_mem_target > 0:
                    new_kvpr_target = new_rate_target / new_mem_target
                else:
                    new_kvpr_target = float('inf') if new_rate_target > 0 else 0.0

                # Compute new system-wide max KVPR
                new_max = new_kvpr_max if new_kvpr_max > new_kvpr_target else new_kvpr_target
                for other_gpu in range(gpu_num):
                    if other_gpu == max_gpu or other_gpu == target_gpu:
                        continue
                    if gpu_kvpr[other_gpu] > new_max:
                        new_max = gpu_kvpr[other_gpu]

                if new_max < best_new_max:
                    best_new_max = new_max
                    best_move = (model_idx, target_gpu)

        if best_move is not None:
            model_idx, target_gpu = best_move
            model = placement[max_gpu].pop(model_idx)
            placement[target_gpu].append(model)
            m_rate = model.req_rate / model.slo
            m_size = model.model_size
            weighted_req_rate[max_gpu] -= m_rate
            shared_kv[max_gpu] += m_size
            weighted_req_rate[target_gpu] += m_rate
            shared_kv[target_gpu] -= m_size
            improved = True

    # Second local search phase: try pair-wise swaps between GPUs to further reduce max KVPR
    improved = True
    max_iterations = 10  # Limit iterations to avoid excessive runtime
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Recalculate current KVPR for each GPU
        gpu_kvpr = []
        for gpu_id in range(gpu_num):
            if shared_kv[gpu_id] > 0:
                kvpr = weighted_req_rate[gpu_id] / shared_kv[gpu_id]
            else:
                kvpr = float('inf') if weighted_req_rate[gpu_id] > 0 else 0.0
            gpu_kvpr.append(kvpr)

        current_global_max = max(gpu_kvpr) if gpu_num > 0 else 0.0
        if current_global_max <= 0 or current_global_max == float('inf'):
            break

        best_global_new_max = current_global_max
        best_global_swap = None

        # Iterate over all pairs of GPUs to find beneficial swaps
        for gpu_a in range(gpu_num):
            for gpu_b in range(gpu_a + 1, gpu_num):
                # Skip pairs of GPUs that are both below the current max for efficiency
                if gpu_kvpr[gpu_a] < current_global_max and gpu_kvpr[gpu_b] < current_global_max:
                    continue
                # Iterate over all pairs of models between gpu_a and gpu_b
                for model_a_idx in range(len(placement[gpu_a])):
                    model_a = placement[gpu_a][model_a_idx]
                    rate_a = model_a.req_rate / model.slo
                    size_a = model_a.model_size

                    for model_b_idx in range(len(placement[gpu_b])):
                        model_b = placement[gpu_b][model_b_idx]
                        rate_b = model_b.req_rate / model_b.slo
                        size_b = model_b.model_size

                        # Check if swapping is feasible for both GPUs
                        # For GPU a: remove model_a, add model_b: memory change = size_a - size_b
                        new_mem_a = shared_kv[gpu_a] + size_a - size_b
                        if new_mem_a < 0:
                            continue
                        # For GPU b: remove model_b, add model_a: memory change = size_b - size_a
                        new_mem_b = shared_kv[gpu_b] + size_b - size_a
                        if new_mem_b < 0:
                            continue

                        # Calculate projected KVPR for GPU a after swap
                        new_rate_a = weighted_req_rate[gpu_a] - rate_a + rate_b
                        if new_mem_a > 0:
                            new_kvpr_a = new_rate_a / new_mem_a
                        else:
                            new_kvpr_a = float('inf') if new_rate_a > 0 else 0.0

                        # Calculate projected KVPR for GPU b after swap
                        new_rate_b = weighted_req_rate[gpu_b] - rate_b + rate_a
                        if new_mem_b > 0:
                            new_kvpr_b = new_rate_b / new_mem_b
                        else:
                            new_kvpr_b = float('inf') if new_rate_b > 0 else 0.0

                        # Compute new system-wide max KVPR
                        new_max = new_kvpr_a if new_kvpr_a > new_kvpr_b else new_kvpr_b
                        for other_gpu in range(gpu_num):
                            if other_gpu == gpu_a or other_gpu == gpu_b:
                                continue
                            if gpu_kvpr[other_gpu] > new_max:
                                new_max = gpu_kvpr[other_gpu]

                        if new_max < best_global_new_max:
                            best_global_new_max = new_max
                            best_global_swap = (gpu_a, model_a_idx, gpu_b, model_b_idx)

        if best_global_swap is not None:
            gpu_a, model_a_idx, gpu_b, model_b_idx = best_global_swap
            model_a = placement[gpu_a][model_a_idx]
            model_b = placement[gpu_b][model_b_idx]

            # Update placements
            placement[gpu_a][model_a_idx] = model_b
            placement[gpu_b][model_b_idx] = model_a

            # Update GPU a metrics
            rate_a = model_a.req_rate / model.a.slo
            size_a = model_a.model_size
            rate_b = model_b.req_rate / model.b.slo
            size_b = model_b.model_size

            weighted_req_rate[gpu_a] = weighted_req_rate[gpu_a] - rate_a + rate_b
            shared_kv[gpu_a] = shared_kv[gpu_a] + size_a - size_b

            # Update GPU b metrics
            weighted_req_rate[gpu_b] = weighted_req_rate[gpu_b] - rate_b + rate_a
            shared_kv[gpu_b] = shared_kv[gpu_b] + size_b - size_a

            improved = True

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