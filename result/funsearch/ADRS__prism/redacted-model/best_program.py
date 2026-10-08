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

    # 1) Sort models by KVPR density ((req_rate/slo)/model_size) in descending order
    #    Prioritize placing "KVPR-dense" models first to optimize early allocation
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) / m.model_size, reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # Remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # Sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes *projected* KVPR after placement
    #    (Forward-looking decision making to avoid suboptimal load balancing)
    for model in sorted_models:
        best_idx = None
        best_projected_ratio = float('inf')
        model_wr = model.req_rate / model.slo
        model_size = model.model_size

        for gpu_id in range(gpu_num):
            if model_size <= shared_kv[gpu_id] and shared_kv[gpu_id] > 0:
                projected_wr = weighted_req_rate[gpu_id] + model_wr
                projected_remaining = shared_kv[gpu_id] - model_size
                if projected_remaining > 0:
                    projected_ratio = projected_wr / projected_remaining
                    if projected_ratio < best_projected_ratio:
                        best_projected_ratio = projected_ratio
                        best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model_wr
        shared_kv[best_idx] -= model_size

    # 4) Post-optimization Phase 1: Iterative local search to reduce maximum KVPR
    #    - Focus on moving models from high-KVPR GPUs to low-KVPR GPUs
    #    - Terminates when no improvement is found or iteration limit reached
    improved = True
    max_iterations = 10 * gpu_num
    iteration = 0
    while improved and iteration < max_iterations:
        improved = False
        iteration += 1

        # Compute current KVPR for all GPUs
        current_kvpr = []
        for gpu_id in range(gpu_num):
            if shared_kv[gpu_id] > 0:
                current_kvpr.append(weighted_req_rate[gpu_id] / shared_kv[gpu_id])
            else:
                current_kvpr.append(float('inf') if weighted_req_rate[gpu_id] > 0 else 0.0)

        max_kvpr = max(current_kvpr) if current_kvpr else 0.0
        min_kvpr = min(current_kvpr) if current_kvpr else 0.0

        if max_kvpr == min_kvpr or max_kvpr == 0:
            break

        high_gpu = current_kvpr.index(max_kvpr)
        low_gpu = current_kvpr.index(min_kvpr)

        # Find best model to move from high_gpu to low_gpu
        best_model_to_move = None
        best_new_max = float('inf')

        for model in placement[high_gpu]:
            if model.model_size <= shared_kv[low_gpu]:
                new_high_demand = weighted_req_rate[high_gpu] - (model.req_rate / model.slo)
                new_high_mem = shared_kv[high_gpu] + model.model_size
                new_low_demand = weighted_req_rate[low_gpu] + (model.req_rate / model.slo)
                new_low_mem = shared_kv[low_gpu] - model.model_size

                if new_high_mem > 0 and new_low_mem > 0:
                    new_high_kvpr = new_high_demand / new_high_mem
                    new_low_kvpr = new_low_demand / new_low_mem
                    other_max = max([current_kvpr[i] for i in range(gpu_num) if i not in (high_gpu, low_gpu)], default=0.0)
                    new_max_kvpr = max(new_high_kvpr, new_low_kvpr, other_max)

                    if new_max_kvpr < best_new_max:
                        best_new_max = new_max_kvpr
                        best_model_to_move = model

        # Execute move if it reduces overall max KVPR
        if best_model_to_move is not None and best_new_max < max_kvpr - 1e-9:
            model = best_model_to_move
            placement[high_gpu].remove(model)
            placement[low_gpu].append(model)

            weighted_req_rate[high_gpu] -= model.req_rate / model.slo
            shared_kv[high_gpu] += model.model_size
            weighted_req_rate[low_gpu] += model.req_rate / model.slo
            shared_kv[low_gpu] -= model.model_size

            improved = True

    # 5) Post-optimization Phase 2: Pairwise GPU model swaps
    #    - Corrects suboptimal placements where movement alone isn't sufficient
    #    - Runs for a fixed small number of iterations to keep complexity low
    for _ in range(2):
        swapped = False
        for g1 in range(gpu_num):
            for g2 in range(g1 + 1, gpu_num):
                # Compute current KVPR for both GPUs
                kvpr1 = weighted_req_rate[g1] / shared_kv[g1] if shared_kv[g1] > 0 else float('inf')
                kvpr2 = weighted_req_rate[g2] / shared_kv[g2] if shared_kv[g2] > 0 else float('inf')
                current_max = max(kvpr1, kvpr2)

                # Try swapping pairs of models between g1 and g2
                for i, m1 in enumerate(placement[g1]):
                    for j, m2 in enumerate(placement[g2]):
                        # Memory feasibility check
                        mem_after_g1 = shared_kv[g1] + m1.model_size - m2.model_size
                        mem_after_g2 = shared_kv[g2] + m2.model_size - m1.model_size
                        if mem_after_g1 < 0 or mem_after_g2 < 0:
                            continue

                        # Compute new KVPRs after swap
                        new_w1 = weighted_req_rate[g1] - (m1.req_rate / m1.slo) + (m2.req_rate / m2.slo)
                        new_w2 = weighted_req_rate[g2] - (m2.req_rate / m2.slo) + (m1.req_rate / m1.slo)
                        new_kvpr1 = new_w1 / mem_after_g1
                        new_kvpr2 = new_w2 / mem_after_g2
                        new_max = max(new_kvpr1, new_kvpr2)

                        if new_max < current_max - 1e-9:
                            # Execute swap
                            placement[g1][i], placement[g2][j] = m2, m1
                            weighted_req_rate[g1] = new_w1
                            weighted_req_rate[g2] = new_w2
                            shared_kv[g1] = mem_after_g1
                            shared_kv[g2] = mem_after_g2
                            swapped = True
                            break
                    if swapped:
                        break
        if not swapped:
            break

    # 6) Post-optimization Phase 3: General pairwise swaps to reduce overall maximum KVPR
    #    - Targets the highest KVPR GPU and tries swaps with all other GPUs
    #    - Further refines placement for minimal maximum KVPR
    improved = True
    while improved:
        improved = False
        kvpr_per_gpu = []
        for g in range(gpu_num):
            if shared_kv[g] > 0:
                kvpr_per_gpu.append(weighted_req_rate[g] / shared_kv[g])
            else:
                kvpr_per_gpu.append(float('inf') if weighted_req_rate[g] > 0 else 0.0)
        current_max_kvpr = max(kvpr_per_gpu) if kvpr_per_gpu else 0.0
        max_gpu = kvpr_per_gpu.index(current_max_kvpr)

        for m1 in list(placement[max_gpu]):
            for other_gpu in range(gpu_num):
                if other_gpu == max_gpu:
                    continue
                for m2 in list(placement[other_gpu]):
                    # Check memory feasibility of swap
                    new_remaining_max = shared_kv[max_gpu] + m1.model_size - m2.model_size
                    new_remaining_other = shared_kv[other_gpu] + m2.model_size - m1.model_size
                    if new_remaining_max < 0 or new_remaining_other < 0:
                        continue

                    # Check new remaining memory is positive (to avoid division by zero later)
                    if new_remaining_max <= 0 or new_remaining_other <= 0:
                        continue

                    # Compute new weighted req rates
                    new_wr_max = weighted_req_rate[max_gpu] - (m1.req_rate / m1.slo) + (m2.req_rate / m2.slo)
                    new_wr_other = weighted_req_rate[other_gpu] - (m2.req_rate / m2.slo) + (m1.req_rate / m1.slo)

                    # Compute new KVPRs for the two GPUs
                    new_kvpr_max = new_wr_max / new_remaining_max
                    new_kvpr_other = new_wr_other / new_remaining_other
                    # Compute new overall maximum KVPR
                    other_kvprs = [kvpr_per_gpu[g] for g in range(gpu_num) if g != max_gpu and g != other_gpu]
                    new_max_kvpr = max(new_kvpr_max, new_kvpr_other, max(other_kvprs, default=0.0))

                    # Accept swap if it reduces overall maximum KVPR
                    if new_max_kvpr < current_max_kvpr - 1e-9:  # small epsilon to avoid floating point issues
                        # Perform the swap
                        placement[max_gpu].remove(m1)
                        placement[other_gpu].remove(m2)
                        placement[max_gpu].append(m2)
                        placement[other_gpu].append(m1)
                        # Update per-GPU states
                        weighted_req_rate[max_gpu] = new_wr_max
                        weighted_req_rate[other_gpu] = new_wr_other
                        shared_kv[max_gpu] = new_remaining_max
                        shared_kv[other_gpu] = new_remaining_other
                        improved = True
                        break
                if improved:
                    break
            if improved:
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