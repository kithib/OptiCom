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

    # Sort models by combined load/memory density (weighted balanced) in descending order
    # Prioritize models that impose high load *and* occupy significant memory
    sorted_models = sorted(
        models,
        key=lambda m: ((m.req_rate / m.slo) * m.model_size) if m.model_size > 0 else (m.req_rate / m.slo),
        reverse=True
    )

    # Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req = [0.0 for _ in range(gpu_num)]            # sum of req_rate/slo per GPU
    max_kvpr = 0.0                                           # track current global maximum KVPR

    # Assign each model to the GPU that minimizes the post-assignment global maximum KVPR
    for model in sorted_models:
        best_idx = None
        best_global_max = float('inf')
        rj_over_sj = model.req_rate / model.slo

        for gpu_id in range(gpu_num):
            new_remaining = remaining_mem[gpu_id] - model.model_size
            if new_remaining > 0:  # Strictly positive to avoid infinite KVPR
                new_weighted = weighted_req[gpu_id] + rj_over_sj
                post_kvpr = new_weighted / new_remaining
                candidate_global_max = post_kvpr if post_kvpr > max_kvpr else max_kvpr
                if candidate_global_max < best_global_max:
                    best_global_max = candidate_global_max
                    best_idx = gpu_id

        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining_mem}"
            )

        placement[best_idx].append(model)
        weighted_req[best_idx] += rj_over_sj
        remaining_mem[best_idx] -= model.model_size
        current_kvpr = weighted_req[best_idx] / remaining_mem[best_idx] if remaining_mem[best_idx] > 0 else 0.0
        if current_kvpr > max_kvpr:
            max_kvpr = current_kvpr

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