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

    # 1) Sort models by combined priority: (req_rate / slo) * model_size in descending order
    # Higher weighted req rate and larger models get placed first for better balancing
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo) * m.model_size, reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU
    current_kvpr = [0.0 for _ in range(gpu_num)]        # track current KVPR per GPU

    # 3) Assign each model to the GPU that minimizes the *resulting* KVPR (post-placement)
    # This improves over only looking at current KVPR by accounting for the impact of the model
    for model in sorted_models:
        best_idx = None
        best_resulting_ratio = float('inf')
        model_weight = model.req_rate / model.slo
        model_size = model.model_size

        for gpu_id in range(gpu_num):
            if model_size <= shared_kv[gpu_id]:
                # Calculate KVPR *after* placing this model on this GPU
                new_weighted = weighted_req_rate[gpu_id] + model_weight
                new_remaining = shared_kv[gpu_id] - model_size
                if new_remaining > 0:
                    resulting_ratio = new_weighted / new_remaining
                    if resulting_ratio < best_resulting_ratio:
                        best_resulting_ratio = resulting_ratio
                        best_idx = gpu_id
                    elif resulting_ratio == best_resulting_ratio:
                        # Tiebreaker: prefer GPU with more remaining memory for future models
                        if best_idx is not None and new_remaining > (shared_kv[best_idx] - model_size):
                            best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model_weight
        shared_kv[best_idx] -= model_size
        if shared_kv[best_idx] > 0:
            current_kvpr[best_idx] = weighted_req_rate[best_idx] / shared_kv[best_idx]
        else:
            current_kvpr[best_idx] = float('inf')

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