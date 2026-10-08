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

    # Pre-validate feasibility to avoid wasted work on impossible placements
    total_model_size = sum(model.model_size for model in models)
    if total_model_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError(
            f"Total model size {total_model_size} GB exceeds total GPU memory "
            f"{gpu_num * GPU_MEM_SIZE} GB. Cannot place models."
        )

    # Pre-compute static per-model values: use (weight, model_size) tuples
    # Sort by descending weight (r_j/slo) to place high-impact models first,
    # then by descending model_size to anchor larger models early and reduce fragmentation
    sorted_models = sorted(models, key=lambda m: -(m.req_rate / m.slo) * m.model_size)

    # Initialize per-GPU states as tight local lists for speed
    placement = [[] for _ in range(gpu_num)]
    remaining_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]
    weighted_req = [0.0 for _ in range(gpu_num)]
    # Precompute current KVPR values for non-target GPUs to avoid redundant calculations
    current_kvpr = [0.0 for _ in range(gpu_num)]

    # Assign each model to the GPU that minimizes the MAXIMUM POST-assignment KVPR
    for model in sorted_models:
        model_size = model.model_size
        weight = model.req_rate / model.slo

        best_idx = -1
        best_max_post_kvpr = float('inf')

        for gpu_id in range(gpu_num):
            avail = remaining_mem[gpu_id]
            if avail >= model_size:
                post_weight = weighted_req[gpu_id] + weight
                post_avail = avail - model_size
                # Project this GPU's KVPR after placing the model
                post_kvpr = post_weight / post_avail if post_avail > 0 else float('inf')
                # Compare post_kvpr with current KVPR values of all other GPUs
                current_max_other = 0.0
                for other_id in range(gpu_num):
                    if other_id == gpu_id:
                        continue
                    other_kvpr = current_kvpr[other_id]
                    current_max_other = other_kvpr if other_kvpr > current_max_other else current_max_other
                candidate_max = post_kvpr if post_kvpr > current_max_other else current_max_other
                if candidate_max < best_max_post_kvpr or (candidate_max == best_max_post_kvpr and post_avail < remaining_mem[best_idx] - model_size if best_idx != -1 else True):
                    best_max_post_kvpr, best_idx = candidate_max, gpu_id

        if best_idx == -1:
            raise ValueError(
                f"Unable to place model of size {model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining_mem}"
            )

        placement[best_idx].append(model)
        weighted_req[best_idx] += weight
        remaining_mem[best_idx] -= model_size
        new_avail = remaining_mem[best_idx]
        current_kvpr[best_idx] = weighted_req[best_idx] / new_avail if new_avail > 0 else float('inf')

    return {gpu_id: placement[gpu_id] for gpu_id in range(gpu_num)}

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