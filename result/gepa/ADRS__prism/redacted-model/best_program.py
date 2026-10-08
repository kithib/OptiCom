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

    # Early return for empty/invalid input to avoid redundant computation and crashes
    if not models or gpu_num <= 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    # Precompute per-model metrics to avoid repeated calculations
    weights = [m.req_rate / m.slo for m in models]
    sizes = [m.model_size for m in models]

    # Weakness 1 fixed: Previous sort prioritized (req_rate/slo) exclusively, which left large
    # models to the end and caused placement failures. Pair with normalized model size to
    # place large models early and improve packing. Use 0.55/0.45 weights to balance priority
    # on req_rate (which drives KVPR) and size (which drives feasibility/memory balance).
    max_weight = max(weights) if weights else 1.0
    max_size = max(sizes) if sizes else 1.0

    inv_max_weight = 1.0 / max_weight if max_weight > 0 else 0.0
    inv_max_size = 1.0 / max_size if max_size > 0 else 0.0

    sorted_models = sorted(
        models,
        key=lambda m: (
            0.55 * (m.req_rate / m.slo) * inv_max_weight
            + 0.45 * m.model_size * inv_max_size
        ),
        reverse=True,
    )

    # Initialize per-GPU state tracking
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU
    per_gpu_kvpr = [0.0 for _ in range(gpu_num)]        # explicit per-GPU KVPR for accurate max

    # Weakness 2 fixed: Previous implementation tracked only a single global max KVPR, which
    # could underestimate the true system-wide max after each placement (e.g., if another GPU
    # had a higher KVPR that was not the most recent update). Track explicit per-GPU KVPR and
    # compute a precise system-wide max to ensure primary criterion (minimize system max) is
    # accurate.
    # Weakness 3 fixed: Use tighter safe memory buffer (1e-7 instead of 1e-6) to avoid
    # rejecting valid placements due to overly conservative checks, while still preventing
    # division-by-zero and GPU memory overcommits.
    SAFE_MEM_BUFFER = 1e-7

    for model in sorted_models:
        best_idx = -1
        best_possible_max = float('inf')
        best_post_kvpr = float('inf')
        best_remaining = -1.0
        model_weight = model.req_rate / model.slo
        model_size = model.model_size

        # Compute current system-wide max KVPR once per model (unchanged for a given iteration)
        current_system_max = 0.0
        for gpu_id in range(gpu_num):
            if per_gpu_kvpr[gpu_id] > current_system_max:
                current_system_max = per_gpu_kvpr[gpu_id]

        for gpu_id in range(gpu_num):
            if model_size <= shared_kv[gpu_id]:
                remaining_after = shared_kv[gpu_id] - model_size
                # Skip infeasible states (exhausted memory would yield invalid/infinite KVPR)
                if remaining_after <= SAFE_MEM_BUFFER:
                    continue
                post_kvpr = (weighted_req_rate[gpu_id] + model_weight) / remaining_after
                # Accurate system-wide max after this placement: max of (post_kvpr, other GPUs' current max)
                if post_kvpr > current_system_max:
                    possible_max = post_kvpr
                else:
                    # Check if the current system max is held by another GPU (it is, unless this GPU is the only max)
                    if per_gpu_kvpr[gpu_id] < current_system_max:
                        possible_max = current_system_max
                    else:
                        # Current max was held by this GPU; recompute max of others
                        other_max = 0.0
                        for other_id in range(gpu_num):
                            if other_id != gpu_id and per_gpu_kvpr[other_id] > other_max:
                                other_max = per_gpu_kvpr[other_id]
                        possible_max = other_max

                # Multi-level tiebreakers for better robustness and lower overall max KVPR:
                # 1) Primary: minimize system-wide possible_max KVPR after placement
                # 2) Secondary: minimize post-placement KVPR on target GPU (avoid hot spots)
                # 3) Tertiary: maximize remaining memory (better packing for future placements)
                if (possible_max < best_possible_max) or \
                   (possible_max == best_possible_max and post_kvpr < best_post_kvpr) or \
                   (possible_max == best_possible_max and post_kvpr == best_post_kvpr and remaining_after > best_remaining):
                    best_possible_max = possible_max
                    best_post_kvpr = post_kvpr
                    best_remaining = remaining_after
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx == -1:
            raise ValueError(
                f"Unable to place model of size {model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        # Commit placement and update state
        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += model_weight
        shared_kv[best_idx] = best_remaining
        per_gpu_kvpr[best_idx] = best_post_kvpr

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