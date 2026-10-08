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

    # Early exit for trivial cases: no models or no GPUs
    if not models or gpu_num <= 0:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    # 1) Prune oversized models first (fail fast if any model is too large)
    # Any model with size > GPU_MEM_SIZE cannot fit on any empty GPU, so fail early
    for model in models:
        if model.model_size > GPU_MEM_SIZE:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB — exceeds GPU memory size {GPU_MEM_SIZE} GB."
            )

    # 2) Sort models by weighted resource demand in descending order:
    # Priority 1: higher (req_rate/slo) * model_size — these models impose the
    #   largest *marginal* KVPR increase when placed (since d(KVPR)/d(placement)
    #   scales with model_weight * model_size / remaining_mem^2)
    # Priority 2: higher raw req_rate/slo (classic load-balance heuristic)
    # This ordering ensures we make the most impactful placement decisions first,
    # reducing the chance of creating hotspots later.
    sorted_models = sorted(
        models,
        key=lambda m: (
            -((m.req_rate / m.slo) * m.model_size),
            -(m.req_rate / m.slo)
        )
    )

    # 3) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    remaining_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]       # sum of r_j / s_j per GPU

    # 4) Assign each model to the GPU that minimizes the *global post-assignment max KVPR*
    # Key improvement over original "best single GPU KVPR": we now explicitly compare
    # candidate placements against the *current maximum KVPR across all other GPUs*
    # so we directly target the objective of minimizing the GLOBAL maximum, rather than
    # just choosing the locally best GPU. Tiebreak by more remaining memory on chosen GPU
    # to reduce future failure risk.
    global_max = 0.0
    for model in sorted_models:
        best_idx = None
        best_post_max = float('inf')
        best_remaining = -1
        model_weight = model.req_rate / model.slo
        model_size = model.model_size
        current_global_max = global_max

        for gpu_id in range(gpu_num):
            remaining = remaining_mem[gpu_id]
            # Correctness: only check if model fits
            if model_size <= remaining:
                # Compute post-assignment KVPR for this GPU
                post_weighted = weighted_req_rate[gpu_id] + model_weight
                post_remaining = remaining - model_size
                post_kvpr = post_weighted / post_remaining
                # Global max after placement is max between existing max and this GPU's new KVPR
                post_max = current_global_max if current_global_max > post_kvpr else post_kvpr

                # Minimize global post-max KVPR; tiebreak by keeping more free memory
                if post_max < best_post_max or (post_max == best_post_max and post_remaining > best_remaining):
                    best_post_max = post_max
                    best_remaining = post_remaining
                    best_idx = gpu_id

        # Failure: if no GPU can fit, raise an error instead of overcommitting
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {remaining_mem}"
            )

        placement[best_idx].append(model)
        new_kvpr = (weighted_req_rate[best_idx] + model_weight) / (remaining_mem[best_idx] - model_size)
        if new_kvpr > global_max:
            global_max = new_kvpr
        weighted_req_rate[best_idx] += model_weight
        remaining_mem[best_idx] -= model_size

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