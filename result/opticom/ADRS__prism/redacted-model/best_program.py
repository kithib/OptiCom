GPU_MEM_SIZE = 80


def compute_model_placement(gpu_num, models):
    def pressure(model):
        return float(model.req_rate) / float(model.slo)

    def pressure_times_size(model):
        return pressure(model) * model.model_size

    items = list(models)
    n_items = len(items)

    def empty_placement():
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    def validate(placement):
        seen = []
        for gpu_id in range(gpu_num):
            bucket = placement.get(gpu_id, [])
            if not isinstance(bucket, list):
                return False
            if sum(model.model_size for model in bucket) > GPU_MEM_SIZE:
                return False
            seen.extend(bucket)
        return (
            len(seen) == n_items
            and len({id(model) for model in seen}) == n_items
            and {id(model) for model in seen} == {id(model) for model in items}
        )

    def max_kvpr(placement):
        worst = 0.0
        for gpu_id in range(gpu_num):
            bucket = placement.get(gpu_id, [])
            size = sum(model.model_size for model in bucket)
            if size >= GPU_MEM_SIZE:
                return float("inf")
            load = sum(pressure(model) for model in bucket)
            worst = max(worst, load / (GPU_MEM_SIZE - size))
        return worst

    def greedy(order):
        placement = empty_placement()
        sizes = [0] * gpu_num
        loads = [0.0] * gpu_num
        for model in order:
            model_pressure = pressure(model)
            best_choice = None
            for gpu_id in range(gpu_num):
                if sizes[gpu_id] + model.model_size > GPU_MEM_SIZE:
                    continue
                old_size, old_load = sizes[gpu_id], loads[gpu_id]
                new_size = old_size + model.model_size
                new_load = old_load + model_pressure
                sizes[gpu_id] = new_size
                loads[gpu_id] = new_load
                projected = 0.0
                for other_gpu in range(gpu_num):
                    remaining = GPU_MEM_SIZE - sizes[other_gpu]
                    projected = max(projected, float("inf") if remaining <= 0 else loads[other_gpu] / remaining)
                tie = (projected, sizes[gpu_id], loads[gpu_id], gpu_id)
                sizes[gpu_id], loads[gpu_id] = old_size, old_load
                if best_choice is None or tie < best_choice[0]:
                    best_choice = (tie, gpu_id, new_size, new_load)
            if best_choice is None:
                return None
            gpu_id = best_choice[1]
            placement[gpu_id].append(model)
            sizes[gpu_id] += model.model_size
            loads[gpu_id] += model_pressure
        return placement

    key_specs = [
        lambda model: pressure(model),
        lambda model: model.model_size,
        pressure_times_size,
        lambda model: pressure(model) / max(1e-12, model.model_size),
        lambda model: model.model_size / max(1e-12, pressure(model)),
    ]

    # Pre-sort items by pressure × size descending as primary baseline
    items.sort(key=pressure_times_size, reverse=True)
    best_score = float("inf")
    for key in key_specs:
        for reverse in (True, False):
            candidate = greedy(sorted(items, key=key, reverse=reverse))
            if candidate is not None and validate(candidate):
                score = max_kvpr(candidate)
                if score < best_score:
                    best_score, best_placement = score, candidate

    if best_placement is None:
        best_placement = empty_placement()
        sizes = [0] * gpu_num
        for model in sorted(items, key=lambda item: item.model_size, reverse=True):
            choices = [gpu_id for gpu_id in range(gpu_num) if sizes[gpu_id] + model.model_size <= GPU_MEM_SIZE]
            if not choices:
                raise ValueError("No feasible placement")
            gpu_id = min(choices, key=lambda idx: sizes[idx])
            best_placement[gpu_id].append(model)
            sizes[gpu_id] += model.model_size

    def copy_placement(placement):
        return {gpu_id: list(placement.get(gpu_id, [])) for gpu_id in range(gpu_num)}

    current = copy_placement(best_placement)
    current_score = max_kvpr(current)
    for _ in range(80):
        gpu_scores = []
        for gpu_id in range(gpu_num):
            bucket = current[gpu_id]
            size = sum(model.model_size for model in bucket)
            load = sum(pressure(model) for model in bucket)
            gpu_scores.append(float("inf") if size >= GPU_MEM_SIZE else load / (GPU_MEM_SIZE - size))
        bottlenecks = sorted(range(gpu_num), key=lambda gpu_id: gpu_scores[gpu_id], reverse=True)[:max(1, min(3, gpu_num))]
        best_candidate = (current_score, None)
        for src_gpu in bottlenecks:
            for src_idx, model in enumerate(list(current[src_gpu])):
                for dst_gpu in range(gpu_num):
                    if dst_gpu == src_gpu:
                        continue
                    if sum(item.model_size for item in current[dst_gpu]) + model.model_size <= GPU_MEM_SIZE:
                        # Project score without full copy
                        src_bucket = current[src_gpu]
                        dst_bucket = current[dst_gpu]
                        src_size = sum(m.model_size for m in src_bucket) - model.model_size
                        dst_size = sum(m.model_size for m in dst_bucket) + model.model_size
                        src_load = sum(pressure(m) for m in src_bucket) - pressure(model)
                        dst_load = sum(pressure(m) for m in dst_bucket) + pressure(model)
                        if src_size < GPU_MEM_SIZE and dst_size < GPU_MEM_SIZE:
                            project_max = max(
                                gpu_scores[g] if g not in (src_gpu, dst_gpu)
                                else (src_load / (GPU_MEM_SIZE - src_size) if g == src_gpu
                                      else dst_load / (GPU_MEM_SIZE - dst_size))
                                for g in range(gpu_num)
                            )
                            if project_max < best_candidate[0] - 1e-12:
                                candidate = copy_placement(current)
                                candidate[src_gpu].pop(src_idx)
                                candidate[dst_gpu].append(model)
                                best_candidate = (project_max, candidate)
                    for dst_idx, other in enumerate(list(current[dst_gpu])):
                        src_size = sum(item.model_size for item in current[src_gpu]) - model.model_size + other.model_size
                        dst_size = sum(item.model_size for item in current[dst_gpu]) - other.model_size + model.model_size
                        if src_size > GPU_MEM_SIZE or dst_size > GPU_MEM_SIZE:
                            continue
                        candidate = copy_placement(current)
                        # Simpler: recompute sizes from candidate
                        candidate[src_gpu][src_idx] = other
                        candidate[dst_gpu][dst_idx] = model
                        project_score = 0.0
                        for g in range(gpu_num):
                            b = candidate[g]
                            s = sum(m.model_size for m in b)
                            if s < GPU_MEM_SIZE:
                                project_score = max(project_score, sum(pressure(m) for m in b) / (GPU_MEM_SIZE - s))
                            else:
                                project_score = float("inf")
                                break
                        if project_score < best_candidate[0] - 1e-12:
                            best_candidate = (project_score, candidate)
        if best_candidate[1] is None:
            break
        current_score, current = best_candidate

    if not validate(current):
        raise ValueError("Invalid final placement")
    return current
