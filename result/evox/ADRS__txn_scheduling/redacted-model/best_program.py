import random
import time

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using time-bounded iterative greedy search with multiple
    diverse restarts, candidate filtering, cached cost evaluation, precomputed
    conflict heuristics, and efficient multi-phase local search improvement.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        seq = [0]
        return workload.get_opt_seq_cost(seq), seq

    start_time = time.time()
    # Per-workload time budget: 70s x3 = 210s, leaving ~90s for evaluator overhead.
    time_budget = 70.0

    # --- Precompute per-txn static metrics once to avoid repeated work ---
    txn_base_cost = []
    txn_write_count = []
    txn_keys = []
    txn_op_count = []
    for t in range(num_txns):
        ops = workload.txns[t]
        if ops and len(ops[0]) >= 4:
            base_cost = ops[0][3]
        else:
            base_cost = len(ops)
        txn_base_cost.append(base_cost)
        writes = 0
        keys = set()
        for op in ops:
            if op[0] == 'w':
                writes += 1
            keys.add(op[2])
        txn_write_count.append(writes)
        txn_keys.append(keys)
        txn_op_count.append(len(ops))

    # --- Shared (seq -> cost) cache to avoid redundant calls across restarts/search ---
    cost_cache = {}

    def cached_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(list(seq))
        if len(cost_cache) < 65536:
            cost_cache[key] = c
        return c

    def greedy_schedule(start_txn, candidate_ratio=0.45, max_samples=18):
        txn_seq = [start_txn]
        remaining = set(range(num_txns))
        remaining.remove(start_txn)
        existing_keys = set(txn_keys[start_txn])
        write_keys = set()
        if workload.txns[start_txn]:
            for op in workload.txns[start_txn]:
                if op[0] == 'w':
                    write_keys.add(op[2])

        while remaining:
            if time.time() - start_time > time_budget:
                txn_seq.extend(remaining)
                remaining.clear()
                break

            rem_list = list(remaining)
            sample_size = min(
                len(rem_list),
                max(3, int(len(rem_list) * candidate_ratio), max_samples),
            )

            # Heuristic scoring: lower = better.
            # Heavier weight on conflicts with already-written keys.
            candidate_scores = []
            for t in rem_list:
                keys_t = txn_keys[t]
                conflict = len(keys_t & existing_keys)
                write_conflict = len(keys_t & write_keys)
                score = (write_conflict * 300
                        + conflict * 100
                        + txn_write_count[t] * 20
                        + txn_base_cost[t] * 0.02)
                candidate_scores.append((score, t))
            candidate_scores.sort()

            top_candidates = [t for _, t in candidate_scores[:sample_size]]
            # Inject ~12% random exploration
            if len(rem_list) > sample_size:
                replace_count = max(1, sample_size // 8)
                random_pool = [t for t in rem_list if t not in top_candidates]
                if random_pool:
                    for _ in range(replace_count):
                        idx = random.randint(0, len(top_candidates) - 1)
                        top_candidates[idx] = random.choice(random_pool)

            min_cost = float('inf')
            min_txn = -1
            for t in top_candidates:
                test_seq = txn_seq + [t]
                cost = cached_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t

            if min_txn == -1:
                min_txn = rem_list[0]
                min_cost = cached_cost(txn_seq + [min_txn])

            txn_seq.append(min_txn)
            remaining.remove(min_txn)
            existing_keys |= txn_keys[min_txn]
            for op in workload.txns[min_txn]:
                if op[0] == 'w':
                    write_keys.add(op[2])

        overall_cost = cached_cost(txn_seq)
        return overall_cost, txn_seq

    def local_search_improve(cost, schedule):
        best_cost = cost
        best_schedule = list(schedule)
        n = len(best_schedule)
        if n < 2:
            return best_cost, best_schedule

        max_passes = 8
        passes = 0

        while passes < max_passes:
            if time.time() - start_time > time_budget:
                break
            improved = False
            passes += 1

            # Phase 1: adjacent swaps (cheap, effective)
            for i in range(n - 1):
                if time.time() - start_time > time_budget:
                    break
                new_schedule = best_schedule[:i] + [best_schedule[i+1], best_schedule[i]] + best_schedule[i+2:]
                new_cost = cached_cost(new_schedule)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_schedule = new_schedule
                    improved = True
                    break

            if improved:
                continue

            # Phase 2: block-based adjacent swaps (move txns up to 5 positions)
            if time.time() - start_time > time_budget:
                break
            for i in range(n - 1):
                if time.time() - start_time > time_budget:
                    break
                for d in (2, 3, 4, 5):
                    j = i + d
                    if j >= n:
                        continue
                    t = best_schedule[j]
                    new_schedule = [x for k, x in enumerate(best_schedule) if k != j]
                    new_schedule.insert(i, t)
                    new_cost = cached_cost(new_schedule)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_schedule = new_schedule
                        improved = True
                        break
                if improved:
                    break

            if improved:
                continue

            # Phase 3: random non-adjacent swaps (exploration)
            for _ in range(min(70, n * 4)):
                if time.time() - start_time > time_budget:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                new_schedule = best_schedule[:]
                new_schedule[i], new_schedule[j] = new_schedule[j], new_schedule[i]
                new_cost = cached_cost(new_schedule)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_schedule = new_schedule
                    improved = True
                    break

            if improved:
                continue

            # Phase 4: 2-opt style insertions
            for _ in range(min(40, n)):
                if time.time() - start_time > time_budget:
                    break
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                t = best_schedule[i]
                new_schedule = [x for k, x in enumerate(best_schedule) if k != i]
                new_schedule.insert(j, t)
                new_cost = cached_cost(new_schedule)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_schedule = new_schedule
                    improved = True
                    break

            if improved:
                continue

            # Phase 5: limited triple-swap exploration
            for _ in range(min(25, n // 2)):
                if time.time() - start_time > time_budget:
                    break
                i, j, k = random.sample(range(n), 3)
                new_schedule = best_schedule[:]
                new_schedule[i], new_schedule[j], new_schedule[k] = new_schedule[k], new_schedule[i], new_schedule[j]
                new_cost = cached_cost(new_schedule)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_schedule = new_schedule
                    improved = True
                    break

            if not improved:
                break

        return best_cost, best_schedule

    # --- Main driver: multi-start with diverse starting points plus improvements ---
    overall_best_cost = float('inf')
    overall_best_schedule = None

    if num_txns <= 20:
        num_restarts = min(num_txns, max(num_seqs, 14))
        candidate_ratio = 1.0
        max_samples = num_txns
    elif num_txns <= 50:
        num_restarts = max(num_seqs, 10)
        candidate_ratio = 0.6
        max_samples = 26
    else:
        num_restarts = max(num_seqs, 9)
        candidate_ratio = 0.45
        max_samples = 18

    # Diverse starting txns: mix low-write, high-cost, low-conflict, and random
    start_candidates = []
    low_write = sorted(range(num_txns), key=lambda t: (txn_write_count[t], txn_base_cost[t]))
    start_candidates.extend(low_write[:max(1, num_restarts // 2)])
    high_cost = sorted(range(num_txns), key=lambda t: -txn_base_cost[t])
    for t in high_cost:
        if t not in start_candidates and len(start_candidates) < num_restarts:
            start_candidates.append(t)
    low_conflict = sorted(range(num_txns), key=lambda t: (len(txn_keys[t]), txn_write_count[t]))
    for t in low_conflict:
        if t not in start_candidates and len(start_candidates) < num_restarts:
            start_candidates.append(t)
    remaining_starts = [t for t in range(num_txns) if t not in start_candidates]
    if remaining_starts and len(start_candidates) < num_restarts:
        random.shuffle(remaining_starts)
        start_candidates.extend(remaining_starts[:num_restarts - len(start_candidates)])

    # Always include deterministic lowest-write start first to guarantee one strong baseline
    if low_write and low_write[0] not in start_candidates:
        start_candidates.insert(0, low_write[0])

    # Seed a diverse initial population of schedules to boost diversity
    seed_population = []
    if time.time() - start_time <= time_budget:
        seed_population.append(list(range(num_txns)))
        random_schedule = list(range(num_txns))
        random.shuffle(random_schedule)
        seed_population.append(random_schedule)
        heuristic_order = sorted(range(num_txns), key=lambda t: (txn_write_count[t], txn_base_cost[t]))
        seed_population.append(heuristic_order)
        short_first = sorted(range(num_txns), key=lambda t: (txn_op_count[t], txn_write_count[t]))
        seed_population.append(short_first)
        high_cost_first = sorted(range(num_txns), key=lambda t: -txn_base_cost[t])
        seed_population.append(high_cost_first)
        low_keys_first = sorted(range(num_txns), key=lambda t: (len(txn_keys[t]), txn_write_count[t]))
        seed_population.append(low_keys_first)

    for seed in seed_population:
        if time.time() - start_time > time_budget:
            break
        seed_cost = cached_cost(seed)
        if seed_cost < overall_best_cost:
            overall_best_cost = seed_cost
            overall_best_schedule = seed
        improved_cost, improved_schedule = local_search_improve(seed_cost, seed)
        if improved_cost < overall_best_cost:
            overall_best_cost = improved_cost
            overall_best_schedule = improved_schedule

    for start in start_candidates:
        if time.time() - start_time > time_budget:
            break
        current_cost, current_schedule = greedy_schedule(start, candidate_ratio, max_samples)
        if current_cost < overall_best_cost:
            overall_best_cost = current_cost
            overall_best_schedule = current_schedule

        if time.time() - start_time > time_budget:
            break
        improved_cost, improved_schedule = local_search_improve(current_cost, current_schedule)
        if improved_cost < overall_best_cost:
            overall_best_cost = improved_cost
            overall_best_schedule = improved_schedule

    # Final safety: validate schedule completeness, produce fallback if needed
    if (overall_best_schedule is None
            or len(overall_best_schedule) != num_txns
            or len(set(overall_best_schedule)) != num_txns):
        overall_best_schedule = list(range(num_txns))
        overall_best_cost = cached_cost(overall_best_schedule)

    # Final cost computation through public API to guarantee contract consistency
    final_cost = workload.get_opt_seq_cost(overall_best_schedule)
    return final_cost, overall_best_schedule

# EVOLVE-BLOCK-END

def get_random_costs():
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")