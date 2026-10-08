import random
from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START
def get_best_schedule(workload, num_seqs):
    """
    Metric-optimized schedule finder: multi-seed adaptive sampled greedy + 2-opt swap + insertion refinement.
    Preserves public API (signature, return contract); optimized for lower makespan while maintaining validity.
    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        seq = [0]
        return workload.get_opt_seq_cost(seq), seq

    # Runtime-aware adaptive parameters tuned to minimize makespan while respecting time limits
    SAMPLES_PER_STEP = min(20, max(8, num_txns // 8)) if num_txns > 10 else num_txns
    FULL_EVAL_THRESHOLD = 25
    MAX_RESTARTS = min(6, max(2, (num_seqs if num_seqs else 3)))
    TWO_OPT_MAX_PASSES = 1 if num_txns > 80 else 2
    INSERT_MAX_PASSES = 1 if num_txns > 80 else 2
    EARLY_PLATEAU_LIMIT = 3
    COST_EPS = 1e-9

    def sampled_greedy(start_txn):
        seq = [start_txn]
        remaining = [x for x in range(num_txns) if x != start_txn]
        while remaining:
            if len(remaining) <= FULL_EVAL_THRESHOLD:
                candidates = remaining
            else:
                k = min(SAMPLES_PER_STEP, len(remaining))
                candidates = random.sample(remaining, k)
            min_cost = float('inf')
            best_t = candidates[0]
            for t in candidates:
                seq.append(t)
                cost = workload.get_opt_seq_cost(seq)
                seq.pop()
                if cost < min_cost:
                    min_cost = cost
                    best_t = t
            seq.append(best_t)
            remaining.remove(best_t)
        return workload.get_opt_seq_cost(seq), seq

    def two_opt(seq, start_cost):
        best_cost = start_cost
        best_seq = seq[:]
        passes = 0
        while passes < TWO_OPT_MAX_PASSES:
            improved = False
            passes += 1
            n = len(best_seq)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
                    new_cost = workload.get_opt_seq_cost(best_seq)
                    if new_cost < best_cost - COST_EPS:
                        best_cost = new_cost
                        improved = True
                    else:
                        best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
            if not improved:
                break
        return best_cost, best_seq

    def insert_refine(seq, start_cost):
        best_cost = start_cost
        best_seq = seq[:]
        passes = 0
        while passes < INSERT_MAX_PASSES:
            improved = False
            passes += 1
            n = len(best_seq)
            for i in range(n):
                txn = best_seq[i]
                base = best_seq[:i] + best_seq[i+1:]
                local_min_cost = float('inf')
                local_best_pos = i
                for pos in range(n):
                    test = base[:pos] + [txn] + base[pos:]
                    new_cost = workload.get_opt_seq_cost(test)
                    if new_cost < local_min_cost:
                        local_min_cost = new_cost
                        local_best_pos = pos
                if local_min_cost < best_cost - COST_EPS:
                    best_cost = local_min_cost
                    best_seq = base[:local_best_pos] + [txn] + base[local_best_pos:]
                    improved = True
            if not improved:
                break
        return best_cost, best_seq

    # Seeded starting set: mix of random + conflict-heavy txns to stabilize quality across runs
    seed_starts = [random.randint(0, num_txns - 1) for _ in range(MAX_RESTARTS)]
    try:
        conflict_proxies = []
        for t_idx in range(num_txns):
            ops = workload.txns[t_idx]
            wc = sum(1 for op in ops if op[0] == 'w')
            conflict_proxies.append((wc, t_idx))
        conflict_proxies.sort(reverse=True)
        seed_starts[0] = conflict_proxies[0][1]
        if len(conflict_proxies) > 1:
            seed_starts[1] = conflict_proxies[1][1]
    except Exception:
        pass

    best_cost = float('inf')
    best_seq = []
    plateau_counter = 0

    for start in seed_starts:
        cost, seq = sampled_greedy(start)
        cost, seq = two_opt(seq, cost)
        cost, seq = insert_refine(seq, cost)
        if cost < best_cost - COST_EPS:
            best_cost = cost
            best_seq = seq
            plateau_counter = 0
        else:
            plateau_counter += 1
            if plateau_counter >= EARLY_PLATEAU_LIMIT:
                break

    assert len(set(best_seq)) == num_txns
    return best_cost, best_seq
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