import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using deterministic greedy best-pick, multiple smart restarts,
    and fine-grained local search (pairwise swap + block repositioning) to minimize makespan.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        only_seq = [0]
        return workload.get_opt_seq_cost(only_seq), only_seq

    # Timeout-avoidance: limit total cost evaluations; scales with workload size
    BASE_BUDGET = 120000
    MAX_COST_EVALS = BASE_BUDGET + num_txns * 400
    eval_budget = [MAX_COST_EVALS]

    def _eval_cost(seq):
        if eval_budget[0] <= 0:
            return float('inf')
        eval_budget[0] -= 1
        return workload.get_opt_seq_cost(seq)

    # Cache conflict footprints (sets of keys touched) to skip useless moves
    footprint_cache = {}
    def _get_conflict_footprint(txn_id):
        if txn_id in footprint_cache:
            return footprint_cache[txn_id]
        keys = set()
        for op in workload.txns[txn_id]:
            keys.add(op[1])
        footprint_cache[txn_id] = keys
        return keys

    def _fp(t):
        return _get_conflict_footprint(t)

    def _build_greedy_from(start_txn):
        """Deterministic greedy: at each step pick the remaining txn that yields the lowest
        incremental absolute cost (ties broken by smallest delta from current cost)."""
        txn_seq = [start_txn]
        remaining_txns = [x for x in range(num_txns) if x != start_txn]
        curr_cost = _eval_cost(txn_seq)
        if curr_cost == float('inf'):
            txn_seq.extend(remaining_txns)
            return float('inf'), txn_seq
        curr_footprint = set(_fp(start_txn))

        while remaining_txns and eval_budget[0] > 0:
            min_cost = float('inf')
            min_delta = float('inf')
            min_txn = -1

            # Prune: only evaluate txns that share a key with the current footprint first
            priority_txns = [t for t in remaining_txns if _fp(t) & curr_footprint]
            other_txns = [t for t in remaining_txns if not (_fp(t) & curr_footprint)]
            candidates = priority_txns if priority_txns else other_txns

            for t in candidates:
                if eval_budget[0] <= 0:
                    break
                txn_seq.append(t)
                cost = _eval_cost(txn_seq)
                txn_seq.pop()
                if cost == float('inf'):
                    continue
                delta = cost - curr_cost
                if cost < min_cost or (cost == min_cost and delta < min_delta):
                    min_cost = cost
                    min_delta = delta
                    min_txn = t

            if min_txn == -1:
                min_txn = remaining_txns[0]

            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)
            if min_cost != float('inf'):
                curr_cost = min_cost
                curr_footprint |= _fp(min_txn)
            else:
                curr_cost = _eval_cost(txn_seq)
                curr_footprint |= _fp(min_txn)

        if remaining_txns:
            txn_seq.extend(remaining_txns)

        overall_cost = _eval_cost(txn_seq)
        return overall_cost, txn_seq

    def _local_search(cost, seq):
        """Try pairwise swaps and single repositioning to escape local minima.
        Prunes moves that cannot change cost by checking conflict footprints."""
        n = len(seq)
        if n < 2 or cost == float('inf'):
            return cost, list(seq)

        improved = True
        best_cost = cost
        best_seq = list(seq)

        while improved and eval_budget[0] > 0:
            improved = False
            # Pairwise swaps - only swap if share a conflict key
            for i in range(n - 1):
                if eval_budget[0] <= 0:
                    break
                fp_i = _fp(best_seq[i])
                for j in range(i + 1, n):
                    if eval_budget[0] <= 0:
                        break
                    if not (fp_i & _fp(best_seq[j])):
                        continue
                    new_seq = best_seq[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_cost = _eval_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
            if improved:
                continue
            # Single repositioning - only move if potential conflict overlap
            for i in range(n):
                if eval_budget[0] <= 0:
                    break
                txn = best_seq[i]
                txn_fp = _fp(txn)
                # window of nearby positions + positions with conflicts
                window_start = max(0, i - 8)
                window_end = min(n, i + 9)
                for j in range(n):
                    if i == j or eval_budget[0] <= 0:
                        continue
                    # prune far, non-conflicting positions
                    if j < window_start or j >= window_end:
                        # check if target has any conflict with moved txn
                        if not (txn_fp & _fp(best_seq[j])):
                            continue
                    new_seq = best_seq[:i] + best_seq[i+1:]
                    new_seq.insert(j, txn)
                    new_cost = _eval_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break

        return best_cost, best_seq

    def _seed_candidates():
        seeds = set()
        seeds.add(0)
        # pick transactions with lowest individual cost as promising seeds
        individual_costs = []
        for t in range(num_txns):
            c = _eval_cost([t])
            if c != float('inf'):
                individual_costs.append((c, t))
        individual_costs.sort()
        for _, t in individual_costs[:max(2, num_txns // 20)]:
            seeds.add(t)
        # add random seeds for exploration
        random_pool = [t for t in range(num_txns) if t not in seeds]
        random.shuffle(random_pool)
        for t in random_pool[:max(1, num_txns // 15)]:
            seeds.add(t)
        return list(seeds)

    best_cost = float('inf')
    best_seq = None

    seeds = _seed_candidates()
    for s in seeds:
        if eval_budget[0] <= 0:
            break
        trial_cost, trial_seq = _build_greedy_from(s)
        if trial_cost < best_cost:
            best_cost = trial_cost
            best_seq = trial_seq

    # final local search on best greedy schedule
    if best_seq is not None and num_txns > 2 and eval_budget[0] > 0:
        ls_cost, ls_seq = _local_search(best_cost, best_seq)
        if ls_cost < best_cost:
            best_cost = ls_cost
            best_seq = ls_seq

    # Safety: ensure valid schedule even if budget completely exhausted
    if best_seq is None or len(set(best_seq)) != num_txns:
        best_seq = list(range(num_txns))
        best_cost = workload.get_opt_seq_cost(best_seq)

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