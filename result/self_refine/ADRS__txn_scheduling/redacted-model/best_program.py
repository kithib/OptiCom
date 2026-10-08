import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using iterative greedy with multiple restarts and local search.
    Preserves original API: returns (lowest makespan, corresponding schedule).
    """
    def compute_makespan(seq):
        return workload.get_opt_seq_cost(seq)

    def compute_marginal_cost(seq, cand):
        test_seq = seq.copy()
        test_seq.append(cand)
        return workload.get_opt_seq_cost(test_seq)

    def get_conflict_overlap(t, current_keys):
        overlap = 0
        for op in workload.txns[t]:
            if op[2] in current_keys:
                overlap += 1
        return overlap

    def greedy_from_candidates(start_txn):
        txn_seq = [start_txn]
        remaining = list(range(workload.num_txns))
        remaining.remove(start_txn)
        current_cost = compute_makespan(txn_seq)

        while remaining:
            current_keys = set()
            for t in txn_seq:
                for op in workload.txns[t]:
                    current_keys.add(op[2])

            n = len(remaining)
            # Sample adaptive candidates: 60% conflict-aware, 40% random (exploration)
            candidate_count = min(max(8, n // 4), n)
            aware_count = candidate_count * 6 // 10
            random_count = candidate_count - aware_count

            # Sort remaining by overlap (desc) then txn len (asc) for conflict-aware
            overlap_sorted = sorted(
                remaining,
                key=lambda t: (-get_conflict_overlap(t, current_keys), len(workload.txns[t]))
            )
            aware_candidates = overlap_sorted[:aware_count]

            remaining_set = set(remaining) - set(aware_candidates)
            random_candidates = random.sample(sorted(remaining_set), min(random_count, len(remaining_set))) if remaining_set else []

            candidates = list(set(aware_candidates + random_candidates))
            if not candidates:
                candidates = remaining[:]

            min_cost = float('inf')
            min_txn = -1
            for t in candidates:
                cost = compute_marginal_cost(txn_seq, t)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t
            txn_seq.append(min_txn)
            remaining.remove(min_txn)
            current_cost = min_cost
        return current_cost, txn_seq

    def local_search_2opt(seq, initial_cost):
        best_cost = initial_cost
        best_seq = list(seq)
        improved = True
        total_iter = 0
        max_total_iter = 5
        while improved and total_iter < max_total_iter:
            improved = False
            total_iter += 1
            n = len(best_seq)
            for i in range(n - 1):
                for j in range(i + 1, min(n, i + 50)):  # Cap window for speed
                    new_seq = list(best_seq)
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_cost = compute_makespan(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq

    def local_search_insert(seq, initial_cost):
        """Try moving each transaction to best possible position."""
        best_cost = initial_cost
        best_seq = list(seq)
        improved = True
        while improved:
            improved = False
            n = len(best_seq)
            for idx in range(n):
                t = best_seq[idx]
                trial_seq = best_seq[:idx] + best_seq[idx+1:]
                best_pos_cost = float('inf')
                best_pos = idx
                # Try up to 20 candidate positions for speed
                pos_candidates = list(range(0, n, max(1, n // 20))) + [n-1]
                pos_candidates = list(set(pos_candidates))
                for pos in pos_candidates:
                    if pos > len(trial_seq):
                        pos = len(trial_seq)
                    new_seq = trial_seq[:pos] + [t] + trial_seq[pos:]
                    new_cost = compute_makespan(new_seq)
                    if new_cost < best_pos_cost:
                        best_pos_cost = new_cost
                        best_pos = pos
                if best_pos_cost < best_cost:
                    new_seq = trial_seq[:best_pos] + [t] + trial_seq[best_pos:]
                    best_seq = new_seq
                    best_cost = best_pos_cost
                    improved = True
                    break
        return best_cost, best_seq

    num_restarts = max(1, min(5, num_seqs if num_seqs >= 1 else 1))
    best_overall_cost = float('inf')
    best_overall_seq = None
    base_seed = random.randint(0, 10**6)

    # Seed smart starting points: mix of shortest txns, random, high-write txns
    start_candidates = []
    all_txns = list(range(workload.num_txns))
    if not all_txns:
        return 0, []

    sorted_by_len = sorted(all_txns, key=lambda t: len(workload.txns[t]))
    start_candidates.append(sorted_by_len[0])
    sorted_by_writes = sorted(all_txns, key=lambda t: sum(1 for op in workload.txns[t] if op[1] == 'w'))
    start_candidates.append(sorted_by_writes[0])
    for i in range(num_restarts - 2):
        rng = random.Random(base_seed + i)
        start_candidates.append(rng.choice(all_txns))

    start_candidates = start_candidates[:num_restarts]

    for start_txn in start_candidates:
        cost, seq = greedy_from_candidates(start_txn)
        cost, seq = local_search_2opt(seq, cost)
        cost, seq = local_search_insert(seq, cost)
        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_seq = seq

    return best_overall_cost, best_overall_seq

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