import random
from heapq import nsmallest
from itertools import combinations

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def _build_dependency_graph(workload):
    """Compute pairwise write-read/write-write dependency weights between transactions."""
    num_txns = workload.num_txns
    txn_ops = []
    for t_idx in range(num_txns):
        ops = workload.txns[t_idx]
        reads = set()
        writes = set()
        for op in ops:
            op_type, key = op[0], op[1]
            if op_type == 'r':
                reads.add(key)
            elif op_type == 'w':
                writes.add(key)
        txn_ops.append((reads, writes))
    
    dep_weight = [[0] * num_txns for _ in range(num_txns)]
    for a in range(num_txns):
        a_writes = txn_ops[a][1]
        for b in range(num_txns):
            if a == b:
                continue
            b_reads, b_writes = txn_ops[b]
            conflict_keys = a_writes & (b_reads | b_writes)
            dep_weight[a][b] = len(conflict_keys)
    return dep_weight


def local_search_improve(cost, seq, workload, max_passes=4, early_exit=True):
    """Combined swap + 2-opt reversal + insertion local search with adaptive passes."""
    improved = True
    best_cost, best_seq = cost, seq.copy()
    current_pass = 0
    n = len(best_seq)
    while improved and current_pass < max_passes:
        improved = False
        current_pass += 1
        # Swap search phase
        for i in range(n):
            if improved and early_exit:
                break
            for j in range(i + 1, n):
                new_seq = best_seq.copy()
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_cost, best_seq = new_cost, new_seq
                    improved = True
                    if early_exit:
                        break
        if improved:
            continue
        # 2-opt reversal search phase
        for i in range(n - 1):
            if improved and early_exit:
                break
            for j in range(i + 1, n):
                new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_cost, best_seq = new_cost, new_seq
                    improved = True
                    if early_exit:
                        break
        if improved:
            continue
        # Insertion search phase (move each txn to globally best position)
        for pos in range(n):
            if improved and early_exit:
                break
            t = best_seq[pos]
            test_seq = best_seq[:pos] + best_seq[pos+1:]
            for insert_pos in range(n):
                if insert_pos == pos:
                    continue
                new_seq = test_seq[:insert_pos] + [t] + test_seq[insert_pos:]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_cost, best_seq = new_cost, new_seq
                    improved = True
                    if early_exit:
                        break
    return best_cost, best_seq


def get_best_schedule(workload, num_seqs):
    """
    Hybrid multi-restart greedy with dependency-weighted candidate filtering,
    per-restart local search, insertion-based refinement, and final exhaustive polish.
    """
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        seq = [0]
        return workload.get_opt_seq_cost(seq), seq

    dep_weight = _build_dependency_graph(workload)
    # Adaptive parameter tuning based on workload size
    top_k_candidates = max(5, min(15, num_txns // 10 + 5))
    random_sample_size = max(2, min(8, num_txns // 12 + 2))
    heuristic_size = 2

    def get_greedy_with_start(start_txn):
        txn_seq = [start_txn]
        remaining_txns = set(range(num_txns))
        remaining_txns.remove(start_txn)
        last = start_txn

        while remaining_txns:
            min_cost = float('inf')
            min_txn = -1
            remaining_list = list(remaining_txns)
            last = txn_seq[-1]

            # Candidate pre-filtering: dep-weighted top-k + random + size heuristics
            total_pool = top_k_candidates + random_sample_size + heuristic_size
            if len(remaining_list) <= total_pool:
                candidates = remaining_list
            else:
                scored = [(-dep_weight[last][t], t) for t in remaining_list]
                top_by_dep = [t for (_, t) in nsmallest(top_k_candidates, scored)]
                remaining_pool = set(remaining_list) - set(top_by_dep)
                random_sample = random.sample(list(remaining_pool), min(random_sample_size, len(remaining_pool)))
                remaining_pool -= set(random_sample)
                size_sorted = sorted(remaining_pool, key=lambda t: workload.txns[t][0][3])
                heuristic_candidates = size_sorted[:heuristic_size]
                candidates = list(set(top_by_dep + random_sample + heuristic_candidates))

            # Evaluate makespan cost for each candidate
            for t in candidates:
                test_seq = txn_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)
                if cost < min_cost:
                    min_cost = cost
                    min_txn = t

            if min_txn == -1:
                min_txn = next(iter(remaining_txns))
            txn_seq.append(min_txn)
            remaining_txns.remove(min_txn)

        current_cost = workload.get_opt_seq_cost(txn_seq)
        # Fast per-restart local search (early exit enabled for speed)
        polished_cost, polished_seq = local_search_improve(current_cost, txn_seq, workload, max_passes=3, early_exit=True)
        return polished_cost, polished_seq

    # Generate diverse starting points
    start_points = set()
    txn_sizes = sorted(range(num_txns), key=lambda t: workload.txns[t][0][3])
    start_points.add(txn_sizes[0])  # Smallest txn
    start_points.add(txn_sizes[-1]) # Largest txn
    write_counts = sorted(range(num_txns), key=lambda t: -sum(1 for op in workload.txns[t] if op[0] == 'w'))
    for t in write_counts[:min(3, num_txns)]:
        start_points.add(t)
    num_trials = min(num_seqs, max(5, num_txns // 8 + 3))
    while len(start_points) < num_trials:
        start_points.add(random.randint(0, num_txns - 1))

    # Run multi-restart greedy
    overall_min_cost = float('inf')
    overall_min_seq = []
    for start in start_points:
        cost, seq = get_greedy_with_start(start)
        if cost < overall_min_cost:
            overall_min_cost, overall_min_seq = cost, seq

    # Insertion-based refinement: try moving each txn to better position
    improved = True
    while improved:
        improved = False
        n = len(overall_min_seq)
        for pos in range(n):
            if improved:
                break
            t = overall_min_seq[pos]
            test_seq = overall_min_seq[:pos] + overall_min_seq[pos+1:]
            for insert_pos in range(n):
                if insert_pos == pos:
                    continue
                new_seq = test_seq[:insert_pos] + [t] + test_seq[insert_pos:]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < overall_min_cost:
                    overall_min_cost, overall_min_seq = new_cost, new_seq
                    improved = True
                    break

    # Final exhaustive local search polish
    overall_min_cost, overall_min_seq = local_search_improve(
        overall_min_cost, overall_min_seq, workload, max_passes=4, early_exit=False
    )

    assert len(set(overall_min_seq)) == num_txns
    return overall_min_cost, overall_min_seq

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