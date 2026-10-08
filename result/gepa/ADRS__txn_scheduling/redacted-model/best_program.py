import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using multiple greedy restarts plus limited local
    pairwise swap refinement, insert-based refinement, and block swaps.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def evaluate_seq_cost(seq):
        return workload.get_opt_seq_cost(seq)

    def initial_single_cost(txn_id):
        txn = workload.txns[txn_id]
        return txn[0][3] if txn else 0

    def greedy_build_from(start_txn):
        txn_seq = [start_txn]
        remaining_txns = list(range(workload.num_txns))
        remaining_txns.remove(start_txn)
        current_base = evaluate_seq_cost(txn_seq)

        for _ in range(workload.num_txns - 1):
            if not remaining_txns:
                break
            best_increment = float('inf')
            best_cost = float('inf')
            best_txn = -1
            for t in remaining_txns:
                test_seq = txn_seq + [t]
                c = evaluate_seq_cost(test_seq)
                increment = c - current_base
                if increment < best_increment or (increment == best_increment and c < best_cost):
                    best_increment = increment
                    best_cost = c
                    best_txn = t
            if best_txn == -1:
                txn_seq.extend(remaining_txns)
                break
            txn_seq.append(best_txn)
            current_base = best_cost
            remaining_txns.remove(best_txn)

        if len(set(txn_seq)) != workload.num_txns:
            txn_seq = list(range(workload.num_txns))
        return evaluate_seq_cost(txn_seq), txn_seq

    def get_greedy_cost_sampled(num_samples, sample_rate, seed_start=None):
        if seed_start is None:
            start_txn = random.randint(0, workload.num_txns - 1)
        else:
            start_txn = seed_start % workload.num_txns
        txn_seq = [start_txn]
        remaining_txns = list(range(workload.num_txns))
        remaining_txns.remove(start_txn)
        current_base = initial_single_cost(start_txn)

        for _ in range(workload.num_txns - 1):
            if not remaining_txns:
                break
            min_increment = float('inf')
            min_cost = float('inf')
            min_txn = -1
            holdout_txns = []

            sample = random.random()
            if sample > sample_rate:
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                txn_seq.append(t)
                remaining_txns.pop(idx)
                current_base = evaluate_seq_cost(txn_seq)
                continue

            effective_samples = min(num_samples, len(remaining_txns))
            candidate_indices = set()
            while len(candidate_indices) < effective_samples and len(remaining_txns) > len(candidate_indices):
                idx = random.randint(0, len(remaining_txns) - 1)
                candidate_indices.add(idx)
            sorted_indices = sorted(candidate_indices, reverse=True)
            for idx in sorted_indices:
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))
                test_seq = txn_seq + [t]
                cost = evaluate_seq_cost(test_seq)
                increment = cost - current_base
                if increment < min_increment or (increment == min_increment and cost < min_cost):
                    min_increment = increment
                    min_cost = cost
                    min_txn = t

            if min_txn == -1:
                if holdout_txns:
                    min_txn = holdout_txns[0]
                    holdout_txns.remove(min_txn)
                elif remaining_txns:
                    min_txn = remaining_txns[0]
                    remaining_txns.remove(min_txn)
                else:
                    break
                txn_seq.append(min_txn)
                remaining_txns.extend(holdout_txns)
                current_base = evaluate_seq_cost(txn_seq)
                continue
            txn_seq.append(min_txn)
            if min_txn in holdout_txns:
                holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)
            current_base = min_cost

        if len(set(txn_seq)) != workload.num_txns:
            txn_seq = list(range(workload.num_txns))
        overall_cost = evaluate_seq_cost(txn_seq)
        return overall_cost, txn_seq

    def get_full_greedy():
        best_cost = float('inf')
        best_seq = None
        starts_to_try = min(6, workload.num_txns)
        for s in range(starts_to_try):
            cost, seq = greedy_build_from(s)
            if cost < best_cost:
                best_cost = cost
                best_seq = seq
        return best_cost, best_seq

    def refine_locally(seq, passes=3, window=10):
        best_cost = evaluate_seq_cost(seq)
        best_seq = list(seq)
        n = len(best_seq)
        for _ in range(passes):
            improved = False
            for i in range(n - 1):
                end = min(i + window, n)
                for j in range(i + 1, end):
                    trial = list(best_seq)
                    trial[i], trial[j] = trial[j], trial[i]
                    c = evaluate_seq_cost(trial)
                    if c < best_cost:
                        best_cost = c
                        best_seq = trial
                        improved = True
            if not improved:
                break
        return best_cost, best_seq

    def insert_refine(seq, passes=1):
        best_cost = evaluate_seq_cost(seq)
        best_seq = list(seq)
        n = len(best_seq)
        for _ in range(passes):
            improved = False
            for i in range(n):
                item = best_seq[i]
                trial_without = best_seq[:i] + best_seq[i+1:]
                base_cost = evaluate_seq_cost(trial_without)
                best_increment = float('inf')
                best_j = -1
                for j in range(n):
                    if j == i:
                        continue
                    trial = trial_without[:j] + [item] + trial_without[j:]
                    c = evaluate_seq_cost(trial)
                    increment = c - base_cost
                    if increment < best_increment or (increment == best_increment and c < best_cost):
                        best_increment = increment
                        best_cost = c
                        best_j = j
                if best_j != -1:
                    best_seq = trial_without[:best_j] + [item] + trial_without[best_j:]
                    improved = True
            if not improved:
                break
        return best_cost, best_seq

    def block_swap_refine(seq, passes=1, block_size=3):
        best_cost = evaluate_seq_cost(seq)
        best_seq = list(seq)
        n = len(best_seq)
        for _ in range(passes):
            improved = False
            for i in range(0, n - block_size + 1):
                for j in range(i + block_size, n - block_size + 1):
                    trial = list(best_seq)
                    block_a = trial[i:i+block_size]
                    block_b = trial[j:j+block_size]
                    trial[i:i+block_size] = block_b
                    trial[j:j+block_size] = block_a
                    c = evaluate_seq_cost(trial)
                    if c < best_cost:
                        best_cost = c
                        best_seq = trial
                        improved = True
            if not improved:
                break
        return best_cost, best_seq

    best_cost = float("inf")
    best_seq = None

    full_cost, full_seq = get_full_greedy()
    if full_cost < best_cost:
        best_cost = full_cost
        best_seq = full_seq

    num_restarts = max(14, int(num_seqs) + 4)
    for r in range(num_restarts):
        seed_start = r if r < workload.num_txns else None
        cost, seq = get_greedy_cost_sampled(num_samples=24, sample_rate=0.95,
                                            seed_start=seed_start)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is not None:
        cost, seq = refine_locally(best_seq, passes=6, window=16)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq
        ins_cost, ins_seq = insert_refine(best_seq, passes=3)
        if ins_cost < best_cost:
            best_cost = ins_cost
            best_seq = ins_seq
        blk_cost, blk_seq = block_swap_refine(best_seq, passes=2, block_size=3)
        if blk_cost < best_cost:
            best_cost = blk_cost
            best_seq = blk_seq
        blk_cost2, blk_seq2 = block_swap_refine(best_seq, passes=1, block_size=2)
        if blk_cost2 < best_cost:
            best_cost = blk_cost2
            best_seq = blk_seq2
        cost, seq = refine_locally(best_seq, passes=4, window=12)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is None or len(set(best_seq)) != workload.num_txns:
        best_seq = list(range(workload.num_txns))
        best_cost = evaluate_seq_cost(best_seq)

    assert len(set(best_seq)) == workload.num_txns
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