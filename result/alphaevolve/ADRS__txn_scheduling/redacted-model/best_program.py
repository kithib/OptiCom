import random

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using hybrid multi-start greedy with incremental cost evaluation,
    deterministic seed controls, adaptive sampling for larger workloads, and refined local search.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    def build_greedy_schedule(start_txn, sample_frac=1.0):
        txn_seq = [start_txn]
        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)
        current_cost = workload.get_opt_seq_cost(txn_seq)

        while remaining_txns:
            rem_list = list(remaining_txns)
            n = len(rem_list)

            # Determine candidate set: full eval for small workloads, sampled for larger
            if n <= 12 or sample_frac >= 1.0:
                candidates = rem_list
            else:
                k = max(5, int(n * sample_frac))
                candidates = random.sample(rem_list, k)

            min_delta = float('inf')
            min_txn = -1
            min_cost = current_cost

            # Incremental cost delta to reduce overhead of repeated evaluations
            for t in candidates:
                txn_seq.append(t)
                cost = workload.get_opt_seq_cost(txn_seq)
                txn_seq.pop()
                delta = cost - current_cost
                if delta < min_delta:
                    min_delta = delta
                    min_txn = t
                    min_cost = cost

            if min_txn == -1:
                min_txn = rem_list[0]
                txn_seq.append(min_txn)
                current_cost = workload.get_opt_seq_cost(txn_seq)
            else:
                txn_seq.append(min_txn)
                current_cost = min_cost
            remaining_txns.remove(min_txn)

        assert len(set(txn_seq)) == workload.num_txns
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        return overall_cost, txn_seq

    def compute_transaction_rank(txn_id):
        standalone_cost = workload.get_opt_seq_cost([txn_id])
        ops = workload.txns[txn_id]
        op_count = len(ops)
        write_count = sum(1 for op in ops if op[2] == 'w')
        return (standalone_cost, op_count * 2 + write_count)

    def get_best_start():
        best_start = 0
        best_start_cost = float('inf')
        for t in range(workload.num_txns):
            cost = workload.get_opt_seq_cost([t])
            if cost < best_start_cost:
                best_start_cost = cost
                best_start = t
        return best_start

    def two_opt_swap_search(best_cost, best_seq):
        if len(best_seq) <= 2:
            return best_cost, best_seq
        improved = True
        while improved:
            improved = False
            n = len(best_seq)
            for i in range(n - 1):
                for j in range(i + 1, min(i + 5, n)):
                    new_seq = best_seq.copy()
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq

    def move_local_search(best_cost, best_seq):
        if len(best_seq) <= 2:
            return best_cost, best_seq
        improved = True
        while improved:
            improved = False
            n = len(best_seq)
            for i in range(n):
                elem = best_seq[i]
                for j in range(n):
                    if i == j:
                        continue
                    new_seq = best_seq[:i] + best_seq[i+1:]
                    new_seq = new_seq[:j] + [elem] + new_seq[j:]
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq

    def block_swap_search(best_cost, best_seq):
        if len(best_seq) <= 4:
            return best_cost, best_seq
        improved = True
        block_size = min(4, len(best_seq) // 4)
        while improved:
            improved = False
            n = len(best_seq)
            for i in range(0, n - block_size, block_size):
                for j in range(i + block_size, n - block_size + 1, block_size):
                    new_seq = best_seq.copy()
                    a = new_seq[i:i+block_size]
                    b = new_seq[j:j+block_size]
                    new_seq[i:i+block_size] = b
                    new_seq[j:j+block_size] = a
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq

    def block_move_search(best_cost, best_seq):
        if len(best_seq) <= 4:
            return best_cost, best_seq
        improved = True
        block_size = min(3, len(best_seq) // 5)
        while improved:
            improved = False
            n = len(best_seq)
            for i in range(n - block_size + 1):
                block = best_seq[i:i+block_size]
                rest = best_seq[:i] + best_seq[i+block_size:]
                for j in range(len(rest) + 1):
                    if j == i:
                        continue
                    new_seq = rest[:j] + block + rest[j:]
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
                        break
                if improved:
                    break
        return best_cost, best_seq

    def perturb_and_improve(best_cost, best_seq, rounds=20):
        if len(best_seq) <= 2:
            return best_cost, best_seq

        for _ in range(rounds):
            perturbed = best_seq.copy()
            move_type = random.random()
            if move_type < 0.4:
                i = random.randint(0, len(perturbed) - 2)
                perturbed[i], perturbed[i+1] = perturbed[i+1], perturbed[i]
            elif move_type < 0.7:
                i, j = random.sample(range(len(perturbed)), 2)
                perturbed[i], perturbed[j] = perturbed[j], perturbed[i]
            else:
                idx = random.randint(0, len(perturbed)-1)
                new_pos = random.randint(0, len(perturbed)-1)
                if idx != new_pos:
                    elem = perturbed.pop(idx)
                    perturbed.insert(new_pos, elem)

            if random.random() < 0.25:
                if len(perturbed) > 3:
                    start = random.randint(0, len(perturbed)-3)
                    end = start + random.randint(2, 4)
                    sub = perturbed[start:end]
                    random.shuffle(sub)
                    perturbed[start:end] = sub

            cost = workload.get_opt_seq_cost(perturbed)
            if cost < best_cost:
                best_cost = cost
                best_seq = perturbed

        return best_cost, best_seq

    best_cost = float('inf')
    best_seq = None
    total_txns = workload.num_txns
    num_restarts = max(8, num_seqs * 2)
    sample_frac = 1.0 if total_txns <= 20 else (0.8 if total_txns <= 50 else 0.75)

    candidates = [get_best_start()]
    ranked = sorted(range(total_txns), key=compute_transaction_rank)
    deterministic_seeds = ranked[:num_restarts // 2]
    candidates.extend(deterministic_seeds)
    if total_txns > 1:
        candidates.append(1 if total_txns > 1 else 0)
    while len(candidates) < num_restarts:
        candidates.append(random.randint(0, total_txns - 1))

    # Deduplicate candidates to avoid redundant runs
    seen = set()
    unique_candidates = []
    for c in candidates:
        if c not in seen:
            seen.add(c)
            unique_candidates.append(c)

    for start in unique_candidates:
        cost, seq = build_greedy_schedule(start, sample_frac)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    # Local search pipeline: layered deterministic refinement then stochastic perturbation
    best_cost, best_seq = two_opt_swap_search(best_cost, best_seq)
    best_cost, best_seq = move_local_search(best_cost, best_seq)
    best_cost, best_seq = block_swap_search(best_cost, best_seq)
    best_cost, best_seq = block_move_search(best_cost, best_seq)
    best_cost, best_seq = two_opt_swap_search(best_cost, best_seq)
    best_cost, best_seq = move_local_search(best_cost, best_seq)
    best_cost, best_seq = block_swap_search(best_cost, best_seq)
    best_cost, best_seq = perturb_and_improve(best_cost, best_seq, rounds=20)

    return best_cost, best_seq

# EVOLVE-BLOCK-END

def get_random_costs():
    workload_size = 100
    workload = Workload(WORKLOAD_1)

    makespan1, schedule1 = get_best_schedule(workload, 12)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 12)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 12)
    cost3 = workload3.get_opt_seq_cost(schedule3)
    print(cost1, cost2, cost3)
    return cost1 + cost2 + cost3, [schedule1, schedule2, schedule3]


if __name__ == "__main__":
    makespan, schedule = get_random_costs()
    print(f"Makespan: {makespan}")