import random
from itertools import combinations

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using multi-restart marginal greedy with exact incremental cost,
    strategic starting points, diverse configuration sweep, exploitation reruns, 
    and two-stage 2-opt local search refinement.

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    def compute_incremental_cost(seq_cost, prev_key_state, new_txn):
        """
        Compute exact incremental cost of appending new_txn + updated key state using workload conflict rules.
        Uses: read-write/write-write conflicts on same key add delay equal to new operation's cost.
        """
        new_seq_cost = seq_cost
        new_key_state = prev_key_state.copy()  # key -> (op_type, accumulated_cost)
        for op in workload.txns[new_txn]:
            op_type, key, op_cost, _ = op
            if key in new_key_state:
                prev_type, prev_cost = new_key_state[key]
                if (prev_type == 'w' or op_type == 'w') and prev_cost > 0:
                    new_seq_cost += op_cost
            new_key_state[key] = (op_type, new_seq_cost)
        return new_seq_cost, new_key_state

    def marginal_greedy(start_txn, num_samples, sample_rate):
        """
        Run a single greedy build from a given start transaction using incremental cost evaluation
        and candidate sampling to balance exploration/exploitation.
        """
        txn_seq = [start_txn]
        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        # Initialize cost/key state for starting transaction
        key_state = {}
        seq_cost = 0
        for op in workload.txns[start_txn]:
            op_type, key, op_cost, _ = op
            key_state[key] = (op_type, seq_cost)
            seq_cost += op_cost

        while remaining_txns:
            # Occasional random step to escape local optima
            if random.random() > sample_rate:
                t = random.choice(list(remaining_txns))
                txn_seq.append(t)
                remaining_txns.remove(t)
                seq_cost, key_state = compute_incremental_cost(seq_cost, key_state, t)
                continue

            # Candidate sampling
            candidate_pool = list(remaining_txns)
            sample_size = min(num_samples, len(candidate_pool))
            if len(candidate_pool) > sample_size:
                candidates = random.sample(candidate_pool, sample_size)
            else:
                candidates = candidate_pool

            min_increment = float('inf')
            min_txn = -1
            min_state = None
            min_new_cost = 0

            for t in candidates:
                new_cost, new_state = compute_incremental_cost(seq_cost, key_state, t)
                increment = new_cost - seq_cost
                if increment < min_increment:
                    min_increment = increment
                    min_txn = t
                    min_state = new_state
                    min_new_cost = new_cost

            if min_txn == -1:
                min_txn = random.choice(list(remaining_txns))
                seq_cost, key_state = compute_incremental_cost(seq_cost, key_state, min_txn)
            else:
                seq_cost = min_new_cost
                key_state = min_state

            remaining_txns.remove(min_txn)
            txn_seq.append(min_txn)

        final_cost = workload.get_opt_seq_cost(txn_seq)
        return final_cost, txn_seq

    def local_search_adjacent_2opt(seq, cost):
        """Fast adjacent-pair swap local search - quick improvements."""
        improved = True
        best_cost = cost
        best_seq = seq.copy()
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                new_seq = best_seq.copy()
                new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = new_seq
                    improved = True
                    break
        return best_cost, best_seq

    def local_search_full_2opt(seq, cost, max_iterations=5):
        """Slower full-pair swap local search - deeper refinement, iteration-limited."""
        best_cost = cost
        best_seq = seq.copy()
        for _ in range(max_iterations):
            improved = False
            for i in range(len(best_seq) - 1):
                for j in range(i + 1, len(best_seq)):
                    new_seq = best_seq[:]
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                    new_cost = workload.get_opt_seq_cost(new_seq)
                    if new_cost < best_cost:
                        best_cost = new_cost
                        best_seq = new_seq
                        improved = True
            if not improved:
                break
        return best_cost, best_seq

    best_cost = float('inf')
    best_seq = None

    # Deterministic exploitation start: pick transaction with fewest writes (low-conflict baseline)
    txn_write_counts = [sum(1 for op in workload.txns[t] if op[0] == 'w') for t in range(workload.num_txns)]
    low_write_start = min(range(workload.num_txns), key=lambda x: txn_write_counts[x])

    # Diverse greedy configurations: varying sample rates/sizes for exploration breadth
    configs = [
        (min(20, workload.num_txns), 1.0),
        (min(30, workload.num_txns), 0.95),
        (min(10, workload.num_txns), 0.85),
        (min(15, workload.num_txns), 0.9),
    ]

    # Multi-restart + config sweep for basin diversity
    num_trials = max(3, min(num_seqs, 5))
    for trial in range(num_trials):
        for num_samples, sample_rate in configs:
            # Alternate between random start and low-write start for exploitation + exploration
            if trial == 0:
                start_txn = low_write_start
            else:
                start_txn = random.randint(0, workload.num_txns - 1)

            trial_cost, trial_seq = marginal_greedy(start_txn, num_samples, sample_rate)

            if trial_cost < best_cost:
                best_cost = trial_cost
                best_seq = trial_seq

    # Exploitation: multiple additional high-exploration runs from low-write start with full sampling
    for _ in range(3):
        extra_cost, extra_seq = marginal_greedy(low_write_start, min(25, workload.num_txns), 1.0)
        if extra_cost < best_cost:
            best_cost = extra_cost
            best_seq = extra_seq

    # Two-stage local search refinement on best schedule
    if best_seq is not None:
        best_cost, best_seq = local_search_adjacent_2opt(best_seq, best_cost)
        best_cost, best_seq = local_search_full_2opt(best_seq, best_cost)

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