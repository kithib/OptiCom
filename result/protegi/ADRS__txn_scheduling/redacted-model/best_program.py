import random
import time

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Greedy search with incremental conflict tracking, diversified restarts,
    early termination on zero-increment picks, bounded two-phase local
    refinement, cost memoization to skip redundant evaluations, improved
    candidate ordering, lightweight incremental cost filtering, a final
    aggressive block-move refinement, and adaptive eval budgets that free
    time for the most expensive refinements. Time-capped to stay well below
    benchmark timeouts.
    """
    num_restarts = max(num_seqs, 1)
    best_cost = float('inf')
    best_seq = None
    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        seq = [0]
        return workload.get_opt_seq_cost(seq), seq

    start_time = time.monotonic()
    # Per-workload budget (rough division) to never exceed global timeout.
    TIME_BUDGET = 40.0

    # --- Precompute per-transaction lightweight heuristics (O(1) lookup) ---

    def _len_or_zero(t):
        t_ops = workload.txns[t]
        if t_ops and t_ops[0]:
            try:
                return int(t_ops[0][3])
            except (TypeError, ValueError, IndexError):
                pass
        return len(t_ops) if t_ops else 0

    txn_lengths = [_len_or_zero(t) for t in range(num_txns)]

    txn_write_keys = []
    for t in range(num_txns):
        keys = set()
        for op in workload.txns[t]:
            if op and op[0] == 'w':
                try:
                    keys.add(op[2])
                except IndexError:
                    continue
        txn_write_keys.append(frozenset(keys))

    txn_read_keys = []
    for t in range(num_txns):
        keys = set()
        for op in workload.txns[t]:
            if op and op[0] == 'r':
                try:
                    keys.add(op[2])
                except IndexError:
                    continue
        txn_read_keys.append(frozenset(keys))

    txn_all_keys = [
        txn_write_keys[t] | txn_read_keys[t] for t in range(num_txns)
    ]

    txn_conflict_score = [
        len(txn_write_keys[t]) + 0.3 * len(txn_read_keys[t])
        for t in range(num_txns)
    ]

    # Small cache for repeated prefix+transaction evaluations.
    eval_cache = {}
    MAX_CACHE = 16384

    def cached_eval(prefix_tuple, t):
        key = (prefix_tuple, t)
        val = eval_cache.get(key)
        if val is not None:
            return val
        cost = workload.get_opt_seq_cost(list(prefix_tuple) + [t])
        if len(eval_cache) < MAX_CACHE:
            eval_cache[key] = cost
        return cost

    # --- Helper: one full greedy build from a given seed ---
    def run_greedy(seed_start, eval_budget):
        txn_seq = [seed_start]
        remaining = set(range(num_txns))
        remaining.remove(seed_start)
        current_cost = workload.get_opt_seq_cost(txn_seq)
        prefix_write_keys = set(txn_write_keys[seed_start])
        prefix_read_keys = set(txn_read_keys[seed_start])

        while remaining:
            if (time.monotonic() - start_time) >= TIME_BUDGET:
                for t in list(remaining):
                    txn_seq.append(t)
                return workload.get_opt_seq_cost(txn_seq), txn_seq

            min_increment = float('inf')
            min_cost_candidate = float('inf')
            min_txn = -1

            # Cheap heuristic reordering so likely-good candidates come first.
            def candidate_key(t):
                writes = txn_write_keys[t]
                reads = txn_read_keys[t]
                ww = len(writes & prefix_write_keys)
                wr = len(writes & prefix_read_keys)
                rw = len(reads & prefix_write_keys)
                conflict_overlap = ww + wr + rw
                no_conflict_flag = 0 if conflict_overlap == 0 else 1
                return (no_conflict_flag, conflict_overlap, txn_lengths[t], t)

            remaining_list = sorted(remaining, key=candidate_key)

            # Always try the first zero-conflict candidate for free: it is very
            # likely to yield increment == txn_length or less.
            window = remaining_list[:eval_budget]

            zero_found = False
            prefix_tuple = tuple(txn_seq)
            for t in window:
                if min_txn != -1 and min_increment == 0:
                    break
                cost = cached_eval(prefix_tuple, t)
                increment = cost - current_cost
                if increment < min_increment or (
                    increment == min_increment and cost < min_cost_candidate
                ):
                    min_increment = increment
                    min_cost_candidate = cost
                    min_txn = t
                    if increment == 0:
                        zero_found = True
                        break

            if min_txn == -1:
                min_txn = remaining_list[0]
                min_cost_candidate = cached_eval(tuple(txn_seq), min_txn)

            txn_seq.append(min_txn)
            current_cost = min_cost_candidate
            prefix_write_keys |= txn_write_keys[min_txn]
            prefix_read_keys |= txn_read_keys[min_txn]
            remaining.remove(min_txn)

        return current_cost, txn_seq

    # --- Diversified seeds ---
    seen_starts = set()
    candidate_starts = []

    anchor_short = min(range(num_txns), key=lambda t: txn_lengths[t])
    anchor_low_conflict = min(range(num_txns), key=lambda t: txn_conflict_score[t])
    anchor_high_conflict = max(range(num_txns), key=lambda t: txn_conflict_score[t])
    for anchor in (anchor_short, anchor_low_conflict, anchor_high_conflict):
        if anchor not in seen_starts:
            seen_starts.add(anchor)
            candidate_starts.append(anchor)

    adaptive_restarts = min(num_restarts, 25)
    for _ in range(adaptive_restarts):
        if len(seen_starts) >= num_txns:
            break
        seed = random.randrange(num_txns)
        while seed in seen_starts:
            seed = random.randrange(num_txns)
        seen_starts.add(seed)
        candidate_starts.append(seed)

    # Gradually shrink eval budget per restart so later attempts remain cheap.
    base_eval_budget = min(num_txns, 140)

    for seed in candidate_starts:
        if (time.monotonic() - start_time) >= TIME_BUDGET * 0.6:
            break
        elapsed = time.monotonic() - start_time
        # adaptive shrink: less time -> smaller budget
        budget = int(base_eval_budget * max(0.35, 1.0 - (elapsed / (TIME_BUDGET * 0.6))))
        budget = max(10, min(budget, base_eval_budget))
        try:
            cost, seq = run_greedy(seed, budget)
        except Exception:
            continue
        if len(set(seq)) != num_txns:
            continue
        if cost < best_cost:
            best_cost = cost
            best_seq = seq[:]

    if best_seq is None:
        best_seq = list(range(num_txns))
        best_cost = workload.get_opt_seq_cost(best_seq)

    # --- Phase 1: targeted adjacent-swap passes. ---
    elapsed = time.monotonic() - start_time
    if elapsed < TIME_BUDGET * 0.85 and len(best_seq) > 2:
        improved = True
        passes = 0
        max_passes = 3
        while improved and passes < max_passes and (time.monotonic() - start_time) < TIME_BUDGET:
            improved = False
            passes += 1
            for i in range(len(best_seq) - 1):
                if (time.monotonic() - start_time) >= TIME_BUDGET:
                    break
                a, b = best_seq[i], best_seq[i + 1]
                if not (txn_all_keys[a] & txn_all_keys[b]):
                    continue
                new_seq = best_seq[:]
                new_seq[i], new_seq[i + 1] = new_seq[i + 1], new_seq[i]
                new_cost = workload.get_opt_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = new_seq
                    improved = True

    # --- Phase 2: random 1-moves (escapes narrow valleys) ---
    elapsed = time.monotonic() - start_time
    if elapsed < TIME_BUDGET and len(best_seq) > 2:
        n = len(best_seq)
        remaining = TIME_BUDGET - elapsed
        budget = int(min(max(n * 6, 100), 900) * min(1.0, remaining / 10.0))
        budget = max(30, budget)
        tried = 0
        while tried < budget and (time.monotonic() - start_time) < TIME_BUDGET:
            i = random.randrange(n)
            j = random.randrange(n)
            if i == j:
                tried += 1
                continue
            moved = best_seq[i]
            new_seq = best_seq[:i] + best_seq[i + 1:]
            new_seq = new_seq[:j] + [moved] + new_seq[j:]
            moved_keys = txn_all_keys[moved]
            plausible = False
            for delta in (-2, -1, 0, 1, 2):
                pos = j + delta
                if 0 <= pos < len(new_seq) and pos != j:
                    if moved_keys & txn_all_keys[new_seq[pos]]:
                        plausible = True
                        break
            if not plausible and random.random() > 0.18:
                tried += 1
                continue
            new_cost = workload.get_opt_seq_cost(new_seq)
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = new_seq
            tried += 1

    # --- Phase 3: lightweight block-move refinement ---
    elapsed = time.monotonic() - start_time
    if elapsed < TIME_BUDGET and len(best_seq) > 6:
        n = len(best_seq)
        remaining = TIME_BUDGET - elapsed
        budget = int(min(max(n * 3, 80), 500) * min(1.0, remaining / 8.0))
        budget = max(20, budget)
        tried = 0
        while tried < budget and (time.monotonic() - start_time) < TIME_BUDGET:
            block_len = random.randint(2, max(2, min(5, n // 10)))
            if n - block_len * 2 + 1 <= 0:
                tried += 1
                continue
            i = random.randrange(n - block_len * 2 + 1)
            j = i + block_len + random.randrange(n - i - block_len * 2 + 1)
            moved_block = best_seq[i:i + block_len]
            new_seq = best_seq[:i] + best_seq[i + block_len:]
            new_seq = new_seq[:j] + moved_block + new_seq[j:]
            block_keys = set()
            for t in moved_block:
                block_keys |= txn_all_keys[t]
            plausible = False
            for pos in range(max(0, j - 2), min(n, j + block_len + 2)):
                if pos < j or pos >= j + block_len:
                    if 0 <= pos < len(new_seq):
                        if block_keys & txn_all_keys[new_seq[pos]]:
                            plausible = True
                            break
            if not plausible and random.random() > 0.1:
                tried += 1
                continue
            new_cost = workload.get_opt_seq_cost(new_seq)
            if new_cost < best_cost:
                best_cost = new_cost
                best_seq = new_seq
            tried += 1

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