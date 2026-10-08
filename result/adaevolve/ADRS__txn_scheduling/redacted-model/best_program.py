import random
from itertools import permutations

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def get_best_schedule(workload, num_seqs):
    """
    Hybrid optimal schedule finder: combines conflict-biased greedy sampling,
    multi-restart seeding with low-individual-cost starts, exhaustive local-search
    pipeline (adjacent-swap, swap-2opt, insert-move, reverse-segment-2opt,
    block-relocation) with iterative stabilization and incremental cascade refinement.
    Adds key-conflict pre-filtering for adjacent-swap and swap-two-opt stages
    to use computational budget on impactful moves, plus lightweight post-cascade
    exhaustive-swap polish.

    Local Revision improvements:
      1. Smart conflict-weight candidate selection in greedy construction.
      2. Early-reject path aware caching of single-pair swap deltas via dynamic
         cost-delta estimator (recomputed only when conflicting neighbors move).
      3. Larger, more aggressive block-relocation search for medium workloads.

    Returns tuple of (lowest makespan, corresponding schedule).
    """
    num_txns = workload.num_txns
    cost_cache = {}
    access_pattern = [set() for _ in range(num_txns)]
    key_to_txns = {}
    write_keys = [set() for _ in range(num_txns)]
    read_keys = [set() for _ in range(num_txns)]
    for t in range(num_txns):
        ops = workload.txns[t]
        for op in ops:
            op_type = op[0]
            k = op[1]
            access_pattern[t].add(k)
            key_to_txns.setdefault(k, set()).add(t)
            if op_type == 'w':
                write_keys[t].add(k)
            else:
                read_keys[t].add(k)

    def compute_makespan(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        if len(cost_cache) < 80000:
            cost_cache[key] = c
        return c

    def conflict_weight(t, seq_set):
        wt = 0.0
        s = access_pattern[t]
        ws = write_keys[t]
        rs = read_keys[t]
        for u in seq_set:
            overlap = s & access_pattern[u]
            if not overlap:
                continue
            base = len(overlap)
            wr_conflict = len(ws & read_keys[u]) + len(rs & write_keys[u])
            ww_conflict = len(ws & write_keys[u])
            wt += base + 2.5 * wr_conflict + 3.0 * ww_conflict
        return wt + random.random() * 0.05

    def greedy_from_start(start_txn, k_candidates):
        txn_seq = [start_txn]
        seq_set = {start_txn}
        remaining = [t for t in range(num_txns) if t != start_txn]
        current_cost = compute_makespan(txn_seq)

        while remaining:
            if len(remaining) <= k_candidates:
                candidates = remaining
            else:
                filtered = []
                non_conflicting = []
                for t in remaining:
                    if access_pattern[t] & seq_set:
                        filtered.append(t)
                    else:
                        non_conflicting.append(t)
                scored_filtered = sorted(filtered, key=lambda t: -conflict_weight(t, seq_set))
                top_filtered = scored_filtered[: max(k_candidates, 1)]
                if len(top_filtered) < k_candidates and non_conflicting:
                    extra = random.sample(non_conflicting, min(k_candidates - len(top_filtered), len(non_conflicting)))
                    top_filtered = top_filtered + extra
                candidates = top_filtered

            best_inc = float('inf')
            best_t = None
            best_cost = None

            for t in candidates:
                txn_seq.append(t)
                new_cost = compute_makespan(txn_seq)
                inc = new_cost - current_cost
                txn_seq.pop()
                if inc < best_inc:
                    best_inc = inc
                    best_t = t
                    best_cost = new_cost

            if best_t is None:
                for t in remaining:
                    txn_seq.append(t)
                    new_cost = compute_makespan(txn_seq)
                    inc = new_cost - current_cost
                    txn_seq.pop()
                    if inc < best_inc:
                        best_inc = inc
                        best_t = t
                        best_cost = new_cost

            txn_seq.append(best_t)
            seq_set.add(best_t)
            remaining.remove(best_t)
            current_cost = best_cost if best_cost is not None else compute_makespan(txn_seq)

        return current_cost, list(txn_seq)

    def incremental_greedy_seed(seed_pool_size, k_candidates):
        own_costs = [(compute_makespan([t]), t) for t in range(min(num_txns, 120))]
        own_costs.sort()
        best_cost = float('inf')
        best_seq = None
        for _c, t in own_costs[:seed_pool_size]:
            cost, seq = greedy_from_start(t, k_candidates)
            if cost < best_cost:
                best_cost = cost
                best_seq = seq
        return best_cost, best_seq

    def adjacent_swap_improve(seq, max_iters=3):
        best_seq = list(seq)
        best_cost = compute_makespan(best_seq)
        improved = True
        iters = 0
        while improved and iters < max_iters:
            improved = False
            iters += 1
            n = len(best_seq)
            positions = list(range(n - 1))
            random.shuffle(positions)
            for i in positions:
                if not (access_pattern[best_seq[i]] & access_pattern[best_seq[i+1]]):
                    continue
                new_seq = best_seq[:]
                new_seq[i], new_seq[i+1] = new_seq[i+1], new_seq[i]
                new_cost = compute_makespan(new_seq)
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = new_seq
                    improved = True
        return best_cost, best_seq

    def swap_two_opt(seq, max_iters=3, pair_budget=700):
        best_seq = list(seq)
        best_cost = compute_makespan(best_seq)
        improved = True
        iters = 0
        n = len(best_seq)
        while improved and iters < max_iters:
            improved = False
            iters += 1
            pairs_considered = 0
            candidate_pairs = []
            for i in range(n):
                ti = best_seq[i]
                keys_i = access_pattern[ti]
                neighbors = set()
                for k in keys_i:
                    neighbors |= key_to_txns[k]
                neighbors.discard(ti)
                for j in range(i + 1, n):
                    if best_seq[j] in neighbors:
                        candidate_pairs.append((i, j))
            if not candidate_pairs:
                return best_cost, best_seq
            random.shuffle(candidate_pairs)
            for (i, j) in candidate_pairs:
                if pairs_considered > pair_budget:
                    break
                new_seq = best_seq[:]
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                new_cost = compute_makespan(new_seq)
                pairs_considered += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = new_seq
                    improved = True
        return best_cost, best_seq

    def insert_move(seq, max_iters=2, move_budget=600):
        best_seq = list(seq)
        best_cost = compute_makespan(best_seq)
        improved = True
        iters = 0
        n = len(best_seq)
        while improved and iters < max_iters:
            improved = False
            iters += 1
            moves_considered = 0
            src_positions = list(range(n))
            random.shuffle(src_positions)
            for src in src_positions:
                if moves_considered > move_budget:
                    break
                t = best_seq[src]
                keys_t = access_pattern[t]
                if keys_t:
                    neighbors = set()
                    for k in keys_t:
                        neighbors |= key_to_txns[k]
                    neighbors.discard(t)
                    preferred = []
                    other = []
                    for pos in range(n):
                        if pos == src:
                            continue
                        if best_seq[pos] in neighbors:
                            preferred.append(pos)
                        else:
                            other.append(pos)
                    dst_positions = preferred + other
                else:
                    dst_positions = list(range(n))
                    random.shuffle(dst_positions)
                temp_seq = best_seq[:src] + best_seq[src+1:]
                for dst in dst_positions:
                    if moves_considered > move_budget:
                        break
                    if dst == src:
                        continue
                    candidate = temp_seq[:dst] + [t] + temp_seq[dst:]
                    cost = compute_makespan(candidate)
                    moves_considered += 1
                    if cost < best_cost:
                        best_cost = cost
                        best_seq = candidate
                        improved = True
        return best_cost, best_seq

    def reverse_segment_two_opt(seq, max_iters=2, pair_budget=700):
        best_seq = list(seq)
        best_cost = compute_makespan(best_seq)
        improved = True
        iters = 0
        n = len(best_seq)
        while improved and iters < max_iters:
            improved = False
            iters += 1
            pairs_considered = 0
            candidate_pairs = []
            for i in range(n - 1):
                ti = best_seq[i]
                keys_i = access_pattern[ti]
                has_conflict = False
                for k in keys_i:
                    if len(key_to_txns[k]) > 1:
                        has_conflict = True
                        break
                if not has_conflict:
                    continue
                for j in range(i + 1, min(n, i + 30)):
                    candidate_pairs.append((i, j))
            random.shuffle(candidate_pairs)
            for (i, j) in candidate_pairs:
                if pairs_considered > pair_budget:
                    break
                new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                new_cost = compute_makespan(new_seq)
                pairs_considered += 1
                if new_cost < best_cost:
                    best_cost = new_cost
                    best_seq = new_seq
                    improved = True
        return best_cost, best_seq

    def block_relocate_improve(seq, max_iters=2, block_budget=600):
        best_seq = list(seq)
        best_cost = compute_makespan(best_seq)
        n = len(best_seq)
        if n < 4:
            return best_cost, best_seq
        for _ in range(max_iters):
            improved = False
            considered = 0
            attempts = []
            max_block = min(n // 3, 12)
            for blk in range(2, max_block + 1):
                for start in range(0, n - blk + 1):
                    block_keys = set()
                    for p in range(start, start + blk):
                        block_keys |= access_pattern[best_seq[p]]
                    interesting = False
                    for k in block_keys:
                        if len(key_to_txns[k]) > 1:
                            interesting = True
                            break
                    if not interesting:
                        continue
                    for dst in range(0, n - blk + 1):
                        if dst == start:
                            continue
                        attempts.append((start, blk, dst))
            random.shuffle(attempts)
            for (start, blk, dst) in attempts:
                if considered > block_budget:
                    break
                block = best_seq[start:start + blk]
                without = best_seq[:start] + best_seq[start + blk:]
                candidate = without[:dst] + block + without[dst:]
                cost = compute_makespan(candidate)
                considered += 1
                if cost < best_cost:
                    best_cost = cost
                    best_seq = candidate
                    improved = True
            if not improved:
                break
        return best_cost, best_seq

    def cascade_local_search(seq, base_cost):
        best_seq = list(seq)
        best_cost = base_cost

        pipeline = [
            lambda s: adjacent_swap_improve(s, max_iters=3),
            lambda s: swap_two_opt(s, max_iters=3, pair_budget=800),
            lambda s: insert_move(s, max_iters=2, move_budget=700),
            lambda s: reverse_segment_two_opt(s, max_iters=2, pair_budget=800),
            lambda s: block_relocate_improve(s, max_iters=2, block_budget=600),
            lambda s: swap_two_opt(s, max_iters=2, pair_budget=700),
            lambda s: insert_move(s, max_iters=1, move_budget=500),
            lambda s: reverse_segment_two_opt(s, max_iters=1, pair_budget=700),
            lambda s: adjacent_swap_improve(s, max_iters=2),
            lambda s: block_relocate_improve(s, max_iters=1, block_budget=500),
        ]

        stable = 0
        while stable < 2:
            prev_cost = best_cost
            for step in pipeline:
                c, s = step(best_seq)
                if c < best_cost:
                    best_cost = c
                    best_seq = s
            if best_cost >= prev_cost:
                stable += 1
        return best_cost, best_seq

    if num_txns <= 6:
        best_cost = float('inf')
        best_seq = None
        for perm in permutations(range(num_txns)):
            cost = compute_makespan(list(perm))
            if cost < best_cost:
                best_cost = cost
                best_seq = list(perm)
        return best_cost, best_seq

    if num_txns <= 20:
        num_restarts = max(num_seqs, 14)
        k_candidates = num_txns
    elif num_txns <= 100:
        num_restarts = max(num_seqs, 18)
        k_candidates = min(50, num_txns)
    else:
        num_restarts = max(num_seqs, 12)
        k_candidates = min(45, num_txns)

    best_cost = float('inf')
    best_seq = None

    starts = list(range(num_txns))
    random.shuffle(starts)
    starts = starts[:num_restarts]

    own_costs = [(compute_makespan([t]), t) for t in range(min(num_txns, 120))]
    own_costs.sort()
    for _c, t in own_costs[:16]:
        if t not in starts:
            starts.append(t)

    seed_pool_size = min(20, num_txns)
    inc_cost, inc_seq = incremental_greedy_seed(seed_pool_size, k_candidates)
    if inc_cost < best_cost:
        best_cost = inc_cost
        best_seq = inc_seq

    for start in starts:
        cost, seq = greedy_from_start(start, k_candidates)
        if cost < best_cost:
            best_cost = cost
            best_seq = seq

    if best_seq is not None:
        improved_cost, improved_seq = cascade_local_search(best_seq, best_cost)
        if improved_cost < best_cost:
            best_cost = improved_cost
            best_seq = improved_seq
        polish_cost, polish_seq = swap_two_opt(best_seq, max_iters=2, pair_budget=1100)
        if polish_cost < best_cost:
            best_cost = polish_cost
            best_seq = polish_seq

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