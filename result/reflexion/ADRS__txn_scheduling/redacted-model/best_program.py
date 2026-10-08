import random
import itertools

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3

# EVOLVE-BLOCK-START

def _build_conflict_sets(workload):
    txn_writes = []
    txn_reads = []
    for t_idx in range(workload.num_txns):
        ws = set()
        rs = set()
        for op in workload.txns[t_idx]:
            op_type = op[0]
            key = op[1]
            if op_type == 'w':
                ws.add(key)
            else:
                rs.add(key)
        txn_writes.append(ws)
        txn_reads.append(rs)
    return txn_writes, txn_reads

def _conflict_penalty(t_idx, cum_writes, cum_reads, txn_writes, txn_reads):
    w = txn_writes[t_idx]
    r = txn_reads[t_idx]
    return len(w & cum_reads) + len(r & cum_writes) + len(w & cum_writes)

def _full_greedy(workload, start_txn):
    num_txns = workload.num_txns
    txn_seq = [start_txn]
    remaining = [t for t in range(num_txns) if t != start_txn]
    current_cost = workload.get_opt_seq_cost(txn_seq)
    while remaining:
        best_delta = None
        best_cost = None
        best_t = -1
        for t in remaining:
            test_seq = txn_seq + [t]
            cost = workload.get_opt_seq_cost(test_seq)
            delta = cost - current_cost
            if (best_delta is None) or (delta < best_delta) or (delta == best_delta and cost < best_cost):
                best_delta = delta
                best_cost = cost
                best_t = t
        txn_seq.append(best_t)
        current_cost = best_cost
        remaining.remove(best_t)
    final_cost = workload.get_opt_seq_cost(txn_seq)
    return final_cost, txn_seq

def _sampled_greedy(workload, start_txn, num_samples):
    num_txns = workload.num_txns
    txn_seq = [start_txn]
    remaining = [t for t in range(num_txns) if t != start_txn]
    current_cost = workload.get_opt_seq_cost(txn_seq)
    while remaining:
        if len(remaining) <= num_samples:
            candidates = remaining[:]
        else:
            candidates = random.sample(remaining, num_samples)
        best_delta = None
        best_cost = None
        best_t = -1
        for t in candidates:
            test_seq = txn_seq + [t]
            cost = workload.get_opt_seq_cost(test_seq)
            delta = cost - current_cost
            if (best_delta is None) or (delta < best_delta) or (delta == best_delta and cost < best_cost):
                best_delta = delta
                best_cost = cost
                best_t = t
        txn_seq.append(best_t)
        current_cost = best_cost
        remaining.remove(best_t)
    final_cost = workload.get_opt_seq_cost(txn_seq)
    return final_cost, txn_seq

def _heuristic_conflict_greedy(workload, start_txn, txn_writes, txn_reads):
    num_txns = workload.num_txns
    existing_writes = set(txn_writes[start_txn])
    existing_reads = set(txn_reads[start_txn])
    txn_seq = [start_txn]
    remaining = [t for t in range(num_txns) if t != start_txn]
    while remaining:
        best_score = None
        best_t = -1
        for t in remaining:
            score = _conflict_penalty(t, existing_writes, existing_reads, txn_writes, txn_reads)
            if best_score is None or score < best_score:
                best_score = score
                best_t = t
        txn_seq.append(best_t)
        existing_writes.update(txn_writes[best_t])
        existing_reads.update(txn_reads[best_t])
        remaining.remove(best_t)
    cost = workload.get_opt_seq_cost(txn_seq)
    return cost, txn_seq

def _pairwise_conflict_heuristic_starts(workload, txn_writes, txn_reads, k):
    num_txns = workload.num_txns
    scores = []
    for t in range(num_txns):
        s = 0
        for u in range(num_txns):
            if u == t:
                continue
            s += len(txn_writes[t] & txn_reads[u])
            s += len(txn_reads[t] & txn_writes[u])
            s += len(txn_writes[t] & txn_writes[u])
        scores.append((s, t))
    scores.sort(reverse=True)
    return [t for _, t in scores[:k]]

def _insertion_search(workload, seq, current_cost, time_limit):
    num_txns = len(seq)
    import time
    improved = True
    attempts = 0
    max_attempts = 3
    while improved and attempts < max_attempts and (time.time() < time_limit):
        improved = False
        attempts += 1
        for i in range(num_txns):
            if time.time() >= time_limit:
                break
            t = seq[i]
            without = seq[:i] + seq[i+1:]
            best_cost = current_cost
            best_pos = i
            for j in range(num_txns):
                if j == i:
                    continue
                new_seq = without[:j] + [t] + without[j:]
                c = workload.get_opt_seq_cost(new_seq)
                if c < best_cost:
                    best_cost = c
                    best_pos = j
            if best_pos != i:
                new_seq = without[:best_pos] + [t] + without[best_pos:]
                seq = new_seq
                current_cost = best_cost
                improved = True
    return seq, current_cost

def _2opt_search(workload, seq, current_cost, time_limit):
    num_txns = len(seq)
    import time
    improved = True
    while improved and (time.time() < time_limit):
        improved = False
        for i in range(num_txns):
            if time.time() >= time_limit:
                break
            for j in range(i + 1, num_txns):
                if time.time() >= time_limit:
                    break
                new_seq = seq[:i] + seq[i:j+1][::-1] + seq[j+1:]
                c = workload.get_opt_seq_cost(new_seq)
                if c < current_cost - 1e-9:
                    seq = new_seq
                    current_cost = c
                    improved = True
    return seq, current_cost

def _shuffle_restart_greedy(workload, txn_writes, txn_reads, num_trials, time_limit):
    import time
    num_txns = workload.num_txns
    best_cost = None
    best_seq = None
    for _ in range(num_trials):
        if time.time() >= time_limit:
            break
        perm = list(range(num_txns))
        random.shuffle(perm)
        c = workload.get_opt_seq_cost(perm)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_seq = perm
    return best_cost, best_seq

def get_best_schedule(workload, num_seqs):
    import time
    start_time = time.time()
    time_limit = start_time + 90.0

    txn_writes, txn_reads = _build_conflict_sets(workload)
    best_cost = None
    best_seq = None

    num_txns = workload.num_txns
    if num_txns == 0:
        return 0, []
    if num_txns == 1:
        seq = [0]
        return workload.get_opt_seq_cost(seq), seq

    start_candidates = set()
    start_candidates.add(0)
    if num_txns > 1:
        start_candidates.add(num_txns - 1)

    write_counts = [(len(txn_writes[t]), t) for t in range(num_txns)]
    write_counts.sort()
    read_counts = [(len(txn_reads[t]), t) for t in range(num_txns)]
    read_counts.sort()

    for _, t in write_counts[:min(3, num_txns)]:
        start_candidates.add(t)
    for _, t in write_counts[-min(3, num_txns):]:
        start_candidates.add(t)
    for _, t in read_counts[-min(2, num_txns):]:
        start_candidates.add(t)

    heuristic_starts = _pairwise_conflict_heuristic_starts(
        workload, txn_writes, txn_reads, min(4, num_txns)
    )
    for t in heuristic_starts:
        start_candidates.add(t)

    for _ in range(min(3, num_txns)):
        start_candidates.add(random.randint(0, num_txns - 1))

    start_list = list(start_candidates)

    if num_txns <= 20:
        def greedy_fn(s):
            return _full_greedy(workload, s)
        full_greedy_budget = 0
    else:
        sample_size = max(5, min(num_txns // 5, 20))
        def greedy_fn(s):
            return _sampled_greedy(workload, s, sample_size)
        full_greedy_budget = 2
        for s in [t for _, t in write_counts[:2]]:
            if time.time() >= time_limit:
                break
            c, seq = _full_greedy(workload, s)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_seq = seq
        for s in heuristic_starts[:2]:
            if time.time() >= time_limit:
                break
            c, seq = _full_greedy(workload, s)
            if best_cost is None or c < best_cost:
                best_cost = c
                best_seq = seq

    for start in start_list:
        if time.time() >= time_limit:
            break
        c, seq = greedy_fn(start)
        if best_cost is None or c < best_cost:
            best_cost = c
            best_seq = seq

    for s in start_list[:min(3, len(start_list))]:
        if time.time() >= time_limit:
            break
        c, seq = _heuristic_conflict_greedy(workload, s, txn_writes, txn_reads)
        if c < best_cost:
            best_cost = c
            best_seq = seq

    extra_restarts = max(2, num_seqs - len(start_list) - full_greedy_budget)
    for _ in range(extra_restarts):
        if time.time() >= time_limit:
            break
        start = random.randint(0, num_txns - 1)
        if num_txns <= 20:
            c, seq = _full_greedy(workload, start)
        else:
            sample_size = max(5, min(num_txns // 5, 20))
            c, seq = _sampled_greedy(workload, start, sample_size)
        if c < best_cost:
            best_cost = c
            best_seq = seq

    if time.time() < time_limit:
        c, seq = _shuffle_restart_greedy(workload, txn_writes, txn_reads, max(2, num_seqs // 4), time_limit)
        if c is not None and c < best_cost:
            best_cost = c
            best_seq = seq

    if best_seq is None:
        best_seq = list(range(num_txns))
        best_cost = workload.get_opt_seq_cost(best_seq)

    improved = True
    current_cost = best_cost
    current_seq = best_seq[:]
    attempts = 0
    max_attempts = max(3, num_txns // 3)
    while improved and attempts < max_attempts and (time.time() < time_limit):
        improved = False
        for i in range(len(current_seq) - 1):
            if time.time() >= time_limit:
                break
            new_seq = current_seq[:i] + [current_seq[i + 1], current_seq[i]] + current_seq[i + 2:]
            c = workload.get_opt_seq_cost(new_seq)
            if c < current_cost:
                current_seq = new_seq
                current_cost = c
                improved = True
        attempts += 1

    if time.time() < time_limit:
        current_seq, current_cost = _insertion_search(workload, current_seq, current_cost, time_limit)
    if time.time() < time_limit:
        current_seq, current_cost = _2opt_search(workload, current_seq, current_cost, time_limit)

    for window in [min(4, num_txns), min(5, num_txns), min(6, num_txns)]:
        if time.time() >= time_limit:
            break
        if window < 2:
            continue
        if window >= 5 and num_txns > 40:
            step = max(1, (len(current_seq) - window + 1) // (60 // window))
            positions = range(0, len(current_seq) - window + 1, step)
        else:
            positions = range(len(current_seq) - window + 1)
        for i in positions:
            if time.time() >= time_limit:
                break
            segment = current_seq[i:i + window]
            best_seg = segment[:]
            best_seg_cost = current_cost
            for perm in itertools.permutations(segment):
                test_seq = current_seq[:i] + list(perm) + current_seq[i + window:]
                c = workload.get_opt_seq_cost(test_seq)
                if c < best_seg_cost:
                    best_seg_cost = c
                    best_seg = list(perm)
            if best_seg != segment:
                current_seq = current_seq[:i] + best_seg + current_seq[i + window:]
                current_cost = best_seg_cost

    extra_swaps = min(70, num_txns * 5)
    for _ in range(extra_swaps):
        if time.time() >= time_limit:
            break
        i = random.randint(0, len(current_seq) - 1)
        j = random.randint(0, len(current_seq) - 1)
        if i == j:
            continue
        new_seq = current_seq[:]
        new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
        c = workload.get_opt_seq_cost(new_seq)
        if c < current_cost:
            current_seq = new_seq
            current_cost = c

    if time.time() < time_limit:
        current_seq, current_cost = _insertion_search(workload, current_seq, current_cost, time_limit)

    assert len(current_seq) == num_txns, "Schedule length mismatch"
    assert len(set(current_seq)) == num_txns, "Schedule has duplicate transactions"
    assert min(current_seq) == 0 and max(current_seq) == num_txns - 1, (
        "Schedule transaction id out of range"
    )

    return current_cost, current_seq

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