import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

# ==========================================
# 1. Operator Definition
# ==========================================

@dataclass
class Operator:
    """
    Defines a single optimization operator.
    An operator dictates HOW the LLM should modify the current artifact.
    """
    name: str
    description: str
    technique_line: str
    instruction_template: str  # The actual meta-prompt instruction injected into the Context Bundle

    def to_dict(self) -> dict:
        return asdict(self)

# ==========================================
# 2. Operator Library Implementation
# ==========================================

class OperatorLibrary:
    """
    Module ⑥: Operator Library.
    Maintains the collection of available operations the controller can choose from.
    """

    def __init__(self):
        self._operators: Dict[str, Operator] = {}
        self._initialize_core_operators()

    def _initialize_core_operators(self):
        """Initializes the 6 core operators defined in the system architecture."""
        
        # 1. Local Revision (局部批评—改写)
        self.register(Operator(
            name="local_revision",
            description="Critique the current artifact locally and rewrite",
            technique_line="TextGrad / ProTeGi",
            instruction_template=(
                "Please perform a Local Revision on the CURRENT ARTIFACT. "
                "First, identify 2-3 specific local weaknesses or suboptimal patterns in the current implementation. "
                "Then, rewrite the artifact to address these specific points while preserving the overall structure and working parts."
            )
        ))

        # 2. Error Repair (基于报错与 trace 修复)
        self.register(Operator(
            name="error_repair",
            description="Repair based on error messages and execution traces",
            technique_line="N/A",
            instruction_template=(
                "Please perform an Error Repair on the CURRENT ARTIFACT. "
                "Carefully analyze the RECENT FAILURES & ERRORS provided in the context. "
                "Identify the exact lines or logic causing the error, and generate a minimal, focused fix to resolve it."
            )
        ))

        # 2b. Reliability Repair (成功率/全 case 通过优先)
        self.register(Operator(
            name="reliability_repair",
            description="Repair partial-success candidates so every evaluator case succeeds before chasing marginal score",
            technique_line="Robustness-first repair / success-rate optimization",
            instruction_template=(
                "Please perform a Reliability Repair on the CURRENT ARTIFACT.\n"
                "The evaluator metrics or recent traces indicate that the candidate has partial success "
                "(for example success_rate < 1.0, successful_runs < total_cases, timeouts, or per-case exceptions). "
                "Your first priority is to make EVERY evaluator case complete successfully; only preserve score improvements "
                "that do not reduce reliability.\n\n"
                "Required structure:\n"
                "1. Identify the exact operation that can throw, timeout, mutate stale state, or return an invalid object.\n"
                "2. Add deterministic validation and safe fallbacks around the public API output. Never return duplicates, missing items, "
                "foreign objects, infeasible placements, NaN/inf values, or partially constructed results.\n"
                "3. If the evaluator recomputes a reported metric from returned values, compute the reported metric from the same final values "
                "immediately before returning. Do not hardcode stale constants or print/return a score that can drift from the evaluator's recomputation.\n"
                "4. Prefer simple robust algorithms and bounded local search over fragile heavy optimizers when failures are caused by edge cases.\n"
                "5. If a local-search loop mutates lists/dicts, copy from the current candidate state before each move and avoid removing objects "
                "that may already have moved. Use index-based reconstruction or membership checks.\n"
                "6. Keep runtime bounded and deterministic. If the optimizer budget is nearly exhausted, return the best validated fallback.\n"
                "7. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 3. Retrieval-Guided Rewrite (结合检索证据改写)
        self.register(Operator(
            name="retrieval_guided_rewrite",
            description="Rewrite the artifact using retrieved evidence",
            technique_line="RAG-style",
            instruction_template=(
                "Please perform a Retrieval-Guided Rewrite. "
                "Strictly incorporate the knowledge and strategies found in the CRITICAL EVIDENCE block. "
                "Adapt the CURRENT ARTIFACT so that it fully aligns with the retrieved best practices or required APIs."
            )
        ))

        # 4. Recombination (组合多个 archive 优点)
        self.register(Operator(
            name="recombination",
            description="Combine strengths from multiple historical exemplars",
            technique_line="OPRO / GEPA",
            instruction_template=(
                "Please perform a Recombination operation. "
                "Analyze the CURRENT ARTIFACT alongside the provided SELECTED EXEMPLARS. "
                "Extract the most successful traits, algorithms, or prompt structures from the exemplars and cross-pollinate them into the current artifact to create a superior hybrid."
            )
        ))

        # 5. Decomposition (拆解子问题再重写)
        self.register(Operator(
            name="decomposition",
            description="Deconstruct into sub-problems before rewriting",
            technique_line="N/A",
            instruction_template=(
                "Please perform a Decomposition-based rewrite. "
                "First, break down the main OBJECTIVE into 2-4 smaller, manageable sub-tasks. "
                "Solve each sub-task logically, and then synthesize them back into a single, cohesive updated artifact."
            )
        ))

        # 6. Verification-Driven Repair (自检 -> 按失败点修复)
        self.register(Operator(
            name="verification_driven_repair",
            description="Self-reflect on failures and repair systematically",
            technique_line="Reflexion",
            instruction_template=(
                "Please perform a Verification-Driven Repair using the Reflexion framework. "
                "Step 1: Write a brief self-reflection on WHY the CURRENT ARTIFACT failed the previous evaluations. "
                "Step 2: Propose a concrete plan to avoid this failure. "
                "Step 3: Execute the plan and output the fully repaired artifact."
            )
        ))

        # 7. Constraint Relaxation (将硬约束转换为软惩罚/物理松弛)
        self.register(Operator(
            name="constraint_relaxation",
            description="Convert hard assertions into soft penalties or add a physical relaxation step",
            technique_line="Penalty Method / Relaxation",
            instruction_template=(
                "Please perform a Constraint Relaxation on the CURRENT ARTIFACT. "
                "The current implementation fails due to strict constraints (e.g., hard assertions crashing on minor overlaps or bounds). "
                "Rewrite the artifact to REMOVE these hard crashes. Instead, calculate the degree of violation (e.g., total overlap area) and apply a heavy numerical penalty to the final objective score. "
                "Optionally, add a brief iterative 'relaxation' or 'repulsion' loop to physically push elements apart before returning the layout."
            )
        ))

        # 8. Numerical Optimization (使用数值求解器突破构造法上限)
        #    关键修复: 不再硬编码 circle_packing / "Sum of Radii" 等问题特定字样，
        #    统一改为"以 OBJECTIVE 为准"的通用指导，避免误导 LLM 打印错误指标。
        self.register(Operator(
            name="numerical_optimization",
            description="Use scipy numerical solvers to find globally better solutions beyond hand-constructed layouts",
            technique_line="scipy.optimize / L-BFGS-B / SLSQP / Basin Hopping / Dual Annealing",
            instruction_template=(
                "Please perform a Numerical Optimization rewrite on the CURRENT ARTIFACT.\n"
                "Hand-constructed / heuristic layouts have clearly plateaued. You MUST now introduce a strong numerical optimizer,\n"
                "driven ENTIRELY by the OBJECTIVE stated in block 1 (do NOT assume any specific problem like circle-packing).\n\n"
                "Required structure:\n"
                "1. Parameterize the full decision variables as a flat float vector `x` (infer the correct dimensionality from the task spec and current code).\n"
                "2. Define a smooth differentiable loss: `loss(x) = -smooth_approx_of_objective(x) + penalty * sum_of_squared_constraint_violations(x)`.\n"
                "   For non-smooth min/max, use LogSumExp smoothing with a large alpha (e.g. 200~2000).\n"
                "3. Run a multi-start strategy: use BOTH\n"
                "   - `scipy.optimize.basinhopping(..., niter >= 50, T=0.005, stepsize ≈ 0.05)` with `minimizer_kwargs={'method':'L-BFGS-B', 'bounds': ...}` starting from the CURRENT best layout, AND\n"
                "   - `scipy.optimize.dual_annealing(loss, bounds, maxiter >= 2000, seed=42)` as a global-search fallback.\n"
                "   Pick whichever final x achieves a lower loss.\n"
                "4. Polish the winner with one more tight `scipy.optimize.minimize(..., method='L-BFGS-B', options={'ftol':1e-10,'gtol':1e-10,'maxiter':2000})`.\n"
                "5. Before returning, PROJECT x back to the feasible region (clip / renormalize) and recompute metrics using the *non-smoothed* true objective for reporting.\n"
                "6. Keep determinism: set numpy.random.seed / default_rng(seed=42) everywhere randomness enters.\n"
                "7. Budget: total wall-clock must stay under 120 seconds on a single CPU core.\n"
                "Output: ONE single self-contained runnable script — no markdown fences, no prose.\n"
            )
        ))

        # 9. Targeted Perturbation (在当前最优基础上做多起点数值爬山)
        #    关键修复: 旧 prompt 只要求"±5% 微调文本"，LLM 无法真正提升连续优化问题。
        #    改为"以当前解为 warm-start 做多起点数值 refinement"，更贴近连续空间爬山。
        self.register(Operator(
            name="targeted_perturbation",
            description="Multi-start local refinement seeded around the incumbent to climb the continuous score landscape",
            technique_line="Multi-start L-BFGS-B / Random Restart",
            instruction_template=(
                "Please perform a Multi-start Refinement on the CURRENT ARTIFACT.\n"
                "Keep the overall algorithmic skeleton intact. If the incumbent is already a strong normalized-score solution, "
                "treat this as a conservative polish step, not a full rewrite.\n"
                "Do all of the following explicitly:\n"
                "1. Extract the CURRENT best parameter vector from the incumbent code and use it as the primary warm-start.\n"
                "2. Preserve the incumbent as an explicit fallback and return it unchanged if all new candidates score worse, fail validation, or exceed the runtime budget.\n"
                "3. Prefer 1-5 minimal changes: scalar hyperparameters, seeds, restart counts, tolerances, local move ordering, penalty weights, or bounds. Do not replace a strong incumbent with an unrelated algorithm.\n"
                "4. Generate ~16 additional starts by adding Gaussian noise (std ≈ 0.02 in normalized coordinates) around the warm-start.\n"
                "5. Also add 4 fresh starts drawn from a problem-appropriate prior only when it is cheap and evaluator-safe.\n"
                "6. For each start, run a bounded local optimizer or local-search polish with the SAME objective/validation contract the incumbent uses.\n"
                "7. Return the single best feasible solution across all starts, measured by the TRUE objective (not a smoothed surrogate or stale reported number).\n"
                "8. Use `np.random.default_rng(seed=42)` so results are deterministic.\n"
                "9. Keep total runtime under 90 seconds.\n"
                "Output: ONE self-contained runnable Python script. No markdown, no prose.\n"
            )
        ))

        # 9b. Metric-aware Search (leaderboard/evaluator contract aware)
        self.register(Operator(
            name="metric_aware_search",
            description="Optimize against the evaluator's actual scoring contract and metric trade-offs",
            technique_line="Evaluator-aware optimization / leaderboard search",
            instruction_template=(
                "Please perform a Metric-Aware Search rewrite on the CURRENT ARTIFACT.\n"
                "Read the OBJECTIVE, the Evaluator Scoring Contract, and the Evaluator Metric Breakdown carefully. "
                "Your goal is to maximize the evaluator's PRIMARY score exactly as implemented, not an informal proxy.\n\n"
                "Required reasoning to apply internally:\n"
                "1. Identify the exact primary metric key and formula (e.g. combined_score, score, reward, target_ratio).\n"
                "2. Identify sub-metric trade-offs visible in the evaluator (e.g. objective value vs success_rate/runtime/validity).\n"
                "3. If the evaluator aggregates only successful cases, consider risk-adjusted strategies that improve the primary score, "
                "while still preserving the public API and returning valid outputs whenever the code chooses to return.\n"
                "4. Prefer cheap discrete search, greedy multi-seeding, local move/swap refinement, thresholding, or algorithm selection "
                "over heavy continuous optimizers when the artifact space is combinatorial.\n"
                "5. Do not fake metrics, edit evaluator state, use file/network/subprocess side effects, or return foreign objects. "
                "All improvements must come from the submitted artifact's public function.\n"
                "6. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9bb. Certified Numerical Search (math inequality / bound tasks)
        self.register(Operator(
            name="certified_numerical_search",
            description="Optimize mathematical bound artifacts while keeping reported metrics consistent with evaluator recomputation",
            technique_line="Certified coefficient/root search / evaluator-consistent numerical discovery",
            instruction_template=(
                "Please perform a Certified Numerical Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this for mathematical inequality, autocorrelation, uncertainty, coefficient, polynomial-root, or upper-bound tasks where "
                "the evaluator independently recomputes the reported metric.\n\n"
                "Required structure:\n"
                "1. Preserve the public API exactly. Return the same data shape and fields the evaluator expects.\n"
                "2. Identify the exact evaluator formula for the primary score and optimize that formula directly. Do not optimize a proxy if the evaluator recomputes roots, ratios, convolutions, integrals, or constants.\n"
                "3. Build a deterministic multi-start numerical search over the real decision variables (coefficients, sample values, knot values, or root parameters). Use bounded random restarts plus local polishing; prefer scipy/numpy and keep a pure-numpy fallback.\n"
                "   Do not rely on optional heavy dependencies such as JAX or Optax. If the current artifact imports them, remove those imports or guard them with a safe scipy/numpy fallback so the evaluator can run in a plain Python environment.\n"
                "4. Enforce constraints by construction where possible. If projection is needed, project before computing any reported constants.\n"
                "5. After choosing the best candidate, recompute every reported value from the final returned object using the same non-smoothed formula the evaluator will use. Never return stale constants, stale roots, NaN, inf, or values inconsistent with the final coefficients/function.\n"
                "   For Hermite/uncertainty C4 tasks, follow the evaluator's forced-zero convention exactly: the returned input coefficient vector is used to build H0,H4,...,H4(m-1), then the evaluator adds one final H4m coefficient to enforce P(0)=0. Try multiple dimensions (for example m=4,5,6,7), use the exact `(r_max**2)/(2*pi)` C4 formula, and validate the final reported C4/r_max with the same root computation before returning.\n"
                "6. For uncertainty C4 specifically, do not accidentally waste the evaluator-added final Hermite term by always pre-forcing the returned vector to make P(0)=0. Search normalized returned coefficient vectors directly, then let the evaluator-style constructor append the final forced-zero coefficient. Also run micro-perturbation/local polishing around any validated incumbent coefficients before trying broad random restarts.\n"
                "7. Keep the incumbent as a validated fallback and return it if every new candidate fails validation or exceeds the runtime budget.\n"
                "8. Use deterministic seeds and bounded loops; target under 120 seconds per candidate unless the benchmark timeout is clearly larger.\n"
                "9. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9bc. Autocorrelation Gradient Search (C1/C2/C3 inequality tasks)
        self.register(Operator(
            name="autocorrelation_gradient_search",
            description="Optimize discretized autocorrelation inequality functions with smooth gradient search and exact final validation",
            technique_line="JAX/Optax smooth max convolution search / Fourier and piecewise seeds",
            instruction_template=(
                "Please perform an Autocorrelation Gradient Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this for autocorrelation inequality tasks where the artifact returns sampled f_values and the evaluator computes "
                "`max(abs(np.convolve(f, f))) * dx / (sum(f)*dx)**2` or a close variant.\n\n"
                "Required structure:\n"
                "1. Preserve the exact public API and return tuple. For C3-style tasks, return `(f_values_np, c3_value, loss_value, n_points)`.\n"
                "2. Implement the final metric with the evaluator's exact NumPy formula using `np.convolve(..., mode='full')`, `dx = 0.5 / n_points`, and the squared integral denominator. The reported c-value and loss must be recomputed from the final returned `f_values_np`.\n"
                "3. Use the incumbent as a warm-start if it contains a good vector/function generator. Keep it as a validated fallback.\n"
                "4. For search, prefer JAX/Optax when available because these benchmarks commonly include them in requirements. Optimize a smooth surrogate: FFT or convolution autocorrelation, logsumexp/softmax approximation of max(abs(conv)), denominator safety, and optional normalization of the integral.\n"
                "5. Seed several structured families, not only random noise: low-frequency Fourier/cosine mixtures, piecewise-constant blocks with alternating signs, center-positive/side-negative shapes, sparse pulses, and small perturbations of the incumbent best.\n"
                "6. Use multi-phase refinement: broad Adam/Optax search on the smooth surrogate, lower learning-rate polish, then exact NumPy evaluation to select the best candidate. If JAX is unavailable, fall back to scipy/numpy local search.\n"
                "7. Keep JAX strictly inside differentiable surrogate code. Do final validation, reporting, random seeding, datetime/timing, and array serialization with NumPy/Python APIs; avoid JAX aliases for APIs that may be missing in older runtimes.\n"
                "   Use conservative Optax APIs: avoid version-sensitive schedule keyword arguments, and keep a constant-learning-rate Adam/AdamW fallback if a schedule constructor may not exist in the runtime.\n"
                "8. Do not skip optimization merely because an initial guess is poor. Normalize or rescale candidates instead; the ratio is scale-invariant, but the integral must not be near zero.\n"
                "9. Keep runtime bounded below the evaluator timeout and deterministic with fixed seeds.\n"
                "10. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9c. Permutation Local Search (transaction scheduling / ordering tasks)
        self.register(Operator(
            name="permutation_local_search",
            description="Optimize permutation/order artifacts with greedy seeds and discrete neighborhood search",
            technique_line="Conflict-aware greedy / 2-opt / insertion / block relocation",
            instruction_template=(
                "Please perform a Permutation Local Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this when the decision variable is an ordering/permutation/schedule rather than continuous coordinates.\n\n"
                "Required structure:\n"
                "1. Preserve the public API exactly. If the evaluator expects schedules, each schedule must be a complete permutation "
                "containing every item exactly once; never return prefixes, duplicates, or foreign objects.\n"
                "2. Treat the evaluator's exact cost function as an expensive oracle. Add a bounded dictionary cache keyed by tuple(sequence).\n"
                "3. Precompute task-specific conflict/interaction features from the input artifact, such as read/write key overlaps, write-write conflicts, "
                "transaction lengths, individual costs, or other pairwise dependency signals.\n"
                "4. Build diverse greedy seeds: lowest incremental oracle cost, conflict-weighted start choices, low individual-cost starts, "
                "and a few randomized restarts if allowed by the task.\n"
                "5. Always keep an incumbent/verified seed pool: include any complete schedules already present in the current artifact, "
                "evaluate them with the true oracle, and keep them as fallbacks before trying riskier rewrites.\n"
                "6. Add deterministic random-kick polishing around the best verified schedules: apply small seeded swaps, insertions, reversals, "
                "and short block moves, then run first-improvement local search from each kicked schedule.\n"
                "7. Refine the best seed with discrete neighborhoods, in this order: adjacent swap, arbitrary pair swap/2-opt, insertion move, "
                "segment reversal, and block relocation. Use conflict filters and budgets so runtime stays below the evaluator timeout.\n"
                "8. If the public API evaluates several independent instances or workloads, keep a separate best fallback and wall-clock/evaluation budget for each instance; do not let one hard instance consume all time or degrade easy-instance schedules.\n"
                "9. Recompute the true objective after every accepted move and return the best complete feasible schedule found.\n"
                "10. Avoid heavy continuous optimizers like scipy.minimize for permutations unless they are only a minor helper; direct discrete search should dominate.\n"
                "11. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9d. Resource Balancing Local Search (GPU/model placement, KV cache pressure)
        self.register(Operator(
            name="resource_balancing_local_search",
            description="Optimize resource-placement artifacts with robust greedy seeding and move/swap balancing",
            technique_line="Multi-key greedy / bottleneck move-swap / load balancing",
            instruction_template=(
                "Please perform a Resource Balancing Local Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this when the decision variable assigns models/jobs/items to GPUs/machines/bins/resources under capacity constraints "
                "and the evaluator measures a max pressure/load/ratio such as KVPR, memory pressure, latency, or imbalance.\n\n"
                "Required structure:\n"
                "1. Preserve the public API exactly and return the evaluator's expected object types. For placement tasks, return every input item "
                "exactly once, assigned to exactly one valid resource.\n"
                "2. Build several deterministic greedy seeds using different sort keys, such as weight, size, weight*size, weight/size, "
                "size/weight, and problem-specific pressure proxies. Try both ascending and descending orders when useful.\n"
                "3. During greedy placement, choose the target resource by the true projected post-placement bottleneck metric, with stable tie-breaks "
                "on remaining capacity and current load.\n"
                "4. Refine the best seed with bounded local search focused on the current bottleneck resource: single-item moves, pair swaps, "
                "small block moves, and 2-opt/order swaps if the algorithm is order-based.\n"
                "5. After every tentative move, validate all capacities and recompute the true evaluator metric. Accept only strict improvements "
                "or safe tie-break improvements; otherwise fully revert using fresh copied candidate state.\n"
                "6. Add a final validation/fallback path that always returns a complete feasible placement when one exists. Avoid fragile continuous "
                "optimizers as the primary method for discrete placement.\n"
                "7. If the evaluator runs many cases, prioritize all-case validity before marginal metric gains. Use per-case time limits, avoid unbounded recursion/enumerating all partitions, and return the incumbent/greedy fallback when local search budget expires.\n"
                "8. Keep runtime bounded and deterministic. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9e. Prefix Cache Ordering Search (LLM SQL / PHC)
        self.register(Operator(
            name="prefix_cache_ordering_search",
            description="Optimize row-wise column ordering for prompt-cache prefix reuse",
            technique_line="Prefix-aware greedy / grouping / vectorized PHC search",
            instruction_template=(
                "Please perform a Prefix Cache Ordering Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this when the evaluator rewards prefix reuse, prefix hit count, PHC, or row-wise column ordering.\n\n"
                "Required structure:\n"
                "1. Preserve the exact public class/function signatures and return types. For dataframe reordering tasks, do not add/remove rows or columns.\n"
                "2. Optimize the evaluator's true prefix score: consecutive fields match from the first column onward, often weighted by string length or length squared.\n"
                "3. Build fast column statistics: value frequencies, transition match counts between consecutive rows, string-length weights, uniqueness, and dependency constraints.\n"
                "4. Generate several deterministic ordering policies: global greedy by weighted match value, row-local greedy conditioned on the previous row, grouped/merged columns for stable prefixes, and dependency-respecting variants.\n"
                "5. Use vectorized pandas/numpy operations and small caches for scoring; avoid exponential recursion on many columns.\n"
                "6. Preserve existing helper methods that the evaluator or parent Algorithm class expects, such as column statistics, merge handling, and fixed reorder fallbacks. Do not delete working scaffolding just to simplify the file.\n"
                "7. Validate shape, row count, total character count, column coverage, duplicates, dependency constraints, and runtime before returning. If any validation fails, return the incumbent/fixed-reorder dataframe rather than raising.\n"
                "8. For llm_sql specifically, keep `class Evolved(Algorithm)` and `reorder(...) -> Tuple[pd.DataFrame, List[List[str]]]`; every dataset must process successfully. A lower but valid score is better than `No files processed successfully`.\n"
                "9. When warm-starting from a strong GGR/fixed-reorder incumbent, prefer small parameter/ordering-policy changes over wholesale recursive rewrites. Avoid ThreadPoolExecutor, row-wise `.itertuples()` attribute tricks, and deep recursion unless the incumbent already uses them safely.\n"
                "10. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9f. Graph Routing Local Search (CloudCast)
        self.register(Operator(
            name="graph_routing_local_search",
            description="Optimize broadcast/routing artifacts with shared-transfer and path-local-search heuristics",
            technique_line="Steiner-style broadcast tree / multi-path routing / cost-aware local search",
            instruction_template=(
                "Please perform a Graph Routing Local Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this for cloud broadcast, network transfer, routing, or graph-topology optimization tasks.\n\n"
                "Required structure:\n"
                "1. Preserve the public API exactly. For CloudCast, keep `search_algorithm(src, dsts, G, num_partitions)`, `BroadCastTopology`, `SingleDstPath`, `make_nx_graph`, `create_broadcast_topology`, and `run_search_algorithm` available with compatible behavior.\n"
                "2. Only emit `BroadCastTopology` paths as contiguous edge triples `[src, dst, G[src][dst]]`; every destination must have every partition populated. A single malformed config makes the score zero, so completeness beats risky cleverness.\n"
                "3. Start from the incumbent shortest-path fallback, then build deterministic candidate broadcast structures using shortest paths, low-cost hubs, Steiner-style shared prefixes, and limited alternative paths.\n"
                "4. Reuse common path prefixes/edges to reduce redundant transfers, and balance expensive network links when the evaluator models contention. Never share an edge by omitting it from a destination path; each destination partition must still be a full source-to-destination route.\n"
                "5. Refine the best route set with bounded edge/path swaps, hub substitutions, and destination reassignment. Use a faithful local proxy based on summed edge costs and fall back to the incumbent route if a candidate is disconnected, discontinuous, or missing partitions.\n"
                "6. Keep runtime bounded and deterministic. Never materialize all simple paths: do not call `list(nx.shortest_simple_paths(...))`. If alternate paths are needed, consume at most 2-3 paths with `itertools.islice` or skip alternatives entirely.\n"
                "7. Avoid modifying file paths, profile loading, imports, or evaluator scaffolding.\n"
                "8. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9g. Expert Load Balancing Search (EPLB)
        self.register(Operator(
            name="expert_load_balancing_search",
            description="Optimize MoE expert rearrangement/replication with bottleneck-aware local search",
            technique_line="Greedy replica allocation / bottleneck swap / low-overhead balancing",
            instruction_template=(
                "Please perform an Expert Load Balancing Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this for MoE/EPLB expert rearrangement, replica allocation, or load-balancing tasks.\n\n"
                "Required structure:\n"
                "1. Preserve `rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)` exactly and return `(phy2log, log2phy, logcnt)` tensors with the same shapes/dtypes/contracts as the incumbent.\n"
                "2. Optimize the actual combined objective: evaluator score is typically `(balancedness_score_expert + speed_score) / 2`, where `speed_score` depends on total inference/evaluation overhead. Heavy Python local search can lose even if balance improves.\n"
                "3. Keep the DeepSeek-style hierarchical algorithm as the safe fallback. Prefer light vectorized improvements to `replicate_experts` and `balanced_packing` over adding nested all-pair searches.\n"
                "4. Build cheap proxies for bottleneck expert load, replica benefit, GPU variance, and max load. Use deterministic greedy seeds only when they remain O(layers * replicas * log_experts) or close to the incumbent complexity.\n"
                "5. Any local moves/swaps must be very bounded and focused on current bottlenecks. Accept only faithful-proxy improvements, fully revert rejected moves, and skip the search path for large tensors if it may hurt speed.\n"
                "6. Validate all IDs, counts, replica sums, nonnegative `logcnt`, and `log2phy` scatter indices before returning. If validation fails, rebuild the full incumbent return tuple including `log2phy`; do not return the raw internal `(phy2log, phyrank, logcnt)` helper tuple.\n"
                "7. Avoid GPU-only dependencies, randomness, excessive `.item()` loops, printing in hot loops, or changing module-level constants/import behavior.\n"
                "8. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 9h. Frontier Algorithm Search (Frontier-CS / C++ algorithmic tasks)
        self.register(Operator(
            name="frontier_algorithm_search",
            description="Improve C++ contest-style solutions through algorithm selection and localized implementation changes",
            technique_line="Competitive programming design / complexity-aware local rewrite",
            instruction_template=(
                "Please perform a Frontier Algorithm Search rewrite on the CURRENT ARTIFACT.\n"
                "Use this for C++ stdin/stdout benchmark tasks judged by correctness and score.\n\n"
                "Required structure:\n"
                "1. Preserve a complete compilable C++ program with main(), exact input/output behavior, and no external dependencies beyond the standard library.\n"
                "2. Infer constraints from the problem statement and evaluator feedback, then choose the strongest feasible algorithmic family: greedy, DP, graph, flow/matching, binary search, local search, or hybrid heuristic.\n"
                "3. Focus changes on the hot algorithm, not formatting. Add robust edge-case handling for empty inputs, bounds, ties, overflow, and time limits.\n"
                "4. Prefer O(n log n) or better where needed; use long long for accumulated costs and stable deterministic tie-breaks.\n"
                "5. If the incumbent compiles and partially scores, preserve its working I/O scaffolding and improve only the scoring logic.\n"
                "6. Output ONLY the complete updated C++ source code; no markdown fences or explanations.\n"
            )
        ))

        # 9i. ARC Grid Program Induction
        self.register(Operator(
            name="arc_grid_program_induction",
            description="Induce ARC grid transformations with rule ensembles and train-example validation",
            technique_line="Program synthesis over grid primitives / hypothesis selection",
            instruction_template=(
                "Please perform an ARC Grid Program Induction rewrite on the CURRENT ARTIFACT.\n"
                "Use this for ARC/ARC-AGI grid transformation tasks.\n\n"
                "Required structure:\n"
                "1. Preserve the evaluator's public API exactly and return grids in the expected list/list or numpy-compatible format.\n"
                "2. Build a library of small deterministic grid primitives: crop/trim, recolor, connected components, bounding boxes, symmetry, translation, scaling, tiling, object counting, line extension, background detection, and pattern completion.\n"
                "3. For each candidate rule or rule composition, test it against all training examples before applying it to test grids. Prefer exact train consistency over plausible prose reasoning.\n"
                "4. Use fallback ensembles ranked by train consistency and simplicity; never return malformed grids, empty grids unless valid, or colors outside the ARC palette.\n"
                "5. Keep runtime bounded and deterministic; avoid external services or file/network side effects.\n"
                "6. Output ONLY the complete updated runnable artifact/source code; no markdown fences or explanations.\n"
            )
        ))

        # 10. Diff-based Rewrite 
        #     适合 stagnation 情况: incumbent 已经很大, 全量重写会丢信息;
        #     要求 LLM 输出 unified diff, synthesizer 会把 diff 应用到 incumbent 得到新 artifact。
        self.register(Operator(
            name="diff_based_rewrite",
            description="Emit a minimal unified-diff against the incumbent instead of a full rewrite, to reduce information loss and copy-paste noise",
            technique_line="AlphaEvolve-style diff generation",
            instruction_template=(
                "Please perform a Diff-based Rewrite on the CURRENT ARTIFACT.\n"
                "Do NOT output the full file. Instead, output exactly ONE unified-diff patch that, when applied\n"
                "to the CURRENT BEST ARTIFACT shown in block 3, produces a STRICTLY BETTER candidate (higher objective score).\n\n"
                "Strict rules:\n"
                "1. Output MUST start with `--- a/program.py` and `+++ b/program.py` headers.\n"
                "2. Use standard `@@ -start,len +start,len @@` hunks with 3 lines of surrounding context.\n"
                "3. Change as little as possible; target 1~3 hunks, each hunk ≤ 40 lines of change.\n"
                "4. Do NOT rename functions that are imported by the evaluator.\n"
                "5. Do NOT introduce new top-level file I/O, network calls, or subprocess launches.\n"
                "6. Preserve the incumbent algorithm as the fallback path; do not remove a known-good solution unless the replacement is validated by the same true objective.\n"
                "7. Prefer changes that tighten the numerical optimizer or local search (tolerances, restart counts, seed schedule, clipping, move ordering, penalty weight, validation formula) over structural rewrites.\n"
                "8. If the incumbent is already near a normalized score ceiling, make a surgical polish patch: 1-5 scalar or local-policy changes, with deterministic behavior and evaluator-exact reporting.\n"
                "9. Do NOT include prose, explanations, or markdown fences — the entire response MUST be the diff and nothing else.\n"
            )
        ))

        # 11. Meta-evolved Operator (EvoX-style)
        self.register(Operator(
            name="meta_evolved_operator",
            description="Rewrite the artifact by first inventing a mutation strategy, then applying it",
            technique_line="EvoX-style strategy evolution",
            instruction_template=(
                "Please perform a Meta-Evolved Rewrite. "
                "First infer the failure regime from the objective, current artifact, archive, and memory. "
                "Then choose a mutation strategy that is different from recent failed attempts, and apply it. "
                "The final output must be only the complete improved artifact, preserving the evaluator contract."
            )
        ))

        # 12. Compile Prompt (DSPy-style)
        self.register(Operator(
            name="compile_prompt",
            description="Compile prompt/demos or code scaffolding against a metric-driven evaluator",
            technique_line="DSPy compile / optimizer selection",
            instruction_template=(
                "Please perform a DSPy-style Compile step. "
                "Treat the evaluator metric as the compiler target. "
                "Extract the required input/output contract, identify useful demonstrations or invariants, "
                "and emit a metric-optimized artifact that preserves the public API exactly."
            )
        ))

        # 13. Self-Feedback Revision (Self-Refine-style)
        self.register(Operator(
            name="self_feedback_revision",
            description="Generate internal multi-aspect feedback and revise locally",
            technique_line="Self-Refine",
            instruction_template=(
                "Please perform a Self-Refine step. "
                "Internally critique the current artifact on correctness, constraint satisfaction, objective value, "
                "runtime cost, and robustness. Then apply the smallest complete revision that addresses the critique. "
                "Do not output the critique; output only the improved artifact."
            )
        ))

        # 14. MCTS-guided Search (LATS-style)
        self.register(Operator(
            name="mcts_guided_search",
            description="Generate several candidate action branches mentally and emit the best one",
            technique_line="LATS / MCTS-guided revision",
            instruction_template=(
                "Please perform an LATS-style tree-search revision. "
                "Consider multiple possible action branches using the environment feedback and reflection history. "
                "Select the branch with the best expected evaluator value and output the resulting artifact only."
            )
        ))

    def register(self, operator: Operator) -> None:
        """Registers a new operator into the library."""
        self._operators[operator.name] = operator

    def get_operator(self, name: str) -> Optional[Operator]:
        """Retrieves an operator by name."""
        return self._operators.get(name)

    def get_instruction(self, name: str) -> str:
        """Retrieves the instruction template for a given operator name. Returns a fallback if not found."""
        operator = self.get_operator(name)
        if operator:
            return operator.instruction_template
        return f"Please modify the CURRENT ARTIFACT based on the standard objective. (Operator '{name}' not found)."

    def list_available_operators(self) -> List[str]:
        """Lists the names of all registered operators."""
        return list(self._operators.keys())

# ==========================================
# 3. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    library = OperatorLibrary()

    print("Testing Operator Library")
    print(f"Available Operators: {library.list_available_operators()}\n")

    # Simulate Controller picking an operator
    chosen_op_name = "recombination"
    print(f"Controller selected: {chosen_op_name}")
    
    operator = library.get_operator(chosen_op_name)
    if operator:
        print(f"Description: {operator.description}")
        print(f"Technique Line: {operator.technique_line}")
        print("-" * 40)
        print(f"Instruction Template (To be injected by Context Assembler):\n{operator.instruction_template}")
    
    print("\n" + "="*50 + "\n")
    
    # Test Fallback
    missing_op = "quantum_optimization"
    print(f"Testing missing operator: {missing_op}")
    print(library.get_instruction(missing_op))
