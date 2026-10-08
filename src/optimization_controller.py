import json
import math
import re
from dataclasses import dataclass, asdict
from enum import Enum
from typing import List, Optional

# Import structures from the optimization_state module we just built
# (Assuming they are in the same directory)
from optimization_state import OptimizationState, ArtifactPreview, OptimizationPhase

# ==========================================
# 1. Action Package Enums
# ==========================================

class ContextScope(str, Enum):
    """Defines what context should be assembled for the Synthesizer."""
    INCUMBENT_ONLY = "incumbent_only"
    ERRORS_AND_INCUMBENT = "errors_and_incumbent"
    FULL_HISTORY = "full_history"
    GLOBAL_DIVERSE = "global_diverse"

class EvaluationDepth(str, Enum):
    """Defines how deeply the candidates should be evaluated."""
    SHALLOW = "shallow"       # e.g., Just syntax check or simple feasibility
    STANDARD = "standard"     # e.g., Run standard unit tests
    DEEP = "deep"             # e.g., Run full benchmark suite and profiling
    NONE = "none"             # Used when delegating to the Strategy Adapter

# ==========================================
# 2. Action Package Definition
# ==========================================

@dataclass
class ActionPackage:
    """
    The output of the Optimization Controller.
    Dictates exactly what the downstream modules should do in this iteration.
    """
    query_plan: List[str]         # What information to retrieve (Query Engine)
    context_scope: ContextScope   # What context to look at (Context Assembler)
    operator_choice: str          # Which operator to use (Operator Library)
    branch_width: int             # How many candidates to generate
    evaluation_depth: EvaluationDepth # How deeply to evaluate
    archive_policy: str           # How to update the archive (e.g., 'accept_if_better')

    def to_json(self) -> str:
        data = asdict(self)
        data['context_scope'] = self.context_scope.value
        data['evaluation_depth'] = self.evaluation_depth.value
        return json.dumps(data, indent=2, ensure_ascii=False)

# ==========================================
# 3. Rule-Based Controller Implementation
# ==========================================

class RuleBasedController:
    """
    Version 1: Rule-based Optimization Controller.
    Implements heuristic rules based on system state and stagnation tracking.
    """
    
    def __init__(self, stagnation_threshold: int = 3):
        # Internal state to track stagnation
        self.stagnation_threshold = stagnation_threshold
        self.consecutive_no_gain = 0
        self.last_best_score = -float('inf')
        self.config = {
            "branch_width_multiplier": 1,
            "exploration_rate": 0.2,
            "operator_weights": {},
            "disabled_operators": [],
            "preferred_operator": None,
            "maximize": True,
            "task_description": "",
        }

    def _branch_width(self, base_width: int) -> int:
        """Apply slow-loop exploration preferences to branch width."""
        multiplier = float(self.config.get("branch_width_multiplier", 1) or 1)
        return max(1, int(round(base_width * multiplier)))

    def _choose_operator(self, candidates: List[str], default: str) -> str:
        """Pick an operator using StrategyAdapter's weights/disable hints."""
        disabled = set(self.config.get("disabled_operators", []) or [])
        weights = self.config.get("operator_weights", {}) or {}
        available = [op for op in candidates if op not in disabled]
        if not available:
            return default
        return max(available, key=lambda op: float(weights.get(op, 1.0)))

    def _preferred_operator(self) -> Optional[str]:
        op = self.config.get("preferred_operator")
        if not op:
            return None
        if op in set(self.config.get("disabled_operators", []) or []):
            return None
        return str(op)

    def _is_permutation_task(self) -> bool:
        """Heuristic detector for discrete ordering/scheduling/permutation tasks."""
        task_text = str(self.config.get("task_description") or "").lower()
        required = any(
            token in task_text
            for token in ("schedule", "scheduling", "ordering", "permutation", "sequence")
        )
        domain_hint = any(
            token in task_text
            for token in ("transaction", "txn", "workload", "conflict", "makespan")
        )
        return required and domain_hint

    def _task_signal_text(self) -> str:
        parts: List[str] = [
            str(self.config.get("task_description") or ""),
            str(self.config.get("benchmark_task_id") or ""),
            str(self.config.get("benchmark_category") or ""),
            str(self.config.get("benchmark_language") or ""),
        ]
        return "\n".join(part for part in parts if part).lower()

    def _state_signal_text(self, state: OptimizationState) -> str:
        """Collect existing controller-visible text for lightweight signal detection."""
        parts: List[str] = [
            self._task_signal_text(),
        ]
        if state.current_incumbent:
            parts.append(str(state.current_incumbent.summary_descriptor or ""))
        parts.extend(str(x) for x in (state.current_frontier_summary or [])[:3])
        parts.extend(str(x) for x in (state.unresolved_errors or [])[:5])
        parts.extend(str(x) for x in (state.recent_revision_signals or [])[:5])
        parts.extend(str(x) for x in (state.uncertainty_flags or [])[:3])
        return "\n".join(part for part in parts if part).lower()

    def _is_certified_math_task(self, state: OptimizationState) -> bool:
        """Detect inequality/bound tasks where evaluator recomputation must drive search."""
        text = self._state_signal_text(state)
        math_category = str(self.config.get("benchmark_category") or "").lower() == "math"
        bound_hint = any(
            token in text
            for token in (
                "autocorr", "uncertainty", "inequality", "upper bound",
                "lower bound", "constant c", "c3", "c4", "hermite",
                "fourier", "largest positive root", "polynomial root",
                "convolve", "recomputed", "reported", "ratio",
            )
        )
        return math_category and bound_hint

    def _is_autocorrelation_task(self, state: OptimizationState) -> bool:
        text = self._state_signal_text(state)
        math_category = str(self.config.get("benchmark_category") or "").lower() == "math"
        return math_category and any(
            token in text
            for token in (
                "autocorr", "autocorrelation", "f ★ f", "convolve",
                "max |f",
            )
        )

    def _is_prompt_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "prompt_optimization", "hotpot", "prompt engineer",
                "initial_prompt", "produce the fields", "dspy", "multi-hop qa",
                "question, passages", "natural-language instruction",
            )
        ) or str(self.config.get("benchmark_language") or "").lower() == "text"

    def _is_frontier_cs_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "frontier-cs", "frontier_cs", "competitive programmer",
                "complete c++ program", "stdin", "stdout", "algorithmic optimization",
            )
        )

    def _is_arc_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "arc_benchmark", "arc benchmark", "arc-agi", "grid transformation",
                "input grids", "output grids", "train examples", "test grid",
            )
        )

    def _is_prefix_cache_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "llm_sql", "prompt caching", "prefix reuse", "prefix hit",
                "column ordering", "reorder", "dataframe", "phc",
            )
        )

    def _is_graph_routing_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "cloudcast", "broadcast", "cloud infrastructure", "data transfer",
                "destination nodes", "network topology", "parallel paths",
            )
        )

    def _is_expert_load_task(self) -> bool:
        text = self._task_signal_text()
        return any(
            token in text
            for token in (
                "eplb", "expert parallelism", "mixture-of-expert", "mixture of expert",
                "moe", "expert rearrangement", "expert replicas", "load metrics",
            )
        )

    def _is_resource_placement_task(self, state: OptimizationState) -> bool:
        """Detect GPU/bin/resource placement problems such as prism/KVPR balancing."""
        text = self._state_signal_text(state)
        resource_hint = any(
            token in text
            for token in (
                "gpu", "gpus", "machine", "server", "resource", "bin",
                "memory", "capacity", "kvcache", "kv cache", "kvpr",
            )
        )
        placement_hint = any(
            token in text
            for token in (
                "placement", "assign", "assignment", "allocate", "allocation",
                "schedule models", "model placement", "load balance", "balancing",
            )
        )
        pressure_hint = any(
            token in text
            for token in (
                "max_kvpr", "kvpr", "pressure", "imbalance", "bottleneck",
                "remaining memory", "shared kv",
            )
        )
        return (resource_hint and placement_hint) or (resource_hint and pressure_hint)

    @staticmethod
    def _is_near_saturated_incumbent(state: OptimizationState) -> bool:
        """Detect normalized high-score incumbents where large rewrites usually regress."""
        incumbent = state.current_incumbent
        if not incumbent or not incumbent.is_feasible:
            return False
        try:
            score = float(incumbent.score)
        except (TypeError, ValueError):
            return False
        return math.isfinite(score) and 0.95 < score <= 1.0

    @staticmethod
    def _text_has_reliability_issue(text: str) -> bool:
        """Return True when a metric summary names an issue on the incumbent itself."""

        for match in re.finditer(r"success[_ ]?rate\s*[=:]\s*([0-9]*\.?[0-9]+)", text):
            try:
                value = float(match.group(1))
            except ValueError:
                continue
            if 1.0 < value <= 100.0:
                value /= 100.0
            if value < 0.999:
                return True

        for match in re.finditer(
            r"successful(?:[_ ]?runs)?\s*[=:]?\s*([0-9]+(?:\.[0-9]+)?)\s*/\s*([0-9]+(?:\.[0-9]+)?)",
            text,
        ):
            done, total = float(match.group(1)), float(match.group(2))
            if total > 0 and done < total:
                return True

        runs_match = re.search(r"successful[_ ]?runs\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text)
        total_match = re.search(r"total[_ ]?cases\s*[=:]\s*([0-9]+(?:\.[0-9]+)?)", text)
        if runs_match and total_match:
            done, total = float(runs_match.group(1)), float(total_match.group(1))
            if total > 0 and done < total:
                return True

        reliability_terms = (
            "failed cases", "per-case exception", "timeout", "timed out",
            "list.remove", "exception", "traceback", "mismatch",
            "validity=0", "marked the artifact invalid", "invalid schedule",
            "missing search_algorithm", "missing function", "no attribute",
            "no files processed", "failed to find a valid solution",
            "all test signals failed", "does not match", "output contract",
            "best_is_passed=false", "is_passed=false", "error=",
        )
        return any(term in text for term in reliability_terms)

    def _has_partial_success_signal(self, state: OptimizationState) -> bool:
        """
        Return True only when the incumbent summary itself is unreliable.

        Recent branch failures are still passed to the LLM as context, but they should
        not force every later iteration into repair once we already have a clean,
        feasible incumbent. This distinction matters for math searches where many
        exploratory candidates fail while the best artifact remains valid.
        """
        if not state.current_incumbent:
            return False
        text = str(state.current_incumbent.summary_descriptor or "").lower()
        return self._text_has_reliability_issue(text)

    def _update_stagnation_tracker(self, state: OptimizationState):
        """Updates internal counter to detect if the optimization has stagnated."""
        if state.current_incumbent:
            score = state.current_incumbent.score
            if not math.isfinite(float(self.last_best_score)):
                self.consecutive_no_gain = 0
                self.last_best_score = score
                return

            maximize = bool(self.config.get("maximize", True))
            no_gain = (
                score <= self.last_best_score
                if maximize
                else score >= self.last_best_score
            )
            if no_gain:
                self.consecutive_no_gain += 1
            else:
                # We made progress! Reset counter and update best score
                self.consecutive_no_gain = 0
                self.last_best_score = score

    def plan(self, state: OptimizationState) -> ActionPackage:
        """
        Takes the compressed OptimizationState and outputs an ActionPackage
        based on a set of predefined heuristic rules.
        """
        self._update_stagnation_tracker(state)

        resource_task = self._is_resource_placement_task(state)
        partial_success = self._has_partial_success_signal(state)
        certified_math_task = self._is_certified_math_task(state)
        autocorrelation_task = self._is_autocorrelation_task(state)
        prompt_task = self._is_prompt_task()
        frontier_task = self._is_frontier_cs_task()
        arc_task = self._is_arc_task()
        prefix_cache_task = self._is_prefix_cache_task()
        graph_routing_task = self._is_graph_routing_task()
        expert_load_task = self._is_expert_load_task()

        # ---------------------------------------------------------
        # Rule -1: A candidate can be "passed" overall while still
        # failing some evaluator cases. Fix reliability before chasing
        # marginal objective gains.
        # ---------------------------------------------------------
        if state.current_incumbent and partial_success:
            if resource_task:
                candidates = ["reliability_repair", "resource_balancing_local_search", "metric_aware_search", "verification_driven_repair"]
            elif autocorrelation_task:
                candidates = ["reliability_repair", "autocorrelation_gradient_search", "certified_numerical_search", "metric_aware_search"]
            elif certified_math_task:
                candidates = ["reliability_repair", "certified_numerical_search", "metric_aware_search", "verification_driven_repair"]
            else:
                candidates = ["reliability_repair", "verification_driven_repair", "error_repair", "metric_aware_search"]
            return ActionPackage(
                query_plan=[
                    "Search for robust full-case repair, success_rate=1.0 strategies, safe fallbacks, and bounded validation"
                ],
                context_scope=ContextScope.ERRORS_AND_INCUMBENT,
                operator_choice=self._choose_operator(candidates, default="reliability_repair"),
                branch_width=self._branch_width(3),
                evaluation_depth=EvaluationDepth.DEEP,
                archive_policy="accept_if_better",
            )

        # ---------------------------------------------------------
        # Rule 3: Continuous no gain -> Trigger Strategy Adapter.
        # With a valid incumbent, delay this until after several local-search
        # rotations; otherwise the adapter tends to broaden exploration before
        # the current solution has been exploited.
        # ---------------------------------------------------------
        adapter_threshold = self.stagnation_threshold
        if state.current_incumbent and state.current_incumbent.is_feasible:
            adapter_threshold = self.stagnation_threshold + 3
        if self.consecutive_no_gain >= adapter_threshold:
            return ActionPackage(
                query_plan=["Query Strategy Adapter for new heuristics or parameter relaxation"],
                context_scope=ContextScope.FULL_HISTORY,
                operator_choice="TRIGGER_STRATEGY_ADAPTER", # Special signal for outer loop
                branch_width=0, # Halt normal generation
                evaluation_depth=EvaluationDepth.NONE,
                archive_policy="none"
            )

        # ---------------------------------------------------------
        # 关键修复 Rule 0 (NEW - 优先级最高):
        # 已有有效 incumbent 且停滞 -> 优先切数值优化/微扰/diff 三类算子轮替
        # 这条规则必须排在 Rule 1 (error_repair) 之前, 否则永远被失败案例占用
        # ---------------------------------------------------------
        if self.consecutive_no_gain > 0 and state.current_incumbent and state.current_incumbent.is_feasible:
            preferred_operator = self._preferred_operator()
            if preferred_operator:
                return ActionPackage(
                    query_plan=["Query Strategy Adapter preferred operator and recent memory summary"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=preferred_operator,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            # 三路轮替: 全量重写 / 多起点爬山 / diff-only, 平衡探索、利用与信息保真度
            rotation = (self.consecutive_no_gain - 1) % 3
            if prompt_task:
                if rotation == 0:
                    operator_choice = self._choose_operator(
                        ["compile_prompt", "self_feedback_revision", "metric_aware_search", "recombination"],
                        default="compile_prompt",
                    )
                    branch_width = 3
                elif rotation == 1:
                    operator_choice = self._choose_operator(
                        ["self_feedback_revision", "compile_prompt", "recombination"],
                        default="self_feedback_revision",
                    )
                    branch_width = 4
                else:
                    operator_choice = self._choose_operator(
                        ["recombination", "compile_prompt", "self_feedback_revision"],
                        default="recombination",
                    )
                    branch_width = 3
                return ActionPackage(
                    query_plan=["Search for metric-driven prompt compilation, answer-format preservation, few-shot failure themes, and concise QA instructions"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(branch_width),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if frontier_task:
                operator_choice = self._choose_operator(
                    ["frontier_algorithm_search", "diff_based_rewrite", "metric_aware_search", "recombination"],
                    default="frontier_algorithm_search",
                )
                return ActionPackage(
                    query_plan=["Search for contest-grade algorithm design, complexity bottlenecks, edge-case handling, and localized C++ improvements"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3 if rotation != 1 else 4),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if arc_task:
                operator_choice = self._choose_operator(
                    ["arc_grid_program_induction", "metric_aware_search", "recombination", "diff_based_rewrite"],
                    default="arc_grid_program_induction",
                )
                return ActionPackage(
                    query_plan=["Search for ARC grid transformation primitives, hypothesis testing against train examples, and fallback rule ensembles"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3 if rotation != 1 else 4),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if prefix_cache_task:
                operator_choice = self._choose_operator(
                    ["prefix_cache_ordering_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="prefix_cache_ordering_search",
                )
                return ActionPackage(
                    query_plan=["Search for prefix-cache column ordering, row-local greedy ordering, grouping, and vectorized PHC scoring"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if graph_routing_task:
                operator_choice = self._choose_operator(
                    ["graph_routing_local_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="graph_routing_local_search",
                )
                return ActionPackage(
                    query_plan=["Search for broadcast Steiner trees, multi-path routing, shared-prefix transfer reuse, and bounded graph local search"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if expert_load_task:
                operator_choice = self._choose_operator(
                    ["expert_load_balancing_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="expert_load_balancing_search",
                )
                return ActionPackage(
                    query_plan=["Search for MoE expert replication, greedy load smoothing, bottleneck swaps, and low-overhead balancing"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if autocorrelation_task:
                if rotation == 0:
                    operator_choice = self._choose_operator(
                        ["autocorrelation_gradient_search", "certified_numerical_search", "metric_aware_search", "targeted_perturbation"],
                        default="autocorrelation_gradient_search",
                    )
                    branch_width = 3
                elif rotation == 1:
                    operator_choice = self._choose_operator(
                        ["certified_numerical_search", "autocorrelation_gradient_search", "targeted_perturbation"],
                        default="certified_numerical_search",
                    )
                    branch_width = 4
                else:
                    operator_choice = self._choose_operator(
                        ["diff_based_rewrite", "autocorrelation_gradient_search", "metric_aware_search"],
                        default="diff_based_rewrite",
                    )
                    branch_width = 3
                return ActionPackage(
                    query_plan=["Search for autocorrelation inequality gradient optimization, smooth max-abs convolution losses, Fourier/piecewise seeds, and evaluator-exact Ck recomputation"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(branch_width),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if certified_math_task:
                if rotation == 0:
                    operator_choice = self._choose_operator(
                        ["certified_numerical_search", "metric_aware_search", "targeted_perturbation", "numerical_optimization"],
                        default="certified_numerical_search",
                    )
                    branch_width = 2
                elif rotation == 1:
                    operator_choice = self._choose_operator(
                        ["targeted_perturbation", "certified_numerical_search", "metric_aware_search"],
                        default="targeted_perturbation",
                    )
                    branch_width = 2
                else:
                    operator_choice = self._choose_operator(
                        ["diff_based_rewrite", "certified_numerical_search", "metric_aware_search"],
                        default="diff_based_rewrite",
                    )
                    branch_width = 2
                return ActionPackage(
                    query_plan=["Search for evaluator-consistent inequality bounds, certified root recomputation, multi-start coefficient search, and stable fallback solutions"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(branch_width),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if resource_task:
                if rotation == 0:
                    operator_choice = self._choose_operator(
                        ["resource_balancing_local_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                        default="resource_balancing_local_search",
                    )
                    branch_width = 3
                elif rotation == 1:
                    operator_choice = self._choose_operator(
                        ["metric_aware_search", "resource_balancing_local_search", "targeted_perturbation", "recombination"],
                        default="metric_aware_search",
                    )
                    branch_width = 4
                else:
                    operator_choice = self._choose_operator(
                        ["diff_based_rewrite", "resource_balancing_local_search", "recombination"],
                        default="diff_based_rewrite",
                    )
                    branch_width = 3
                return ActionPackage(
                    query_plan=["Search for GPU/resource placement, bottleneck load balancing, multi-key greedy seeding, and safe move/swap local search"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(branch_width),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better"
                )

            if self._is_permutation_task():
                if rotation == 0:
                    operator_choice = self._choose_operator(
                        ["permutation_local_search", "metric_aware_search", "targeted_perturbation", "recombination"],
                        default="permutation_local_search",
                    )
                    branch_width = 3
                elif rotation == 1:
                    operator_choice = self._choose_operator(
                        ["targeted_perturbation", "permutation_local_search", "recombination"],
                        default="targeted_perturbation",
                    )
                    branch_width = 4
                else:
                    operator_choice = self._choose_operator(
                        ["recombination", "permutation_local_search", "diff_based_rewrite"],
                        default="recombination",
                    )
                    branch_width = 3
                return ActionPackage(
                    query_plan=["Search for permutation scheduling, conflict-aware greedy seeding, 2-opt, insertion, and block-relocation local search"],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(branch_width),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better"
                )

            if rotation == 0:
                operator_choice = self._choose_operator(
                    ["metric_aware_search", "numerical_optimization", "targeted_perturbation", "recombination"],
                    default="metric_aware_search",
                )
                branch_width = 3
            elif rotation == 1:
                operator_choice = self._choose_operator(
                    ["targeted_perturbation", "recombination", "metric_aware_search", "numerical_optimization"],
                    default="targeted_perturbation",
                )
                branch_width = 4
            else:
                # 停滞越久, 越倾向于"只改几行"式的 diff 改写, 避免 LLM 复读丢信息
                operator_choice = self._choose_operator(
                    ["diff_based_rewrite", "targeted_perturbation", "recombination"],
                    default="diff_based_rewrite",
                )
                branch_width = 3
            return ActionPackage(
                query_plan=["Search for evaluator-aware optimization, metric trade-offs, and numerical/discrete local search"],
                context_scope=ContextScope.GLOBAL_DIVERSE,
                operator_choice=operator_choice,
                branch_width=self._branch_width(branch_width),
                evaluation_depth=EvaluationDepth.DEEP,
                archive_policy="accept_if_better"
            )

        # ---------------------------------------------------------
        # Rule 1: No incumbent yet AND errors exist -> Prioritize Repair
        # ---------------------------------------------------------
        if state.unresolved_errors and not (state.current_incumbent and state.current_incumbent.is_feasible):
            error_preview = state.unresolved_errors[0][:50] if state.unresolved_errors else "unknown error"
            
            # Special case for assertion/logic errors that might need constraint relaxation
            if "AssertionError" in error_preview or "Overlap" in error_preview or "Bounds" in error_preview:
                operator_choice = "constraint_relaxation"
            else:
                operator_choice = "error_repair" # Fix bug: was "repair_mutation" before
                
            return ActionPackage(
                query_plan=[f"Search documentation/fixes for: {error_preview}"],
                context_scope=ContextScope.ERRORS_AND_INCUMBENT,
                operator_choice=operator_choice,
                branch_width=self._branch_width(2), # Keep branch width small for focused repair
                evaluation_depth=EvaluationDepth.SHALLOW, # Shallow eval is enough to check if error persists
                archive_policy="accept_if_feasible"
            )

        # ---------------------------------------------------------
        # Rule 2 (fallback - seldom used now): skip (folded into Rule 0)
        # ---------------------------------------------------------
        if False:
            pass

        # ---------------------------------------------------------
        # Default: Normal Improvement Phase
        # 关键修复: 对有 incumbent 的数值/构造类问题, 默认算子必须是能真正
        # 爬山的 numerical_optimization; 仅在完全冷启动 (没有任何可行解) 时退回
        # 到 local_revision 以免把 prompt 变成空壳数值求解脚本。
        # ---------------------------------------------------------
        has_feasible_incumbent = (
            state.current_incumbent is not None
            and state.current_incumbent.is_feasible
        )
        if has_feasible_incumbent:
            if (
                self._is_near_saturated_incumbent(state)
                and self._task_signal_text()
                and not (prompt_task or frontier_task or arc_task)
            ):
                operator_choice = self._choose_operator(
                    ["diff_based_rewrite", "targeted_perturbation", "metric_aware_search"],
                    default="diff_based_rewrite",
                )
                return ActionPackage(
                    query_plan=[
                        "Retrieve incumbent-preserving local polish tactics, minimal parameter perturbations, evaluator-exact validation, and fallback-safe diffs"
                    ],
                    context_scope=ContextScope.GLOBAL_DIVERSE,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.DEEP,
                    archive_policy="accept_if_better",
                )

            if prompt_task:
                operator_choice = self._choose_operator(
                    ["compile_prompt", "self_feedback_revision", "metric_aware_search", "recombination"],
                    default="compile_prompt",
                )
                return ActionPackage(
                    query_plan=["Retrieve prompt compilation tactics, QA error themes, and output-format constraints"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if frontier_task:
                operator_choice = self._choose_operator(
                    ["frontier_algorithm_search", "metric_aware_search", "diff_based_rewrite", "recombination"],
                    default="frontier_algorithm_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve contest algorithms, complexity constraints, and C++ edge-case repair tactics"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if arc_task:
                operator_choice = self._choose_operator(
                    ["arc_grid_program_induction", "metric_aware_search", "recombination", "diff_based_rewrite"],
                    default="arc_grid_program_induction",
                )
                return ActionPackage(
                    query_plan=["Retrieve ARC grid transformation primitives and train-example consistency checks"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if prefix_cache_task:
                operator_choice = self._choose_operator(
                    ["prefix_cache_ordering_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="prefix_cache_ordering_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve prefix-cache PHC scoring, column-order heuristics, and vectorized grouping strategies"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if graph_routing_task:
                operator_choice = self._choose_operator(
                    ["graph_routing_local_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="graph_routing_local_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve broadcast routing, graph Steiner heuristics, and transfer-cost local search"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if expert_load_task:
                operator_choice = self._choose_operator(
                    ["expert_load_balancing_search", "metric_aware_search", "recombination", "targeted_perturbation"],
                    default="expert_load_balancing_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve MoE expert load balancing, replica allocation, and bottleneck swap heuristics"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if autocorrelation_task:
                operator_choice = self._choose_operator(
                    ["autocorrelation_gradient_search", "certified_numerical_search", "metric_aware_search", "targeted_perturbation"],
                    default="autocorrelation_gradient_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve smooth autocorrelation objectives, JAX/Optax gradient search, deterministic seeds, and exact np.convolve validation"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if certified_math_task:
                operator_choice = self._choose_operator(
                    ["certified_numerical_search", "metric_aware_search", "targeted_perturbation", "numerical_optimization"],
                    default="certified_numerical_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve evaluator-consistent inequality optimization, coefficient/root search, and certified metric recomputation"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(2),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if self._is_permutation_task():
                operator_choice = self._choose_operator(
                    ["permutation_local_search", "metric_aware_search", "targeted_perturbation", "recombination"],
                    default="permutation_local_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve permutation scheduling and conflict-aware local search techniques"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            if resource_task:
                operator_choice = self._choose_operator(
                    ["resource_balancing_local_search", "metric_aware_search", "reliability_repair", "recombination"],
                    default="resource_balancing_local_search",
                )
                return ActionPackage(
                    query_plan=["Retrieve resource placement, bottleneck balancing, greedy bin-packing, and move/swap local search techniques"],
                    context_scope=ContextScope.INCUMBENT_ONLY,
                    operator_choice=operator_choice,
                    branch_width=self._branch_width(3),
                    evaluation_depth=EvaluationDepth.STANDARD,
                    archive_policy="accept_if_better",
                )

            operator_choice = self._choose_operator(
                ["metric_aware_search", "numerical_optimization", "targeted_perturbation", "recombination", "diff_based_rewrite"],
                default="metric_aware_search",
            )
            return ActionPackage(
                query_plan=["Retrieve evaluator scoring contract, metric trade-offs, and advanced optimization techniques"],
                context_scope=ContextScope.INCUMBENT_ONLY,
                operator_choice=operator_choice,
                branch_width=self._branch_width(3),
                evaluation_depth=EvaluationDepth.STANDARD,
                archive_policy="accept_if_better",
            )

        if prompt_task:
            operator_choice = self._choose_operator(
                ["compile_prompt", "self_feedback_revision", "metric_aware_search"],
                default="compile_prompt",
            )
            branch_width = 3
        elif frontier_task:
            operator_choice = self._choose_operator(
                ["frontier_algorithm_search", "metric_aware_search", "recombination"],
                default="frontier_algorithm_search",
            )
            branch_width = 3
        elif arc_task:
            operator_choice = self._choose_operator(
                ["arc_grid_program_induction", "metric_aware_search", "recombination"],
                default="arc_grid_program_induction",
            )
            branch_width = 3
        elif prefix_cache_task:
            operator_choice = self._choose_operator(
                ["prefix_cache_ordering_search", "metric_aware_search", "recombination"],
                default="prefix_cache_ordering_search",
            )
            branch_width = 2
        elif graph_routing_task:
            operator_choice = self._choose_operator(
                ["graph_routing_local_search", "metric_aware_search", "recombination"],
                default="graph_routing_local_search",
            )
            branch_width = 2
        elif expert_load_task:
            operator_choice = self._choose_operator(
                ["expert_load_balancing_search", "metric_aware_search", "recombination"],
                default="expert_load_balancing_search",
            )
            branch_width = 2
        elif autocorrelation_task:
            operator_choice = self._choose_operator(
                ["autocorrelation_gradient_search", "certified_numerical_search", "metric_aware_search"],
                default="autocorrelation_gradient_search",
            )
            branch_width = 3
        elif certified_math_task:
            operator_choice = self._choose_operator(
                ["certified_numerical_search", "metric_aware_search", "numerical_optimization"],
                default="certified_numerical_search",
            )
            branch_width = 2
        else:
            operator_choice = "local_revision"
            branch_width = 3

        return ActionPackage(
            query_plan=["Retrieve local optimization techniques"],
            context_scope=ContextScope.INCUMBENT_ONLY,
            operator_choice=operator_choice,
            branch_width=self._branch_width(branch_width),
            evaluation_depth=EvaluationDepth.STANDARD,
            archive_policy="accept_if_better"
        )

# ==========================================
# 4. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    controller = RuleBasedController(stagnation_threshold=3)

    print("Scenario 1: Unresolved errors exist (Trigger Rule 1 - Repair)")
    state_with_error = OptimizationState(
        unresolved_errors=["SyntaxError: invalid syntax at line 42"],
        current_phase=OptimizationPhase.REPAIR
    )
    action_1 = controller.plan(state_with_error)
    print(action_1.to_json())
    print("\n" + "="*50 + "\n")

    print("Scenario 2: Normal improvement step")
    # Setting an incumbent to clear the error condition and simulate a baseline
    state_normal = OptimizationState(
        current_incumbent=ArtifactPreview(artifact_id="v1", score=0.8, is_feasible=True, summary_descriptor="Baseline")
    )
    action_2 = controller.plan(state_normal)
    print(action_2.to_json())
    print("\n" + "="*50 + "\n")

    print("Scenario 3: Correctness passed but no gain (Trigger Rule 2 - Increase Branch Width)")
    # Score is still 0.8, so consecutive_no_gain will increase to 1
    state_stagnant = OptimizationState(
        current_incumbent=ArtifactPreview(artifact_id="v2", score=0.8, is_feasible=True, summary_descriptor="Minor tweak")
    )
    action_3 = controller.plan(state_stagnant)
    print(action_3.to_json())
    print("\n" + "="*50 + "\n")

    print("Scenario 4: Continuous no gain (Trigger Rule 3 - Strategy Adapter)")
    # Force consecutive_no_gain to reach the threshold
    controller.plan(state_stagnant) # reaches 2
    action_4 = controller.plan(state_stagnant) # reaches 3 (threshold)
    print(action_4.to_json())
