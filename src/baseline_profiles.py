"""
Baseline profiles for instantiating known LLM optimization systems as
configurations of the OptiCom framework.

The table is intentionally explicit: each profile corresponds to the
Configuration Tuple C = (A, Q, O, E, M, S) from the framework note.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional

from optimization_controller import ActionPackage, ContextScope, EvaluationDepth
from optimization_state import OptimizationState


@dataclass(frozen=True)
class BaselineProfile:
    name: str
    display_name: str
    artifact_space: str
    query_policy: str
    operators: List[str]
    evaluation_protocol: str
    memory_policy: str
    strategy_policy: str
    branch_width: int = 3
    evaluation_depth: EvaluationDepth = EvaluationDepth.STANDARD
    context_scope: ContextScope = ContextScope.INCUMBENT_ONLY
    archive_policy: str = "accept_if_better"
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["evaluation_depth"] = self.evaluation_depth.value
        data["context_scope"] = self.context_scope.value
        return data


BASELINE_PROFILES: Dict[str, BaselineProfile] = {
    "textgrad": BaselineProfile(
        name="textgrad",
        display_name="TextGrad",
        artifact_space="any variable",
        query_policy="none",
        operators=["local_revision"],
        evaluation_protocol="textual_gradient",
        memory_policy="momentum",
        strategy_policy="none",
        branch_width=2,
        context_scope=ContextScope.ERRORS_AND_INCUMBENT,
        description="Local revision driven by textual feedback/momentum.",
    ),
    "opro": BaselineProfile(
        name="opro",
        display_name="OPRO",
        artifact_space="prompt",
        query_policy="none",
        operators=["recombination"],
        evaluation_protocol="scalar_score",
        memory_policy="in_prompt",
        strategy_policy="none",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="In-prompt optimization over scored candidates.",
    ),
    "protegi": BaselineProfile(
        name="protegi",
        display_name="ProTeGi",
        artifact_space="prompt",
        query_policy="none",
        operators=["local_revision"],
        evaluation_protocol="score_plus_textual_gradient",
        memory_policy="beam_set",
        strategy_policy="none",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Beam-style local prompt revision with textual gradients.",
    ),
    "reflexion": BaselineProfile(
        name="reflexion",
        display_name="Reflexion",
        artifact_space="action/code",
        query_policy="self_reflect",
        operators=["verification_driven_repair"],
        evaluation_protocol="binary_to_verbal",
        memory_policy="episodic",
        strategy_policy="none",
        branch_width=2,
        context_scope=ContextScope.ERRORS_AND_INCUMBENT,
        description="Self-reflection plus verification-driven repair.",
    ),
    "gepa": BaselineProfile(
        name="gepa",
        display_name="GEPA",
        artifact_space="prompt/code",
        query_policy="trace_read",
        operators=["local_revision", "recombination"],
        evaluation_protocol="multi_metric_plus_diag",
        memory_policy="implicit",
        strategy_policy="mutation_adapt",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Trace-aware local revision/recombination with mutation adaptation.",
    ),
    "adaevolve": BaselineProfile(
        name="adaevolve",
        display_name="AdaEvolve",
        artifact_space="code",
        query_policy="none",
        operators=["local_revision", "recombination"],
        evaluation_protocol="fitness",
        memory_policy="statistics",
        strategy_policy="three_level_adaptive",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Fitness-driven code evolution with adaptive operator preferences.",
    ),
    "evox": BaselineProfile(
        name="evox",
        display_name="EvoX",
        artifact_space="code",
        query_policy="none",
        operators=["meta_evolved_operator"],
        evaluation_protocol="multi_task",
        memory_policy="strategy_evolution",
        strategy_policy="co_evolution",
        branch_width=3,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Meta-evolved operator strategy over code artifacts.",
    ),
    "funsearch": BaselineProfile(
        name="funsearch",
        display_name="FunSearch",
        artifact_space="function",
        query_policy="none",
        operators=["recombination"],
        evaluation_protocol="auto_evaluator",
        memory_policy="none",
        strategy_policy="none",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Function-level recombination scored by an automatic evaluator.",
    ),
    "alphaevolve": BaselineProfile(
        name="alphaevolve",
        display_name="AlphaEvolve",
        artifact_space="codebase",
        query_policy="implicit",
        operators=["local_revision", "recombination", "diff_based_rewrite"],
        evaluation_protocol="multi_evaluator",
        memory_policy="prompt_history",
        strategy_policy="none",
        branch_width=4,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Codebase evolution with prompt history and local/recombination edits.",
    ),
    "voyager": BaselineProfile(
        name="voyager",
        display_name="Voyager",
        artifact_space="skill code",
        query_policy="env_probe",
        operators=["error_repair"],
        evaluation_protocol="self_verification",
        memory_policy="skill_library",
        strategy_policy="curriculum",
        branch_width=2,
        context_scope=ContextScope.ERRORS_AND_INCUMBENT,
        description="Environment probing and skill-library style repair.",
    ),
    "dspy": BaselineProfile(
        name="dspy",
        display_name="DSPy",
        artifact_space="prompt+demos",
        query_policy="demo_bootstrap",
        operators=["compile_prompt"],
        evaluation_protocol="metric_driven",
        memory_policy="none",
        strategy_policy="optimizer_select",
        branch_width=3,
        context_scope=ContextScope.GLOBAL_DIVERSE,
        description="Metric-driven compile/optimizer selection over prompt+demos.",
    ),
    "self_refine": BaselineProfile(
        name="self_refine",
        display_name="Self-Refine",
        artifact_space="text output",
        query_policy="self_feedback",
        operators=["self_feedback_revision"],
        evaluation_protocol="multi_aspect_verbal",
        memory_policy="none",
        strategy_policy="none",
        branch_width=2,
        context_scope=ContextScope.ERRORS_AND_INCUMBENT,
        description="Generate self-feedback, then locally revise.",
    ),
    "lats": BaselineProfile(
        name="lats",
        display_name="LATS",
        artifact_space="action trace",
        query_policy="env_feedback_reflect",
        operators=["mcts_guided_search"],
        evaluation_protocol="llm_value_plus_reflect",
        memory_policy="reflect_history",
        strategy_policy="none",
        branch_width=4,
        context_scope=ContextScope.FULL_HISTORY,
        description="Tree-search inspired action revision with reflection history.",
    ),
}

ALIASES = {
    "self-refine": "self_refine",
    "selfrefine": "self_refine",
    "alpha-evolve": "alphaevolve",
    "fun-search": "funsearch",
    "ada-evolve": "adaevolve",
    "oa": "opticom",
    "opticom": "opticom",  # canonical name: full framework, no baseline restriction
}


def normalize_baseline_name(name: Optional[str]) -> str:
    raw = (name or "opticom").strip().lower().replace(" ", "_")
    return ALIASES.get(raw, raw)


def list_baseline_names() -> List[str]:
    return list(BASELINE_PROFILES.keys())


def get_baseline_profile(name: Optional[str]) -> Optional[BaselineProfile]:
    normalized = normalize_baseline_name(name)
    if normalized in ("", "default", "opticom"):
        return None
    if normalized not in BASELINE_PROFILES:
        raise ValueError(
            f"Unknown baseline '{name}'. Available baselines: {', '.join(list_baseline_names())}"
        )
    return BASELINE_PROFILES[normalized]


def profile_catalog_json() -> str:
    return json.dumps(
        {name: profile.to_dict() for name, profile in BASELINE_PROFILES.items()},
        indent=2,
        ensure_ascii=False,
    )


def build_query_plan(profile: BaselineProfile, state: OptimizationState) -> List[str]:
    policy = profile.query_policy
    if policy == "none":
        return []
    if policy == "self_reflect":
        return [
            "Self-reflect on the latest execution trace and convert binary failures into verbal feedback.",
        ]
    if policy == "trace_read":
        return [
            "Read execution trace, diagnostics, failed cases, and artifact hotspots before mutation.",
        ]
    if policy == "implicit":
        return [
            "Use implicit benchmark and prompt-history signals to guide codebase evolution.",
        ]
    if policy == "env_probe":
        return [
            "Probe environment dependencies, APIs, runtime constraints, and evaluator contract.",
        ]
    if policy == "demo_bootstrap":
        return [
            "Bootstrap demonstrations or examples that reveal the metric-driven output contract.",
        ]
    if policy == "self_feedback":
        return [
            "Generate multi-aspect self-feedback about correctness, quality, and constraint satisfaction.",
        ]
    if policy == "env_feedback_reflect":
        return [
            "Read environment feedback and reflection history for tree-search style action selection.",
            "Probe environment dependencies and evaluator contract.",
        ]
    return [f"Apply baseline query policy: {policy}"]


def filter_memory_snapshot(profile: Optional[BaselineProfile], snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Restrict OptimizationMemory exposure to match each baseline's M dimension."""
    if profile is None:
        return snapshot

    policy = profile.memory_policy
    if policy == "none":
        return {}
    if policy == "momentum":
        return {
            "recent_revision_signals": snapshot.get("recent_revision_signals", [])[-1:],
        }
    if policy in ("in_prompt", "beam_set", "prompt_history", "implicit", "episodic", "skill_library", "reflect_history"):
        return {
            "total_events": snapshot.get("total_events", 0),
            "critical_rules": snapshot.get("critical_rules", []),
            "recent_revision_signals": snapshot.get("recent_revision_signals", []),
            "global_llm_summary": snapshot.get("global_llm_summary", "No summary generated yet."),
        }
    if policy in ("statistics", "strategy_evolution"):
        return {
            "total_events": snapshot.get("total_events", 0),
            "operator_performance": snapshot.get("operator_performance", {}),
            "critical_rules": snapshot.get("critical_rules", []),
            "recent_revision_signals": snapshot.get("recent_revision_signals", []),
            "global_llm_summary": snapshot.get("global_llm_summary", "No summary generated yet."),
        }
    return snapshot


class BaselineController:
    """Controller that freezes OptiCom into a named baseline profile."""

    def __init__(self, profile: BaselineProfile, stagnation_threshold: int = 3):
        self.profile = profile
        self.stagnation_threshold = stagnation_threshold
        self.consecutive_no_gain = 0
        self.last_best_score = -float("inf")
        self.step = 0
        self.config: Dict[str, Any] = {
            "branch_width_multiplier": 1,
            "exploration_rate": 0.0,
            "operator_weights": {},
            "disabled_operators": [],
            "preferred_operator": None,
            "baseline": profile.name,
            "maximize": True,
        }

    def _update_stagnation_tracker(self, state: OptimizationState) -> None:
        if not state.current_incumbent:
            return
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
            self.consecutive_no_gain = 0
            self.last_best_score = score

    def _operator_for_state(self, state: OptimizationState) -> str:
        operators = self.profile.operators
        if state.unresolved_errors:
            for repair_op in ("verification_driven_repair", "error_repair", "self_feedback_revision"):
                if repair_op in operators:
                    return repair_op

        preferred = self.config.get("preferred_operator")
        if preferred and preferred in operators:
            return str(preferred)

        weights = self.config.get("operator_weights", {}) or {}
        disabled = set(self.config.get("disabled_operators", []) or [])
        available = [op for op in operators if op not in disabled]
        if not available:
            available = operators

        adaptive_profiles = {"gepa", "adaevolve", "evox", "dspy"}
        if self.profile.name in adaptive_profiles and weights:
            return max(available, key=lambda op: float(weights.get(op, 1.0)))

        return available[(self.step - 1) % len(available)]

    def _branch_width(self) -> int:
        multiplier = float(self.config.get("branch_width_multiplier", 1) or 1)
        return max(1, int(round(self.profile.branch_width * multiplier)))

    def plan(self, state: OptimizationState) -> ActionPackage:
        self.step += 1
        self._update_stagnation_tracker(state)

        if (
            self.profile.strategy_policy in {"three_level_adaptive", "co_evolution", "mutation_adapt", "optimizer_select"}
            and self.consecutive_no_gain >= self.stagnation_threshold
        ):
            return ActionPackage(
                query_plan=[f"Adapt strategy for baseline {self.profile.display_name} after stagnation."],
                context_scope=ContextScope.FULL_HISTORY,
                operator_choice="TRIGGER_STRATEGY_ADAPTER",
                branch_width=0,
                evaluation_depth=EvaluationDepth.NONE,
                archive_policy="none",
            )

        return ActionPackage(
            query_plan=build_query_plan(self.profile, state),
            context_scope=self.profile.context_scope,
            operator_choice=self._operator_for_state(state),
            branch_width=self._branch_width(),
            evaluation_depth=self.profile.evaluation_depth,
            archive_policy=self.profile.archive_policy,
        )
