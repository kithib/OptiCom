import sys
from pathlib import Path


V1_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = V1_ROOT / "src"
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def test_default_query_engine_returns_useful_evidence():
    from query_engine import QueryEngine

    engine = QueryEngine()
    results = engine.run([
        "Search for numerical optimization techniques",
        "Check current environment dependencies",
        "What worked in past memory?",
    ])

    assert len(results) == 3
    assert all("No handler registered" not in result.evidence for result in results)
    assert all(result.evidence.strip() for result in results)


def test_memory_normalizes_failure_type_keys():
    from evaluation_stack import FailureType
    from optimization_memory import (
        ActionPackage,
        EvaluationRecord,
        ExecutionTrace,
        GeneratedArtifact,
        OptimizationMemory,
        QueryResult,
    )

    memory = OptimizationMemory(llm_client=None)
    memory.write(
        ActionPackage(operator_choice="error_repair", query_plan=["fix"], branch_width=1),
        [QueryResult(query_type="knowledge", evidence="repair evidence")],
        [GeneratedArtifact(artifact_id="art_1", content="bad")],
        [ExecutionTrace(is_completed=False, stdout="", stderr="SyntaxError")],
        [
            EvaluationRecord(
                artifact_id="art_1",
                failure_type=FailureType.FEASIBILITY_ERROR,
                diagnostics="SyntaxError",
                revision_signal="Fix syntax",
                is_passed=False,
                metrics={},
            )
        ],
    )

    assert len(memory.retrieve_by_failure_type("feasibility_error")) == 1


def test_strategy_adapter_preferences_are_read_by_controller():
    from llm_clients import MockLLMClient
    from operator_library import OperatorLibrary
    from optimization_controller import RuleBasedController
    from optimization_memory import OptimizationMemory, OperatorStats
    from optimization_state import ArtifactPreview, OptimizationState
    from strategy_adapter import StrategyAdapter

    controller = RuleBasedController(stagnation_threshold=5)
    library = OperatorLibrary()
    memory = OptimizationMemory(llm_client=None)
    memory.operator_stats["recombination"] = OperatorStats(attempts=2, successes=2)

    adapter = StrategyAdapter(
        llm_client=MockLLMClient(),
        controller=controller,
        operator_library=library,
    )
    adapter._adjust_operator_weights(memory)

    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=1.0,
            is_feasible=True,
            summary_descriptor="baseline",
        )
    )
    action = controller.plan(state)

    assert action.operator_choice == "recombination"


def test_controller_prefers_permutation_search_for_scheduling_tasks():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["task_description"] = (
        "Transaction scheduling workload: find a permutation/sequence ordering "
        "that minimizes makespan under read/write conflicts."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=100.0,
            is_feasible=True,
            summary_descriptor="valid schedule",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "permutation_local_search"


def test_controller_prefers_resource_balancing_for_prism_like_tasks():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["task_description"] = (
        "GPU model placement task: assign models to GPUs with 80GB memory capacity "
        "and minimize the maximum KVPR / KV cache pressure."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.95,
            is_feasible=True,
            summary_descriptor="valid placement; metrics: success_rate=1.0",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "resource_balancing_local_search"


def test_controller_polishes_high_score_prism_like_tasks_with_diff():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["task_description"] = (
        "GPU model placement task: assign models to GPUs with 80GB memory capacity "
        "and minimize the maximum KVPR / KV cache pressure."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.961,
            is_feasible=True,
            summary_descriptor="valid placement; metrics: success_rate=1.0",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "diff_based_rewrite"


def test_controller_repairs_partial_success_before_more_optimization():
    from optimization_controller import ContextScope, EvaluationDepth, RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["task_description"] = (
        "GPU model placement task: assign models to GPUs and minimize max_kvpr."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.92356,
            is_feasible=True,
            summary_descriptor=(
                "Current top performer; metrics: score=0.92356, "
                "success_rate=0.96, successful_runs=48, total_cases=50"
            ),
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "reliability_repair"
    assert action.context_scope == ContextScope.ERRORS_AND_INCUMBENT
    assert action.evaluation_depth == EvaluationDepth.DEEP


def test_controller_does_not_overrepair_branch_mismatch_with_clean_incumbent():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["benchmark_category"] = "math"
    controller.config["task_description"] = (
        "Math uncertainty inequality task: return coefficients, C4 bound, and root."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.88,
            is_feasible=True,
            summary_descriptor="valid incumbent",
        )
    )
    state.update_from_memory(
        errors=["Benchmark Evaluator Error: C4 mismatch: reported 2.49, recomputed 125.0"],
        revision_signals=["Make returned C4 consistent with recomputed value."],
        uncertainty_flags=[],
    )

    action = controller.plan(state)

    assert action.operator_choice == "certified_numerical_search"
    assert action.branch_width == 2
    assert action.branch_width == 2


def test_controller_repairs_incumbent_metric_mismatch():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["benchmark_category"] = "math"
    controller.config["task_description"] = (
        "Math uncertainty inequality task: return coefficients, C4 bound, and root."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.88,
            is_feasible=True,
            summary_descriptor="best_is_passed=false; error=C4 mismatch: reported 2.49, recomputed 125.0",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "reliability_repair"


def test_controller_delays_strategy_adapter_with_feasible_incumbent():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=3)
    controller.last_best_score = 1.0
    controller.consecutive_no_gain = 2
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=1.0,
            is_feasible=True,
            summary_descriptor="valid incumbent",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice != "TRIGGER_STRATEGY_ADAPTER"
    assert action.operator_choice == "diff_based_rewrite"


def test_controller_triggers_strategy_adapter_after_extra_feasible_rotations():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=3)
    controller.last_best_score = 1.0
    controller.consecutive_no_gain = 5
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=1.0,
            is_feasible=True,
            summary_descriptor="valid incumbent",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "TRIGGER_STRATEGY_ADAPTER"


def test_controller_prefers_autocorrelation_search_for_autocorr_tasks():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["benchmark_category"] = "math"
    controller.config["benchmark_task_id"] = "third_autocorr_ineq"
    controller.config["task_description"] = (
        "Find a better upper bound for the third autocorrelation inequality constant C3."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.61,
            is_feasible=True,
            summary_descriptor="valid incumbent; metrics: c3=2.37, combined_score=0.61",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "autocorrelation_gradient_search"


def test_controller_polishes_near_saturated_autocorr_incumbent():
    from optimization_controller import ContextScope, EvaluationDepth, RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["benchmark_category"] = "math"
    controller.config["benchmark_task_id"] = "third_autocorr_ineq"
    controller.config["task_description"] = (
        "Find a better upper bound for the third autocorrelation inequality constant C3."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.995,
            is_feasible=True,
            summary_descriptor="valid incumbent; metrics: c3=1.005, combined_score=0.995",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "diff_based_rewrite"
    assert action.context_scope == ContextScope.GLOBAL_DIVERSE
    assert action.evaluation_depth == EvaluationDepth.DEEP
    assert action.branch_width == 2


def test_controller_prefers_certified_math_search_for_uncertainty_tasks():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    controller = RuleBasedController(stagnation_threshold=5)
    controller.config["benchmark_category"] = "math"
    controller.config["benchmark_task_id"] = "uncertainty_ineq"
    controller.config["task_description"] = (
        "Find a better upper bound for the uncertainty inequality constant C4 using Hermite coefficients."
    )
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.91,
            is_feasible=True,
            summary_descriptor="valid incumbent; metrics: c4_bound=0.35, combined_score=0.91",
        )
    )

    action = controller.plan(state)

    assert action.operator_choice == "certified_numerical_search"


def test_controller_uses_domain_operators_for_targeted_benchmarks():
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    cases = [
        ("ADRS", "llm_sql", "Prompt caching column ordering with prefix reuse.", "prefix_cache_ordering_search", 2),
        ("ADRS", "cloudcast", "Cloud broadcast routing across network topology.", "graph_routing_local_search", 2),
        ("ADRS", "eplb", "MoE expert rearrangement with expert replicas and load metrics.", "expert_load_balancing_search", 2),
        ("arc_benchmark", "arc_benchmark", "ARC grid transformation from train examples.", "arc_grid_program_induction", 3),
        ("frontier-cs-eval", "frontier-cs-eval", "Complete C++ program for stdin/stdout competitive programming.", "frontier_algorithm_search", 3),
        ("prompt_optimization", "hotpot_qa", "Prompt engineer for multi-hop QA over question and passages.", "compile_prompt", 3),
    ]

    for category, task_id, description, expected, expected_branch_width in cases:
        controller = RuleBasedController(stagnation_threshold=5)
        controller.config["benchmark_category"] = category
        controller.config["benchmark_task_id"] = task_id
        controller.config["task_description"] = description
        if category == "prompt_optimization":
            controller.config["benchmark_language"] = "text"
        state = OptimizationState(
            current_incumbent=ArtifactPreview(
                artifact_id="best",
                score=0.5,
                is_feasible=True,
                summary_descriptor="valid incumbent",
            )
        )

        action = controller.plan(state)

        assert action.operator_choice == expected
        assert action.branch_width == expected_branch_width


def test_cloudcast_operator_preserves_topology_contract():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("graph_routing_local_search")
    prompt = operator.instruction_template

    assert "search_algorithm(src, dsts, G, num_partitions)" in prompt
    assert "BroadCastTopology" in prompt
    assert "make_nx_graph" in prompt
    assert "every partition populated" in prompt
    assert "single malformed config makes the score zero" in prompt
    assert "do not call `list(nx.shortest_simple_paths(...))`" in prompt


def test_eplb_operator_prioritizes_actual_speed_weighted_objective():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("expert_load_balancing_search")
    prompt = operator.instruction_template

    assert "rebalance_experts(weight, num_replicas, num_groups, num_nodes, num_gpus)" in prompt
    assert "(balancedness_score_expert + speed_score) / 2" in prompt
    assert "DeepSeek-style hierarchical algorithm as the safe fallback" in prompt
    assert "do not return the raw internal `(phy2log, phyrank, logcnt)`" in prompt


def test_permutation_operator_uses_verified_seed_pool_and_kick_polish():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("permutation_local_search")
    prompt = operator.instruction_template

    assert "incumbent/verified seed pool" in prompt
    assert "complete schedules already present in the current artifact" in prompt
    assert "deterministic random-kick polishing" in prompt
    assert "short block moves" in prompt


def test_prefix_cache_operator_prefers_small_safe_changes_for_strong_incumbents():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("prefix_cache_ordering_search")
    prompt = operator.instruction_template

    assert "strong GGR/fixed-reorder incumbent" in prompt
    assert "prefer small parameter/ordering-policy changes" in prompt
    assert "Avoid ThreadPoolExecutor" in prompt
    assert "No files processed successfully" in prompt


def test_targeted_perturbation_preserves_strong_incumbents():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("targeted_perturbation")
    prompt = operator.instruction_template

    assert "conservative polish step" in prompt
    assert "Preserve the incumbent as an explicit fallback" in prompt
    assert "Do not replace a strong incumbent with an unrelated algorithm" in prompt
    assert "TRUE objective" in prompt


def test_certified_numerical_operator_uses_uncertainty_forced_zero_convention():
    from operator_library import OperatorLibrary

    operator = OperatorLibrary().get_operator("certified_numerical_search")
    prompt = operator.instruction_template

    assert "evaluator adds one final H4m coefficient" in prompt
    assert "Search normalized returned coefficient vectors directly" in prompt
    assert "micro-perturbation/local polishing around any validated incumbent" in prompt


def test_controllers_respect_minimization_stagnation_direction():
    from baseline_profiles import BaselineController, get_baseline_profile
    from optimization_controller import RuleBasedController
    from optimization_state import ArtifactPreview, OptimizationState

    def state_with_score(score: float) -> OptimizationState:
        return OptimizationState(
            current_incumbent=ArtifactPreview(
                artifact_id=f"art_{score}",
                score=score,
                is_feasible=True,
                summary_descriptor="candidate",
            )
        )

    full_controller = RuleBasedController(stagnation_threshold=99)
    full_controller.config["maximize"] = False
    full_controller.plan(state_with_score(10.0))
    full_controller.plan(state_with_score(9.0))
    assert full_controller.consecutive_no_gain == 0
    full_controller.plan(state_with_score(9.5))
    assert full_controller.consecutive_no_gain == 1

    baseline_controller = BaselineController(
        profile=get_baseline_profile("textgrad"),
        stagnation_threshold=99,
    )
    baseline_controller.config["maximize"] = False
    baseline_controller.plan(state_with_score(10.0))
    baseline_controller.plan(state_with_score(9.0))
    assert baseline_controller.consecutive_no_gain == 0
    baseline_controller.plan(state_with_score(9.5))
    assert baseline_controller.consecutive_no_gain == 1


def test_archive_rejects_non_finite_scores():
    from archive_manager import ArchiveManager, EvaluationRecord, GeneratedArtifact

    archive = ArchiveManager(objective_metric="runtime_score")
    artifact = GeneratedArtifact("art_nan", "content", {})
    record = EvaluationRecord(
        artifact_id="art_nan",
        metrics={"runtime_score": float("nan")},
        failure_type="none",
        diagnostics="bad score",
        confidence=1.0,
        revision_signal="fix score",
        is_passed=True,
    )

    archive.update([artifact], [record])

    assert archive.best_artifact is None
    assert len(archive.rejection_archive) == 1


def test_benchmark_harness_empty_metrics_are_invalid_score():
    from benchmark_adapter import BenchmarkTaskHarness

    dummy = type(
        "DummyHarness",
        (),
        {
            "objective_metric": None,
            "METRIC_PRIORITIES": BenchmarkTaskHarness.METRIC_PRIORITIES,
        },
    )()
    assert BenchmarkTaskHarness._pick_score(dummy, {}) == -float("inf")


def test_benchmark_harness_preserves_small_construction_metrics():
    from benchmark_adapter import BenchmarkTaskHarness

    metrics = BenchmarkTaskHarness._extract_numeric_metrics(
        {
            "combined_score": 0.91,
            "coeffs": [1.0, -0.01, 2e-4],
            "large_blob": list(range(300)),
        }
    )

    assert metrics["combined_score"] == 0.91
    assert metrics["coeffs"] == [1.0, -0.01, 2e-4]
    assert "large_blob" not in metrics


def test_benchmark_task_description_includes_evaluator_scoring_excerpt(tmp_path):
    from benchmark_adapter import BenchmarkTask

    task_dir = tmp_path / "metric_task"
    task_dir.mkdir()
    initial_path = task_dir / "initial_program.py"
    evaluator_path = task_dir / "evaluator.py"
    initial_path.write_text("def solve():\n    return 1\n", encoding="utf-8")
    evaluator_path.write_text(
        "def evaluate(program_path):\n"
        "    success_rate = 0.5\n"
        "    avg_score = 2.0\n"
        "    return {'combined_score': avg_score + success_rate, 'success_rate': success_rate}\n",
        encoding="utf-8",
    )
    task = BenchmarkTask(
        task_id="metric_task",
        category="unit",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )

    description = task.get_task_description()

    assert "Evaluator Scoring Contract" in description
    assert "combined_score" in description
    assert "success_rate" in description


def test_benchmark_registry_reads_config_variants_for_text_prompt_tasks(tmp_path):
    from benchmark_adapter import BenchmarkRegistry

    task_dir = tmp_path / "prompt_optimization" / "hotpot_qa"
    task_dir.mkdir(parents=True)
    (task_dir / "initial_prompt.txt").write_text(
        "Given the fields `question`, `passages`, produce the fields `answer`.",
        encoding="utf-8",
    )
    (task_dir / "evaluator.py").write_text(
        "def evaluate(path):\n"
        "    return {'combined_score': 0.5}\n",
        encoding="utf-8",
    )
    (task_dir / "config_adaevolve.yaml").write_text(
        "prompt:\n"
        "  system_message: You are an expert prompt engineer for multi-hop QA.\n",
        encoding="utf-8",
    )

    task = BenchmarkRegistry(tmp_path).get("hotpot_qa")
    description = task.get_task_description()

    assert task.language == "text"
    assert "expert prompt engineer" in description
    assert "Return only the improved prompt text" in description
    assert "valid source code" not in description


def test_benchmark_harness_kills_isolated_evaluator_timeout(tmp_path):
    from benchmark_adapter import BenchmarkTask, BenchmarkTaskHarness
    from artifact_synthesizer import GeneratedArtifact

    task_dir = tmp_path / "slow_task"
    task_dir.mkdir()
    evaluator_path = task_dir / "evaluator.py"
    evaluator_path.write_text(
        "import time\n"
        "def evaluate(program_path):\n"
        "    time.sleep(5)\n"
        "    return {'combined_score': 1.0}\n",
        encoding="utf-8",
    )
    initial_path = task_dir / "initial_program.py"
    initial_path.write_text("# noop\n", encoding="utf-8")

    task = BenchmarkTask(
        task_id="slow_task",
        category="unit",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )
    harness = BenchmarkTaskHarness(
        task=task,
        timeout_seconds=1,
        python_executable=sys.executable,
    )

    trace = harness.run([GeneratedArtifact("slow", "# candidate\n")])[0]

    assert trace.is_completed is False
    assert trace.return_code == -1
    assert "exceeded 1s" in trace.stderr


def test_benchmark_harness_treats_evaluator_error_result_as_failure(tmp_path):
    from types import SimpleNamespace

    from artifact_synthesizer import GeneratedArtifact
    from benchmark_adapter import BenchmarkTask, BenchmarkTaskHarness
    from evaluation_stack import EvaluationStack, FailureType
    from llm_clients import MockLLMClient

    task_dir = tmp_path / "error_task"
    task_dir.mkdir()
    evaluator_path = task_dir / "evaluator.py"
    evaluator_path.write_text(
        "def evaluate(program_path):\n"
        "    return {'combined_score': 0.0, 'error': 'C1 mismatch: reported 1.0, computed 2.0'}\n",
        encoding="utf-8",
    )
    initial_path = task_dir / "initial_program.py"
    initial_path.write_text("# noop\n", encoding="utf-8")

    task = BenchmarkTask(
        task_id="error_task",
        category="unit",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )
    artifact = GeneratedArtifact("bad", "# candidate\n")
    harness = BenchmarkTaskHarness(
        task=task,
        timeout_seconds=3,
        python_executable=sys.executable,
    )

    trace = harness.run([artifact])[0]
    record = EvaluationStack(llm_client=MockLLMClient()).evaluate(
        [artifact],
        [trace],
        SimpleNamespace(objective="runtime", task_description="unit task"),
    )[0]

    assert trace.is_completed is True
    assert trace.return_code == 1
    assert trace.metrics["combined_score"] == 0.0
    assert "C1 mismatch" in trace.stderr
    assert record.failure_type == FailureType.CORRECTNESS_ERROR
    assert record.is_passed is False


def test_benchmark_harness_rejects_failed_diff_patch_before_evaluator(tmp_path):
    from artifact_synthesizer import GeneratedArtifact
    from benchmark_adapter import BenchmarkTask, BenchmarkTaskHarness

    task_dir = tmp_path / "diff_task"
    task_dir.mkdir()
    marker_path = task_dir / "evaluator_was_called"
    evaluator_path = task_dir / "evaluator.py"
    evaluator_path.write_text(
        "from pathlib import Path\n"
        f"MARKER = Path({str(marker_path)!r})\n"
        "def evaluate(program_path):\n"
        "    MARKER.write_text('called', encoding='utf-8')\n"
        "    return {'combined_score': 1.0}\n",
        encoding="utf-8",
    )
    initial_path = task_dir / "initial_program.py"
    initial_path.write_text("def run():\n    return 1\n", encoding="utf-8")
    task = BenchmarkTask(
        task_id="diff_task",
        category="unit",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )
    artifact = GeneratedArtifact(
        "bad_diff",
        "--- a/program.py\n+++ b/program.py\n@@ -1 +1 @@\n-bad\n+diff\n",
        {"diff_apply_error": "patch did not apply"},
    )

    trace = BenchmarkTaskHarness(
        task=task,
        timeout_seconds=3,
        python_executable=sys.executable,
    ).run([artifact])[0]

    assert trace.is_completed is True
    assert trace.return_code == 1
    assert trace.metrics["score"] == -float("inf")
    assert "DiffApplyError" in trace.stderr
    assert not marker_path.exists()


def test_benchmark_harness_rejects_python_syntax_before_evaluator(tmp_path):
    from artifact_synthesizer import GeneratedArtifact
    from benchmark_adapter import BenchmarkTask, BenchmarkTaskHarness

    task_dir = tmp_path / "syntax_task"
    task_dir.mkdir()
    marker_path = task_dir / "evaluator_was_called"
    evaluator_path = task_dir / "evaluator.py"
    evaluator_path.write_text(
        "from pathlib import Path\n"
        f"MARKER = Path({str(marker_path)!r})\n"
        "def evaluate(program_path):\n"
        "    MARKER.write_text('called', encoding='utf-8')\n"
        "    return {'combined_score': 1.0}\n",
        encoding="utf-8",
    )
    initial_path = task_dir / "initial_program.py"
    initial_path.write_text("def run():\n    return 1\n", encoding="utf-8")
    task = BenchmarkTask(
        task_id="syntax_task",
        category="unit",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )

    trace = BenchmarkTaskHarness(
        task=task,
        timeout_seconds=3,
        python_executable=sys.executable,
    ).run([GeneratedArtifact("bad_syntax", "def run(:\n    return 1\n")])[0]

    assert trace.is_completed is True
    assert trace.return_code == 1
    assert trace.metrics["score"] == -float("inf")
    assert "SyntaxPrecheck" in trace.stderr
    assert not marker_path.exists()


def test_artifact_synthesizer_marks_unapplied_unified_diff():
    from artifact_synthesizer import ArtifactSynthesizer

    class Client:
        def generate(self, prompt, system_instruction, n=1):
            return [
                "--- a/program.py\n+++ b/program.py\n@@ -1 +1 @@\n"
                "-def missing():\n+def changed():\n"
            ]

    operator = type("Operator", (), {
        "name": "diff_based_rewrite",
        "instruction_template": "Return a unified diff.",
    })()
    context = type("Context", (), {
        "formatted_prompt": "prompt",
        "metadata": {"incumbent_code": "def run():\n    return 1\n"},
    })()

    artifact = ArtifactSynthesizer(Client()).generate(operator, context, n=1)[0]

    assert artifact.metadata["applied_from_diff"] is False
    assert "could not be applied" in artifact.metadata["diff_apply_error"]


def test_unified_diff_applier_can_anchor_on_single_changed_line():
    from artifact_synthesizer import _apply_unified_diff

    source = "a = 1\nb = 2\nc = 3\n"
    diff = (
        "--- a/program.py\n"
        "+++ b/program.py\n"
        "@@ -1,3 +1,3 @@\n"
        " a = 1\n"
        "-b = 2\n"
        "+b = 20\n"
        " stale_context = 99\n"
    )

    patched = _apply_unified_diff(source, diff)

    assert patched == "a = 1\nb = 20\nc = 3\n"


def test_evaluation_classifies_missing_module_as_feasibility_error():
    from types import SimpleNamespace

    from artifact_synthesizer import GeneratedArtifact
    from evaluation_stack import EvaluationStack, ExecutionTrace, FailureType
    from llm_clients import MockLLMClient

    artifact = GeneratedArtifact("missing_dep", "import sympy")
    trace = ExecutionTrace(
        artifact_id="missing_dep",
        is_completed=True,
        return_code=1,
        stdout="",
        stderr="[Benchmark Evaluator Error] No module named 'sympy'",
        execution_time_ms=1.0,
        reproduction_command="benchmark evaluate",
        metrics={},
    )

    record = EvaluationStack(llm_client=MockLLMClient()).evaluate(
        [artifact],
        [trace],
        SimpleNamespace(objective="runtime", task_description="dependency test"),
    )[0]

    assert record.failure_type == FailureType.FEASIBILITY_ERROR
    assert record.is_passed is False


def test_evaluation_rejects_explicit_invalidity_even_with_finite_score():
    from types import SimpleNamespace

    from evaluation_stack import (
        EvaluationStack,
        ExecutionTrace,
        FailureType,
        GeneratedArtifact,
    )
    from llm_clients import MockLLMClient

    evaluator = EvaluationStack(llm_client=MockLLMClient())
    artifact = GeneratedArtifact("bad", "MOCK_RESPONSE")
    trace = ExecutionTrace(
        artifact_id="bad",
        is_completed=True,
        return_code=0,
        stdout="",
        stderr="",
        execution_time_ms=1.0,
        reproduction_command="mock",
        metrics={"score": 0.0, "sum_radii": 0.0, "validity": 0.0},
    )
    problem = SimpleNamespace(objective="runtime", task_description="circle packing")

    record = evaluator.evaluate([artifact], [trace], problem)[0]

    assert record.failure_type == FailureType.CORRECTNESS_ERROR
    assert not record.is_passed
    assert record.metrics["runtime_score"] == 0.0
    assert record.metrics["validity"] == 0.0


def test_evaluation_preserves_small_construction_metrics():
    from types import SimpleNamespace

    from evaluation_stack import EvaluationStack, ExecutionTrace, GeneratedArtifact
    from llm_clients import MockLLMClient

    artifact = GeneratedArtifact("coeff_candidate", "def run(): pass")
    trace = ExecutionTrace(
        artifact_id="coeff_candidate",
        is_completed=True,
        return_code=0,
        stdout="",
        stderr="",
        execution_time_ms=1.0,
        reproduction_command="benchmark evaluate",
        metrics={
            "combined_score": 0.9,
            "coeffs": [1.0, -0.01, 2e-4],
        },
    )

    record = EvaluationStack(llm_client=MockLLMClient()).evaluate(
        [artifact],
        [trace],
        SimpleNamespace(objective="runtime", task_description="coefficient task"),
    )[0]

    assert record.is_passed is True
    assert record.metrics["coeffs"] == [1.0, -0.01, 2e-4]


def test_evaluation_can_disable_cost_soft_limit():
    from types import SimpleNamespace

    from evaluation_stack import EvaluationStack, ExecutionTrace, GeneratedArtifact
    from llm_clients import MockLLMClient

    artifact = GeneratedArtifact("slow_but_valid", "def run(): pass")
    trace = ExecutionTrace(
        artifact_id="slow_but_valid",
        is_completed=True,
        return_code=0,
        stdout="",
        stderr="",
        execution_time_ms=120000.0,
        reproduction_command="benchmark evaluate",
        metrics={"combined_score": 1.0},
    )

    record = EvaluationStack(
        llm_client=MockLLMClient(),
        cost_soft_limit_ms=None,
    ).evaluate(
        [artifact],
        [trace],
        SimpleNamespace(objective="runtime", task_description="slow benchmark"),
    )[0]

    assert record.is_passed is True
    assert record.metrics["runtime_score"] == 1.0


def test_state_glue_does_not_mark_failed_scored_candidate_feasible():
    from archive_manager import ArchiveManager, EvaluationRecord, GeneratedArtifact
    from optimization_memory import OptimizationMemory
    from optimization_state import OptimizationState
    from run import update_state_glue

    archive = ArchiveManager(objective_metric="runtime_score")
    artifact = GeneratedArtifact("bad", "invalid code", {})
    record = EvaluationRecord(
        artifact_id="bad",
        metrics={"runtime_score": 0.0, "score": 0.0},
        failure_type="correctness_error",
        diagnostics="validity=0",
        confidence=0.95,
        revision_signal="Fix hard constraints.",
        is_passed=False,
    )
    archive.update([artifact], [record])

    state = OptimizationState()
    update_state_glue(
        state=state,
        archive=archive,
        memory=OptimizationMemory(llm_client=None),
        records=[record],
        obj_key="runtime_score",
    )

    assert state.current_incumbent is not None
    assert state.current_incumbent.score == 0.0
    assert state.current_incumbent.is_feasible is False


def test_run_loads_only_passed_warm_start_result(tmp_path, monkeypatch):
    import json

    import run

    monkeypatch.setattr(run, "RESULT_ROOT", tmp_path / "result")
    cfg_path = tmp_path / "warm.yaml"
    cfg_path.write_text("runner: {}\n", encoding="utf-8")
    cfg = {
        "runner": {"baseline": "opticom", "warm_start_from_result": True},
        "llm": {"provider": "openai", "model": "unit-model"},
    }
    task_slug = "inline__warm"
    out_dir = tmp_path / "result" / "opticom" / task_slug / "openai_unit-model"
    out_dir.mkdir(parents=True)
    (out_dir / "best_program.py").write_text("def run():\n    return 1\n", encoding="utf-8")
    (out_dir / "result.json").write_text(
        json.dumps(
            {
                "best_is_passed": True,
                "code_file": "best_program.py",
                "runtime_score": 0.75,
                "all_metrics": {"runtime_score": 0.75, "combined_score": 0.75},
                "best_artifact_id": "art_best",
            }
        ),
        encoding="utf-8",
    )

    warm = run._load_warm_start_result(
        cfg=cfg,
        cfg_path=cfg_path,
        benchmark_task=None,
        objective_key="runtime_score",
    )

    assert warm is not None
    assert warm["score"] == 0.75
    assert "def run" in warm["code"]

    data = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    data["best_is_passed"] = False
    (out_dir / "result.json").write_text(json.dumps(data), encoding="utf-8")

    assert run._load_warm_start_result(
        cfg=cfg,
        cfg_path=cfg_path,
        benchmark_task=None,
        objective_key="runtime_score",
    ) is None


def test_baseline_profiles_cover_reference_systems():
    from baseline_profiles import get_baseline_profile, list_baseline_names

    expected = {
        "textgrad",
        "opro",
        "protegi",
        "reflexion",
        "gepa",
        "adaevolve",
        "evox",
        "funsearch",
        "alphaevolve",
        "voyager",
        "dspy",
        "self_refine",
        "lats",
    }

    assert expected.issubset(set(list_baseline_names()))
    assert list_baseline_names()[0] == "textgrad"
    assert list_baseline_names()[-1] == "lats"
    assert get_baseline_profile("Self-Refine").name == "self_refine"
    assert get_baseline_profile("opticom") is None
    assert get_baseline_profile("gepa").operators == ["local_revision", "recombination"]


def test_baseline_controller_reflexion_uses_reflection_repair():
    from baseline_profiles import BaselineController, get_baseline_profile
    from optimization_state import ArtifactPreview, OptimizationState

    profile = get_baseline_profile("reflexion")
    controller = BaselineController(profile=profile, stagnation_threshold=99)
    state = OptimizationState(
        current_incumbent=ArtifactPreview(
            artifact_id="best",
            score=0.1,
            is_feasible=False,
            summary_descriptor="failing artifact",
        )
    )
    state.update_from_memory(
        errors=["AssertionError: output contract mismatch"],
        revision_signals=["Repair the verification failure."],
        uncertainty_flags=[],
    )

    action = controller.plan(state)

    assert action.operator_choice == "verification_driven_repair"
    assert action.branch_width == profile.branch_width
    assert "Self-reflect" in action.query_plan[0]


def test_baseline_memory_filters_follow_profile_policies():
    from baseline_profiles import filter_memory_snapshot, get_baseline_profile

    snapshot = {
        "total_events": 5,
        "operator_performance": {"local_revision": {"success_rate": 0.5}},
        "critical_rules": ["preserve API"],
        "recent_revision_signals": ["old", "new"],
        "global_llm_summary": "summary",
        "extra": "should not leak",
    }

    assert filter_memory_snapshot(get_baseline_profile("funsearch"), snapshot) == {}
    assert filter_memory_snapshot(
        get_baseline_profile("textgrad"), snapshot
    ) == {"recent_revision_signals": ["new"]}

    adaevolve_memory = filter_memory_snapshot(get_baseline_profile("adaevolve"), snapshot)
    assert adaevolve_memory["operator_performance"] == snapshot["operator_performance"]
    assert "extra" not in adaevolve_memory


def test_operator_library_exposes_baseline_specific_operators():
    from operator_library import OperatorLibrary

    library = OperatorLibrary()
    required = {
        "meta_evolved_operator",
        "compile_prompt",
        "self_feedback_revision",
        "mcts_guided_search",
        "verification_driven_repair",
        "error_repair",
        "diff_based_rewrite",
        "metric_aware_search",
        "permutation_local_search",
        "reliability_repair",
        "resource_balancing_local_search",
        "certified_numerical_search",
        "autocorrelation_gradient_search",
        "prefix_cache_ordering_search",
        "graph_routing_local_search",
        "expert_load_balancing_search",
        "frontier_algorithm_search",
        "arc_grid_program_induction",
    }

    assert all(library.get_operator(name) is not None for name in required)


def test_strategy_adapter_rejects_meta_operator_templates():
    from operator_library import OperatorLibrary
    from optimization_controller import RuleBasedController
    from optimization_memory import OptimizationMemory
    from strategy_adapter import StrategyAdapter

    class MetaOperatorClient:
        def generate(self, prompt, system_instruction, n=1):
            return [
                '{"name": "bad", "description": "bad", '
                '"instruction_template": "Format your response strictly as a JSON object '
                'with required keys name, description, instruction_template."}'
            ]

    controller = RuleBasedController(stagnation_threshold=5)
    library = OperatorLibrary()
    adapter = StrategyAdapter(
        llm_client=MetaOperatorClient(),
        controller=controller,
        operator_library=library,
    )

    report = adapter._introduce_new_operator(OptimizationMemory(llm_client=None))

    assert report is not None
    assert report.startswith("Rejected new operator")
    assert controller.config["preferred_operator"] is None
    assert library.get_operator("dynamic_bad") is None


def test_context_includes_baseline_profile_when_supplied():
    from baseline_profiles import get_baseline_profile
    from context_assembler import ContextAssembler

    profile = get_baseline_profile("opro")
    bundle = ContextAssembler().build(
        state={
            "objective": "Improve a prompt.",
            "active_constraints": [],
            "current_artifact": "baseline artifact",
            "unresolved_errors": [],
        },
        query_results=[],
        archive_snapshot={},
        memory_snapshot={},
        action_package={
            "operator_choice": "recombination",
            "baseline_profile": profile.to_dict(),
        },
    )

    assert "### 1.5 BASELINE CONFIGURATION" in bundle.formatted_prompt
    assert "System: OPRO" in bundle.formatted_prompt
    assert "Evaluation protocol: scalar_score" in bundle.formatted_prompt


def test_context_includes_evaluator_metric_breakdown():
    from context_assembler import ContextAssembler

    bundle = ContextAssembler().build(
        state={
            "objective": "Maximize combined_score.",
            "active_constraints": [],
            "current_artifact": "def solve(): pass",
            "unresolved_errors": [],
        },
        query_results=[],
        archive_snapshot={
            "best_score": 35.7,
            "best_artifact_id": "best",
            "best_metrics": {
                "runtime_score": 35.7,
                "combined_score": 35.7,
                "success_rate": 0.04,
                "max_kvpr": 35.67,
            },
            "top_metric_summaries": [
                {
                    "artifact_id": "best",
                    "is_passed": True,
                    "metrics": {"combined_score": 35.7, "success_rate": 0.04},
                }
            ],
        },
        memory_snapshot={
            "recent_metric_observations": [
                "event=1, operator=metric_aware_search, passed=True, metrics={combined_score=35.7}"
            ]
        },
        action_package={"operator_choice": "metric_aware_search"},
    )

    assert "### 2.6 EVALUATOR METRIC BREAKDOWN" in bundle.formatted_prompt
    assert "success_rate=0.04" in bundle.formatted_prompt
    assert "[Recent Metric Observations]" in bundle.formatted_prompt


def test_result_archive_rejects_failed_best_candidate():
    from pathlib import Path

    from run import _result_dir, save_result_if_best

    is_new_best, _out_dir, previous = save_result_if_best(
        llm_name="pytest_mock",
        task_slug="failed_candidate",
        benchmark_task=None,
        cfg={"llm": {"provider": "mock", "model": "mock"}, "runner": {}},
        cfg_path=Path(__file__),
        best_record_metrics={"runtime_score": 0.0},
        best_code="invalid",
        best_artifact_id="bad",
        best_is_passed=False,
        objective_key="runtime_score",
        maximize=True,
        improvement_history=[],
        elapsed_sec=0.0,
    )

    assert is_new_best is False
    assert previous is None
    assert _out_dir == _result_dir(
        baseline_name="opticom",
        task_slug="failed_candidate",
        llm_name="pytest_mock",
    )


def test_baseline_artifact_summary_marks_no_pass(tmp_path):
    from run_baselines import _write_artifacts

    summary = _write_artifacts(
        out_dir=tmp_path,
        baseline="textgrad",
        benchmark_task=None,
        bundle={
            "code": "invalid",
            "metrics": {"runtime_score": 0.0, "sum_radii": 0.0},
            "objective_key": "runtime_score",
            "artifact_id": "bad",
            "is_passed": False,
            "improvement_history": [],
        },
        elapsed=0.1,
    )

    assert summary["best_passed"] is False
    assert (tmp_path / "textgrad" / "summary.json").exists()


def test_result_archive_layout_is_baseline_task_model(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import run

    monkeypatch.setattr(run, "RESULT_ROOT", tmp_path)
    cfg = {
        "llm": {"provider": "mock", "model": "unit"},
        "runner": {"baseline": "gepa", "max_iterations": 1, "stagnation_threshold": 3},
        "benchmark": {"enabled": False},
    }

    is_new_best, out_dir, previous = run.save_result_if_best(
        llm_name="mock_unit",
        task_slug="math__circle_packing",
        benchmark_task=None,
        cfg=cfg,
        cfg_path=Path(__file__),
        best_record_metrics={
            "runtime_score": 1.23,
            "sum_radii": 3.21,
            "combined_score": 1.23,
            "c4_bound": 0.35,
            "r_max": 1.48,
            "coeffs": [1.0, -0.01, 2e-4],
        },
        best_code="print('ok')",
        best_artifact_id="art_ok",
        best_is_passed=True,
        objective_key="runtime_score",
        maximize=True,
        improvement_history=[],
        elapsed_sec=0.5,
    )

    assert is_new_best is True
    assert previous is None
    assert out_dir == tmp_path / "gepa" / "math__circle_packing" / "mock_unit"
    assert (out_dir / "result.json").exists()
    result_data = json.loads((out_dir / "result.json").read_text(encoding="utf-8"))
    assert result_data["c4_bound"] == 0.35
    assert result_data["r_max"] == 1.48
    assert result_data["coeffs"] == [1.0, -0.01, 2e-4]


def test_runtime_summary_collects_result_and_baseline_suite_rows(tmp_path):
    import json

    from summarize_runtimes import collect_runtime_rows

    result_root = tmp_path / "result"
    baseline_runs_root = tmp_path / "baseline_runs"

    old_result = result_root / "mock_old" / "math__old_task" / "result.json"
    old_result.parent.mkdir(parents=True)
    old_result.write_text(
        json.dumps(
            {
                "elapsed_sec": 8,
                "task_slug": "math__old_task",
                "runtime_score": 0.2,
                "llm": {"provider": "mock", "model": "old"},
            }
        ),
        encoding="utf-8",
    )

    new_result = result_root / "gepa" / "math__new_task" / "mock_new" / "result.json"
    new_result.parent.mkdir(parents=True)
    new_result.write_text(
        json.dumps(
            {
                "elapsed_sec": 3,
                "task_slug": "math__new_task",
                "benchmark_task_id": "new_task",
                "baseline": "gepa",
                "model_slug": "mock_new",
                "runtime_score": 0.7,
            }
        ),
        encoding="utf-8",
    )

    suite_file = baseline_runs_root / "20260101_000000" / "suite_summary.json"
    suite_file.parent.mkdir(parents=True)
    archive_dir = result_root / "textgrad" / "math__suite_task" / "mock_suite"
    suite_file.write_text(
        json.dumps(
            {
                "created_at": "2026-01-01T00:00:00",
                "benchmark": "suite_task",
                "runs": [
                    {
                        "baseline": "textgrad",
                        "status": "ok",
                        "elapsed_sec": 1.5,
                        "runtime_score": 0.9,
                        "archive_dir": str(archive_dir),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    rows = collect_runtime_rows(result_root, baseline_runs_root, source="all")
    rows.sort(key=lambda row: row.elapsed_sec)

    assert [row.elapsed_sec for row in rows] == [1.5, 3.0, 8.0]
    assert [(row.task_id, row.baseline, row.model, row.source) for row in rows] == [
        ("suite_task", "textgrad", "mock_suite", "baseline_suite"),
        ("new_task", "gepa", "mock_new", "best_result"),
        ("old_task", "opticom", "mock_old", "best_result"),
    ]


def test_benchmark_preflight_copies_eplb_asset_from_task_root(tmp_path):
    from benchmark_adapter import BenchmarkTask
    from benchmark_preflight import ensure_task_assets

    task_dir = tmp_path / "eplb"
    evaluator_dir = task_dir / "evaluator"
    evaluator_dir.mkdir(parents=True)
    initial_path = task_dir / "initial_program.py"
    evaluator_path = evaluator_dir / "evaluator.py"
    root_asset = task_dir / "expert-load.json"
    initial_path.write_text("# noop\n", encoding="utf-8")
    evaluator_path.write_text("def evaluate(program_path): return {'score': 1.0}\n", encoding="utf-8")
    root_asset.write_text('{"ok": true}\n', encoding="utf-8")

    task = BenchmarkTask(
        task_id="eplb",
        category="ADRS",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )

    ok, missing, actions = ensure_task_assets(task, auto_download=True)

    assert ok is True
    assert missing == []
    assert (evaluator_dir / "expert-load.json").read_text(encoding="utf-8") == '{"ok": true}\n'
    assert any("Copied" in action for action in actions)


def test_benchmark_preflight_downloads_cloudcast_assets_without_wget(tmp_path, monkeypatch):
    from benchmark_adapter import BenchmarkTask
    import benchmark_preflight
    from benchmark_preflight import ensure_task_assets

    task_dir = tmp_path / "cloudcast"
    evaluator_dir = task_dir / "evaluator"
    evaluator_dir.mkdir(parents=True)
    initial_path = task_dir / "initial_program.py"
    evaluator_path = evaluator_dir / "evaluator.py"
    initial_path.write_text("# noop\n", encoding="utf-8")
    evaluator_path.write_text("def evaluate(program_path): return {'score': 1.0}\n", encoding="utf-8")

    downloaded = []

    def fake_download(url, target):
        downloaded.append((url, target))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("asset\n", encoding="utf-8")

    def fail_script(script):
        raise AssertionError("download script should not be needed when direct downloads work")

    monkeypatch.setattr(benchmark_preflight, "_download_url", fake_download)
    monkeypatch.setattr(benchmark_preflight, "_run_download_script", fail_script)

    task = BenchmarkTask(
        task_id="cloudcast",
        category="ADRS",
        root_dir=task_dir,
        initial_program_path=initial_path,
        evaluator_module_path=evaluator_path,
    )

    ok, missing, actions = ensure_task_assets(task, auto_download=True)

    assert ok is True
    assert missing == []
    assert len(downloaded) == 7
    assert (evaluator_dir / "profiles" / "cost.csv").exists()
    assert any("Downloading" in action for action in actions)
