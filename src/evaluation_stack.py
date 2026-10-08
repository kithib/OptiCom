import re
import math
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ==========================================
# 0. Mock Data Structures (From Upstream)
# ==========================================
# We define these here lightly so the file can run standalone.
# In the real framework, they are imported from `problem_formulator.py` and `execution_harness.py`.

@dataclass
class Budget:
    max_llm_calls: int = 50
    max_sandbox_runs: int = 100
    max_wall_clock_mins: float = 60.0
    max_api_cost_usd: float = 10.0

@dataclass
class OptimizationProblem:
    task_description: str
    artifact_type: str
    objective: str
    constraints: List[str]
    budget: Budget
    eval_protocol: str
    admission_criterion: str

@dataclass
class GeneratedArtifact:
    artifact_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ExecutionTrace:
    artifact_id: str
    is_completed: bool
    return_code: int
    stdout: str
    stderr: str
    execution_time_ms: float
    reproduction_command: str
    metrics: Dict[str, Any] = field(default_factory=dict)

# Import the LLM Client from the synthesizer module for Gate 5
from artifact_synthesizer import BaseLLMClient

# ==========================================
# 1. Output Data Structures
# ==========================================

class FailureType(str, Enum):
    """Categorizes the exact layer where the artifact failed."""
    NONE = "none"                                 # Passed all gates
    FEASIBILITY_ERROR = "feasibility_error"       # Failed to run, syntax error, hard crash
    CORRECTNESS_ERROR = "correctness_error"       # Ran, but failed assertions/unit tests
    OBJECTIVE_SUBOPTIMAL = "objective_suboptimal" # Passed tests, but score is too low
    COST_EXCEEDED = "cost_exceeded"               # Too slow, uses too much memory/tokens
    ROBUSTNESS_FAILURE = "robustness_failure"     # Fails on edge cases

@dataclass
class EvaluationRecord:
    """
    The unified output of the Evaluation Stack.
    Matches the schema defined in the architecture diagram perfectly.
    """
    artifact_id: str
    metrics: Dict[str, float]       # e.g., {"score": 0.95, "latency": 120.5, "robustness": 0.9}
    failure_type: FailureType       # Where did it fail?
    diagnostics: str                # Raw extraction of the error/issue
    confidence: float               # How confident is the stack in this evaluation (0.0 to 1.0)
    revision_signal: str            # ★ The core textual feedback for the next generation round
    is_passed: bool                 # Helper flag: Did it pass all mandatory gates?

# ==========================================
# 2. Evaluation Stack Implementation
# ==========================================

class EvaluationStack:
    """
    Module ⑨: Evaluation Stack.
    Implements a 5-layer short-circuiting evaluation funnel dynamically driven by the Problem Formulator.
    """
    
    def __init__(
        self,
        llm_client: BaseLLMClient,
        enable_robustness: bool = False,
        cost_soft_limit_ms: Optional[float] = 30000.0,
    ):
        # Inject the LLM client for advanced heuristic evaluation (like Robustness)
        self.llm_client = llm_client
        self.enable_robustness = enable_robustness
        self.cost_soft_limit_ms = (
            float("inf") if cost_soft_limit_ms is None else float(cost_soft_limit_ms)
        )

    def evaluate(self, artifacts: List[GeneratedArtifact], traces: List[ExecutionTrace], problem: OptimizationProblem) -> List[EvaluationRecord]:
        """
        Processes a batch of artifacts and their corresponding execution traces 
        through the 5-layer evaluation stack using the defined optimization problem.
        """
        records = []
        # Match trace to artifact (assuming 1:1 mapping and aligned lists for MVP)
        for artifact, trace in zip(artifacts, traces):
            record = self._evaluate_single(artifact, trace, problem)
            records.append(record)
        return records

    def _evaluate_single(self, artifact: GeneratedArtifact, trace: ExecutionTrace, problem: OptimizationProblem) -> EvaluationRecord:
        """
        The 5-layer funnel. Returns early (short-circuits) if a hard gate fails.
        """
        metrics: Dict[str, float] = {}
        objective_type = problem.objective
        obj_key = f"{objective_type.value if hasattr(objective_type, 'value') else objective_type}_score"
        
        # ---------------------------------------------------------
        # Gate 1: Feasibility (Can it run? Is it legal?)
        # ---------------------------------------------------------
        if not trace.is_completed:
            diagnostics = trace.stderr.strip() or trace.stdout.strip()
            revision_signal = self._generate_revision_signal(FailureType.FEASIBILITY_ERROR, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id, metrics=metrics, failure_type=FailureType.FEASIBILITY_ERROR,
                diagnostics=diagnostics[:500], confidence=1.0, revision_signal=revision_signal, is_passed=False
            )
            
        if trace.return_code != 0:
            diagnostics = trace.stderr.strip() or trace.stdout.strip()
            # Distinguish syntax errors from runtime logic/assertion failures
            if (
                "SyntaxError" in diagnostics
                or "IndentationError" in diagnostics
                or "ModuleNotFoundError" in diagnostics
                or "No module named" in diagnostics
                or "ImportError" in diagnostics
            ):
                failure_type = FailureType.FEASIBILITY_ERROR
            else:
                failure_type = FailureType.CORRECTNESS_ERROR
                
            revision_signal = self._generate_revision_signal(failure_type, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id, metrics=metrics, failure_type=failure_type,
                diagnostics=diagnostics[:1000], confidence=0.9, revision_signal=revision_signal, is_passed=False
            )
            
        # ---------------------------------------------------------
        # Gate 2: Correctness (Did it pass tests?)
        # ---------------------------------------------------------
        # Check if stdout contains common test failure keywords
        if "AssertionError" in trace.stdout or "FAILED" in trace.stdout:
            diagnostics = self._extract_failed_test_case(trace.stdout)
            revision_signal = self._generate_revision_signal(FailureType.CORRECTNESS_ERROR, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id, metrics=metrics, failure_type=FailureType.CORRECTNESS_ERROR,
                diagnostics=diagnostics, confidence=0.9, revision_signal=revision_signal, is_passed=False
            )

        # Some benchmark evaluators encode hard constraint failure as numeric metadata
        # while still returning a finite score. Treat explicit invalidity as a failed
        # correctness gate so invalid candidates do not become feasible incumbents.
        invalidity_reason = self._detect_explicit_invalidity(trace.metrics)
        if invalidity_reason:
            score = self._score_from_trace_metrics(trace.metrics)
            metrics[obj_key] = score
            metrics["score"] = score
            self._copy_numeric_trace_metrics(trace.metrics, metrics, obj_key)
            metrics["execution_time_ms"] = trace.execution_time_ms
            diagnostics = f"Benchmark marked the artifact invalid: {invalidity_reason}."
            revision_signal = self._generate_revision_signal(FailureType.CORRECTNESS_ERROR, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id,
                metrics=metrics,
                failure_type=FailureType.CORRECTNESS_ERROR,
                diagnostics=diagnostics,
                confidence=0.95,
                revision_signal=revision_signal,
                is_passed=False,
            )
            
        # ---------------------------------------------------------
        # Gate 3: Objective Value (Read from Problem Formulator)
        # ---------------------------------------------------------
        score = self._score_from_trace_metrics(trace.metrics)
        
        # 关键: 同时写入 objective_type.value 作为 key, 以便 Archive Manager 读取
        metrics[obj_key] = score
        metrics["score"] = score  # 通用 key

        # 把 trace.metrics 里 harness 层已经填好的 benchmark 原生数值指标
        # (sum_radii / combined_score / target_ratio / accuracy ...) 一并带进来,
        # 便于 Archive / run.py 报告"绝对值"而不仅仅是归一化的 *_score.
        self._copy_numeric_trace_metrics(trace.metrics, metrics, obj_key)
        
        # 关键修复: Gate 3 不再使用硬编码 0.7 阈值 (对构造类最大化问题无意义)
        # 改为: 仅当 score 是明确的失败标记(-inf / NaN)时才判定为 OBJECTIVE_SUBOPTIMAL
        # 真正的"是否足够好"交给 Archive Manager 通过 incumbent 比较来处理
        if not isinstance(score, (int, float)) or not math.isfinite(float(score)):
            diagnostics = f"The {objective_type} score is invalid/unparseable: {score}."
            revision_signal = self._generate_revision_signal(FailureType.OBJECTIVE_SUBOPTIMAL, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id, metrics=metrics, failure_type=FailureType.OBJECTIVE_SUBOPTIMAL,
                diagnostics=diagnostics, confidence=0.95, revision_signal=revision_signal, is_passed=False
            )

        # ---------------------------------------------------------
        # Gate 4: Cost (Read from Problem Formulator's Budget)
        # ---------------------------------------------------------
        metrics["execution_time_ms"] = trace.execution_time_ms
        
        # 软预算由配置控制；真正的硬超时仍由 harness.benchmark_timeout 决定。
        soft_limit_ms = self.cost_soft_limit_ms
        if trace.execution_time_ms > soft_limit_ms:
            diagnostics = f"Execution took {trace.execution_time_ms:.2f}ms, exceeding the soft budget limit of {soft_limit_ms}ms."
            revision_signal = self._generate_revision_signal(FailureType.COST_EXCEEDED, diagnostics)
            return EvaluationRecord(
                artifact_id=trace.artifact_id, metrics=metrics, failure_type=FailureType.COST_EXCEEDED,
                diagnostics=diagnostics, confidence=1.0, revision_signal=revision_signal, is_passed=False
            )

        # ---------------------------------------------------------
        # Gate 5: Robustness (Generalization / Edge cases via LLM)
        # ---------------------------------------------------------
        # Call the LLM to evaluate robustness ONLY IF enabled (saves API cost & time)
        if self.enable_robustness:
            robustness_score, robustness_feedback = self._evaluate_robustness_with_llm(artifact, trace, problem)
            metrics["robustness_score"] = robustness_score
            
            if robustness_score < 0.3: # Lowered threshold for configurable robustness
                diagnostics = f"LLM Robustness Evaluation Failed (Score: {robustness_score}). Feedback: {robustness_feedback}"
                revision_signal = self._generate_revision_signal(FailureType.ROBUSTNESS_FAILURE, diagnostics)
                return EvaluationRecord(
                    artifact_id=trace.artifact_id, metrics=metrics, failure_type=FailureType.ROBUSTNESS_FAILURE,
                    diagnostics=diagnostics, confidence=0.85, revision_signal=revision_signal, is_passed=False
                )
        
        return EvaluationRecord(
            artifact_id=trace.artifact_id,
            metrics=metrics,
            failure_type=FailureType.NONE,
            diagnostics="All gates passed successfully.",
            confidence=0.9,
            revision_signal="Artifact performs exceptionally well and is robust. Minor polishing can be applied if needed.",
            is_passed=True
        )

    # --- Helper Methods for Signal Extraction ---

    @staticmethod
    def _is_numeric(value: Any) -> bool:
        return isinstance(value, (int, float)) and not isinstance(value, bool)

    @classmethod
    def _is_small_json_metric(cls, value: Any, *, max_items: int = 256) -> bool:
        """Allow compact construction data such as coefficient vectors into archives."""
        if value is None or isinstance(value, (str, bool)):
            return True
        if cls._is_numeric(value):
            return math.isfinite(float(value))
        if isinstance(value, list):
            if len(value) > max_items:
                return False
            return all(cls._is_small_json_metric(v, max_items=max_items) for v in value)
        if isinstance(value, dict):
            if len(value) > max_items:
                return False
            return all(
                isinstance(k, str) and cls._is_small_json_metric(v, max_items=max_items)
                for k, v in value.items()
            )
        return False

    @classmethod
    def _score_from_trace_metrics(cls, trace_metrics: Dict[str, Any]) -> float:
        score = trace_metrics.get("score", None)
        if cls._is_numeric(score):
            return float(score)
        for key in ("combined_score", "sum_radii", "target_ratio",
                    "accuracy", "reward", "value"):
            value = trace_metrics.get(key)
            if cls._is_numeric(value):
                return float(value)
        return -float("inf")

    @classmethod
    def _copy_numeric_trace_metrics(
        cls,
        trace_metrics: Dict[str, Any],
        metrics: Dict[str, float],
        obj_key: str,
    ) -> None:
        reserved = {"score", obj_key}
        preserve_json_keys = {"coeffs", "metric_version"}
        for key, value in (trace_metrics or {}).items():
            if key in reserved:
                continue
            if isinstance(value, bool):
                metrics.setdefault(key, float(value))
            elif cls._is_numeric(value) and math.isfinite(float(value)):
                metrics.setdefault(key, float(value))
            elif key in preserve_json_keys and cls._is_small_json_metric(value):
                metrics.setdefault(key, value)

    @staticmethod
    def _detect_explicit_invalidity(trace_metrics: Dict[str, Any]) -> str:
        validity = (trace_metrics or {}).get("validity")
        if isinstance(validity, bool) and not validity:
            return "validity=False"
        if isinstance(validity, (int, float)) and not isinstance(validity, bool) and float(validity) <= 0.0:
            return f"validity={validity}"
        for key in ("valid", "is_valid"):
            value = (trace_metrics or {}).get(key)
            if value is False:
                return f"{key}=False"
        return ""

    def _evaluate_robustness_with_llm(self, artifact: GeneratedArtifact, trace: ExecutionTrace, problem: OptimizationProblem) -> tuple[float, str]:
        """
        Uses the LLM to evaluate the robustness and edge-case handling of the artifact.
        Returns a tuple of (score: float, feedback: str).
        """
        sys_instruction = (
            "You are an expert software reviewer. Your task is to evaluate the robustness "
            "and generalization capabilities of the provided code artifact. Think about edge cases, "
            "empty inputs, malformed data, and large scale limits.\n\n"
            "Format your response EXACTLY as follows:\n"
            "SCORE: <a float between 0.0 and 1.0>\n"
            "FEEDBACK: <brief explanation of vulnerabilities or edge cases>"
        )
        
        user_prompt = (
            f"Task Objective: {problem.task_description}\n\n"
            f"Artifact to Evaluate:\n```python\n{artifact.content}\n```\n\n"
            f"Execution Output / Trace:\n{trace.stdout[:500]}\n"
        )
        
        try:
            responses = self.llm_client.generate(user_prompt, sys_instruction, n=1)
            raw_response = responses[0]
            
            # Parse the structured response
            score = 1.0
            feedback = "Looks robust."
            
            score_match = re.search(r"SCORE:\s*([0-9.]+)", raw_response)
            if score_match:
                score = float(score_match.group(1))
                
            feedback_match = re.search(r"FEEDBACK:\s*(.*)", raw_response, re.DOTALL)
            if feedback_match:
                feedback = feedback_match.group(1).strip()
                
            return score, feedback
        except Exception as e:
            # Graceful fallback if the LLM call fails
            return 0.5, f"Robustness evaluation failed due to LLM error: {str(e)}"

    def _extract_failed_test_case(self, stdout: str) -> str:
        """Attempts to parse exact failure reasons from test output."""
        lines = stdout.split('\n')
        for i, line in enumerate(lines):
            if "AssertionError" in line or "Exception" in line:
                return "\n".join(lines[max(0, i-2):i+3])
        return "Unknown correctness failure (Parsed from STDOUT)."

    def _generate_revision_signal(self, failure_type: FailureType, diagnostics: str) -> str:
        """
        Constructs the 'Textual Gradient' or 'Verbal Feedback' for the LLM.
        This provides explicit, actionable instructions for the next iteration.
        """
        if failure_type == FailureType.FEASIBILITY_ERROR:
            return f"[Actionable Feedback] The code failed to execute or threw an exception. Analyze this stack trace and fix the root cause:\n{diagnostics}"
        elif failure_type == FailureType.CORRECTNESS_ERROR:
            return f"[Actionable Feedback] The code runs, but fails functional requirements or tests. Focus on fixing this specific logic failure:\n{diagnostics}"
        elif failure_type == FailureType.OBJECTIVE_SUBOPTIMAL:
            return f"[Actionable Feedback] The functional logic is correct, but the objective value is too low. Explore alternative algorithms. Details: {diagnostics}"
        elif failure_type == FailureType.COST_EXCEEDED:
            return f"[Actionable Feedback] The artifact exceeded the budget (time/tokens). Optimize for time/space complexity. Details: {diagnostics}"
        elif failure_type == FailureType.ROBUSTNESS_FAILURE:
            return f"[Actionable Feedback] The artifact is fragile against edge cases. Address the following robustness vulnerabilities:\n{diagnostics}"
        
        return "[Actionable Feedback] Keep refining."

# ==========================================
# 3. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    print("Initiating Evaluation Stack with LLM Robustness...")
    
    class MockClient:
        def generate(self, p, s, n=1):
            if "fragile_sort" in p:
                return ["SCORE: 0.3\nFEEDBACK: Code crashes on empty arrays or edge cases. Not robust."]
            return ["SCORE: 0.95\nFEEDBACK: Excellent robustness, handles typical edge cases safely."]

    llm_client = MockClient()
    
    eval_stack = EvaluationStack(llm_client=llm_client)
    
    # 1. Mocking the Optimization Problem (from Problem Formulator)
    problem_definition = OptimizationProblem(
        task_description="Write an efficient sorting function.",
        artifact_type="code",
        objective="runtime",
        constraints=["Must not use external libraries"],
        budget=Budget(max_wall_clock_mins=5.0), # 5 minute overall budget
        eval_protocol="isolated_sandbox_execution",
        admission_criterion="Score > 0.7"
    )
    
    # 2. Mocking Artifacts
    artifacts = [
        # Pass: Efficient and logically sound
        GeneratedArtifact("art_correct", "def sort_arr(arr):\n    return sorted(arr) if arr else []"),
        # Gate 1 Fail: Syntax error
        GeneratedArtifact("art_feasibility", "def sort_arr(arr):\n    return sorted(arr"),
        # Gate 2 Fail: Logic Error (returns None)
        GeneratedArtifact("art_correctness", "def sort_arr(arr):\n    return arr.sort()"),
        # Gate 3 Fail: Inefficient logic (bad objective score)
        GeneratedArtifact("art_objective", "def sort_arr(arr):\n    # O(N^2) bubble sort implementation..."),
        # Gate 4 Fail: Timeout logic
        GeneratedArtifact("art_cost", "import time\ndef sort_arr(arr):\n    time.sleep(1)\n    return sorted(arr)"),
        # Gate 5 Fail: Logic works but fails edge cases (trigger mock LLM low score)
        GeneratedArtifact("art_robustness", "def fragile_sort(arr):\n    return [arr[0]] + sorted(arr[1:])"),
    ]
    
    # 3. Mocking Traces (Normally provided by Execution Harness)
    traces = [
        # Gate 5 Pass
        ExecutionTrace("art_correct", is_completed=True, return_code=0, stdout="Success", stderr="", execution_time_ms=45.0, metrics={"score": 0.95}, reproduction_command="python art_correct.py"),
        # Fails Gate 1
        ExecutionTrace("art_feasibility", is_completed=True, return_code=1, stdout="", stderr="SyntaxError: unexpected EOF while parsing", execution_time_ms=10.0, metrics={}, reproduction_command="python art_feasibility.py"),
        # Fails Gate 2
        ExecutionTrace("art_correctness", is_completed=True, return_code=0, stdout="FAILED test_sort.py\nAssertionError: Expected [1, 2] but got None", stderr="", execution_time_ms=20.0, metrics={"score": 0.0}, reproduction_command="pytest art_correctness.py"),
        # Fails Gate 3 (Score is 0.5, below 0.7 threshold)
        ExecutionTrace("art_objective", is_completed=True, return_code=0, stdout="Success", stderr="", execution_time_ms=150.0, metrics={"score": 0.5}, reproduction_command="python art_objective.py"),
        # Fails Gate 4 (Execution time is 1200ms, above 500ms soft limit)
        ExecutionTrace("art_cost", is_completed=True, return_code=0, stdout="Success", stderr="", execution_time_ms=1200.0, metrics={"score": 0.9}, reproduction_command="python art_cost.py"),
        # Fails Gate 5 (Mock LLM will give it a 0.3 score)
        ExecutionTrace("art_robustness", is_completed=True, return_code=0, stdout="Success", stderr="", execution_time_ms=50.0, metrics={"score": 0.9}, reproduction_command="python art_robustness.py"),
    ]
    
    # 4. Evaluate the batch
    records = eval_stack.evaluate(artifacts, traces, problem_definition)
    
    # 5. Display the unified Evaluation Records
    print("\n" + "="*60)
    for record in records:
        status = "PASSED" if record.is_passed else "FAILED"
        print(f"[{record.artifact_id}] {status} at Layer: {record.failure_type.name}")
        print(f"  Metrics: {record.metrics}")
        if not record.is_passed:
            print(f"  Diagnostics: {record.diagnostics}")
        print(f"  --> Revision Signal (Textual Gradient):\n      {record.revision_signal}")
        print("-" * 60)
