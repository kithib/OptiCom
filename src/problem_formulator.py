import json
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any

# ==========================================
# 1. Core Enum Definitions (Corresponding to diagram categories)
# ==========================================

class ArtifactType(str, Enum):
    """Artifact types to be optimized"""
    CODE = "code"
    PROMPT = "prompt"
    ALGORITHM = "algorithm"
    WORKFLOW = "workflow"
    SKILL = "skill"
    CONFIG = "config"

class ObjectiveType(str, Enum):
    """Primary objective functions"""
    CORRECTNESS = "correctness"
    ACCURACY = "accuracy"
    RUNTIME = "runtime"
    COST = "cost"
    ROBUSTNESS = "robustness"

# ==========================================
# 2. Core Data Structure Definitions
# ==========================================

@dataclass
class Budget:
    """4. Resource Budget"""
    max_llm_calls: int = 50           # Maximum LLM calls
    max_sandbox_runs: int = 100       # Maximum sandbox executions
    max_wall_clock_mins: int = 60     # Maximum wall-clock time (minutes)
    max_api_cost_usd: float = 10.0    # Maximum API cost (USD)

@dataclass
class OptimizationProblem:
    """
    Output of the Problem Formulator: Standardized optimization problem definition
    (Covers the 6 core definitions in the diagram)
    """
    task_description: str             # Original natural language task
    artifact_type: ArtifactType       # 1. Artifact type
    objective: ObjectiveType          # 2. Objective function
    constraints: List[str]            # 3. List of hard constraints
    budget: Budget                    # 4. Resource budget
    eval_protocol: str                # 5. Evaluation protocol
    admission_criterion: str          # 6. Admission criterion

    def to_json(self) -> str:
        """Helper method: for printing and serialization"""
        # Simple helper to convert Dataclass to dict
        data = {
            "task_description": self.task_description,
            "artifact_type": self.artifact_type.value,
            "objective": self.objective.value,
            "constraints": self.constraints,
            "budget": self.budget.__dict__,
            "eval_protocol": self.eval_protocol,
            "admission_criterion": self.admission_criterion
        }
        return json.dumps(data, indent=2, ensure_ascii=False)

# ==========================================
# 3. Rule-Based Formulator Implementation
# ==========================================

class RuleBasedFormulator:
    """
    Version 1: Rule-based problem formulator.
    Applies preset templates based on the input task_type.
    """
    
    def __init__(self):
        # Preset template library (Priors / Templates)
        self.templates: Dict[str, Dict[str, Any]] = {
            "code_generation": {
                "artifact_type": ArtifactType.CODE,
                "objective": ObjectiveType.CORRECTNESS,
                "constraints": [
                    "Must be syntactically valid in the target language",
                    "Must adhere to established coding conventions",
                    "Must not contain malicious or destructive operations"
                ],
                "budget": Budget(max_llm_calls=20, max_sandbox_runs=50),
                "eval_protocol": "isolated_sandbox_execution",
                "admission_criterion": "Test pass rate > Incumbent, or (pass rate == Incumbent AND execution time < Incumbent)"
            },
            "prompt_optimization": {
                "artifact_type": ArtifactType.PROMPT,
                "objective": ObjectiveType.ACCURACY,
                "constraints": [
                    "Prompt token length must stay within the model's context window limits",
                    "Output format instructions must be strictly preserved"
                ],
                "budget": Budget(max_llm_calls=100, max_sandbox_runs=100),
                "eval_protocol": "llm_as_a_judge_on_validation_set",
                "admission_criterion": "Win rate > 50% against Incumbent via LLM judge, or lower token cost with equivalent accuracy"
            },
            "algorithm_tuning": {
                "artifact_type": ArtifactType.ALGORITHM,
                "objective": ObjectiveType.RUNTIME,
                "constraints": [
                    "Must maintain 100% functional correctness relative to the original algorithm",
                    "Must not exceed standard memory allocation bounds for the environment"
                ],
                # 关键修复: 数值优化类 benchmark (heilbronn / circle_packing ...) 单轮耗时
                # 40~120s 属于常态, max_iter=20 时上限 benchmark_timeout=600s 也会接近封顶,
                # 原 60 分钟 wall-clock 会把整轮框架活活拖死; 同步放大 sandbox_runs。
                "budget": Budget(max_llm_calls=80, max_sandbox_runs=400, max_wall_clock_mins=180),
                "eval_protocol": "microbenchmark_profiling",
                "admission_criterion": "Statistically significant reduction in p99 latency or memory usage, with 0 correctness regressions"
            },
            "workflow_design": {
                "artifact_type": ArtifactType.WORKFLOW,
                "objective": ObjectiveType.ROBUSTNESS,
                "constraints": [
                    "Must not contain cyclic dependencies or infinite loops",
                    "Must implement resilient error handling and fallback mechanisms"
                ],
                "budget": Budget(max_llm_calls=20, max_sandbox_runs=50),
                "eval_protocol": "mock_api_integration_tests",
                "admission_criterion": "Higher overall success rate or better fault-tolerance under simulated API failure conditions"
            },
            "skill_discovery": {
                "artifact_type": ArtifactType.SKILL,
                "objective": ObjectiveType.ACCURACY,
                "constraints": [
                    "Must not rely on hardcoded local paths or user-specific environment variables",
                    "Must strictly implement standard I/O signature and specifications"
                ],
                "budget": Budget(max_llm_calls=40, max_sandbox_runs=80),
                "eval_protocol": "tool_calling_eval_suite",
                "admission_criterion": "Higher success rate on held-out validation tasks with fewer invalid tool calls"
            },
            "config_tuning": {
                "artifact_type": ArtifactType.CONFIG,
                "objective": ObjectiveType.COST,
                "constraints": [
                    "Must comply with overall system hardware constraints and quotas",
                    "Must not degrade system stability or availability"
                ],
                "budget": Budget(max_llm_calls=10, max_sandbox_runs=300),
                "eval_protocol": "load_testing_framework",
                "admission_criterion": "Lower resource cost or higher throughput while strictly satisfying SLA requirements"
            }
        }

    def formulate(self, task_description: str, task_type: str,
                  budget_overrides: Optional[Dict[str, Any]] = None) -> OptimizationProblem:
        """
        Converts natural language tasks and task types into standardized optimization problems.

        Args:
            task_description: 任务自然语言描述。
            task_type: 预设模板名称 (例如 "algorithm_tuning")。
            budget_overrides: 可选的 Budget 字段覆盖字典, 支持
                `max_llm_calls` / `max_sandbox_runs` / `max_wall_clock_mins` / `max_api_cost_usd`,
                方便在 yaml 层级针对特定 benchmark 调大/调小资源预算而无需改代码。
        """
        if task_type not in self.templates:
            raise ValueError(f"Unsupported Task Type: {task_type}. Currently supported: {list(self.templates.keys())}")

        template = self.templates[task_type]

        # ---- 按需覆盖 Budget ----
        tmpl_budget: Budget = template["budget"]
        if budget_overrides:
            tmpl_budget = Budget(
                max_llm_calls=int(budget_overrides.get("max_llm_calls", tmpl_budget.max_llm_calls)),
                max_sandbox_runs=int(budget_overrides.get("max_sandbox_runs", tmpl_budget.max_sandbox_runs)),
                max_wall_clock_mins=int(budget_overrides.get("max_wall_clock_mins", tmpl_budget.max_wall_clock_mins)),
                max_api_cost_usd=float(budget_overrides.get("max_api_cost_usd", tmpl_budget.max_api_cost_usd)),
            )

        # Instantiate and return OptimizationProblem
        return OptimizationProblem(
            task_description=task_description,
            artifact_type=template["artifact_type"],
            objective=template["objective"],
            constraints=template["constraints"],
            budget=tmpl_budget,
            eval_protocol=template["eval_protocol"],
            admission_criterion=template["admission_criterion"]
        )

# ==========================================
# 4. MVP Quick Test (Execution Harness)
# ==========================================
if __name__ == "__main__":
    formulator = RuleBasedFormulator()

    print("Test Scenario 1: Code Generation Task")
    task_1 = "Write an efficient Python function to implement fast exponentiation of matrices."
    problem_1 = formulator.formulate(task_description=task_1, task_type="code_generation")
    print(problem_1.to_json())
    print("\n" + "="*50 + "\n")

    print("Test Scenario 2: Prompt Optimization Task")
    task_2 = "Optimize a prompt for extracting news summaries, requiring it to be as cost-effective and accurate as possible."
    problem_2 = formulator.formulate(task_description=task_2, task_type="prompt_optimization")
    print(problem_2.to_json())
    print("\n" + "="*50 + "\n")
    
    print("Test Scenario 3: Algorithm Tuning Task")
    task_3 = "Tune a sorting algorithm for highly skewed data distributions."
    problem_3 = formulator.formulate(task_description=task_3, task_type="algorithm_tuning")
    print(problem_3.to_json())
    print("\n" + "="*50 + "\n")

    print("Test Scenario 4: Workflow Design Task")
    task_4 = "Design an agentic workflow for automatically categorizing and responding to customer support tickets."
    problem_4 = formulator.formulate(task_description=task_4, task_type="workflow_design")
    print(problem_4.to_json())
    print("\n" + "="*50 + "\n")

    print("Test Scenario 5: Skill Discovery Task")
    task_5 = "Synthesize a reusable skill for scraping e-commerce product pricing tables across different DOM structures."
    problem_5 = formulator.formulate(task_description=task_5, task_type="skill_discovery")
    print(problem_5.to_json())
    print("\n" + "="*50 + "\n")

    print("Test Scenario 6: Configuration Tuning Task")
    task_6 = "Optimize the hyperparameters of a Redis caching layer to minimize latency while strictly controlling cost."
    problem_6 = formulator.formulate(task_description=task_6, task_type="config_tuning")
    print(problem_6.to_json())