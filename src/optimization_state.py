import json
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import List, Dict, Optional, Any

# ==========================================
# 1. Core Enum Definitions
# ==========================================

class OptimizationPhase(str, Enum):
    """
    Current phase of the optimization process.
    Matches the defined linear progression: feasibility -> repair -> improvement -> polishing
    """
    FEASIBILITY = "feasibility"   # Exploring to find any valid solution
    REPAIR = "repair"             # Fixing constraints/errors in current candidates
    IMPROVEMENT = "improvement"   # Optimizing the main objective score
    POLISHING = "polishing"       # Fine-tuning, removing dead code, edge-case hardening

# ==========================================
# 2. Supporting Data Structures
# ==========================================

@dataclass
class RemainingBudget:
    """Tracks the depleted budget to prevent infinite loops."""
    llm_calls: int
    sandbox_runs: int
    wall_clock_mins: float
    api_cost_usd: float

    def is_exhausted(self) -> bool:
        """Returns True if any critical budget threshold is hit."""
        return (self.llm_calls <= 0 or 
                self.sandbox_runs <= 0 or 
                self.api_cost_usd <= 0.0)

@dataclass
class OperatorStat:
    """Tracks the success statistics of a specific operator (e.g., 'mutate', 'crossover')."""
    attempts: int = 0
    successes: int = 0
    cumulative_reward: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0

@dataclass
class ArtifactPreview:
    """
    A lightweight representation of a solution.
    We don't store the full AST or massive strings here to keep the state 'compressed'.
    """
    artifact_id: str
    score: float
    is_feasible: bool
    summary_descriptor: str  # e.g., "Uses recursive divide-and-conquer"

# ==========================================
# 3. Core Optimization State
# ==========================================

@dataclass
class OptimizationState:
    """
    The compressed control state of the entire system.
    Provides context for the Optimization Controller to plan the next Action Package.
    """
    # 1. Incumbent & Frontier
    current_incumbent: Optional[ArtifactPreview] = None
    current_frontier_summary: List[str] = field(default_factory=list)
    
    # 2. Errors, Constraints & Budget
    unresolved_errors: List[str] = field(default_factory=list)
    active_constraints: List[str] = field(default_factory=list)
    remaining_budget: RemainingBudget = field(
        default_factory=lambda: RemainingBudget(0, 0, 0.0, 0.0)
    )
    
    # 3. Uncertainty & Statistics
    uncertainty_flags: List[str] = field(default_factory=list)
    operator_success_statistics: Dict[str, OperatorStat] = field(default_factory=dict)
    
    # 4. Signals & Phase
    recent_revision_signals: List[str] = field(default_factory=list)
    current_phase: OptimizationPhase = OptimizationPhase.FEASIBILITY

    # ==========================================
    # System Update Interfaces (Hooks for Archive & Memory)
    # ==========================================

    def consume_budget(self, llm_calls: int = 0, sandbox_runs: int = 0, api_cost_usd: float = 0.0) -> None:
        """Deducts resources from the remaining budget."""
        self.remaining_budget.llm_calls -= llm_calls
        self.remaining_budget.sandbox_runs -= sandbox_runs
        self.remaining_budget.api_cost_usd -= api_cost_usd

    def update_from_archive(self, best_artifact: Optional[ArtifactPreview], frontier_summary: List[str]) -> None:
        """Updates the incumbent and frontier from the Archive Manager."""
        if best_artifact:
            self.current_incumbent = best_artifact
        self.current_frontier_summary = frontier_summary

    def update_from_memory(self, errors: List[str], revision_signals: List[str], uncertainty_flags: List[str]) -> None:
        """Updates insights, errors, and signals from the Optimization Memory."""
        self.unresolved_errors = errors
        self.recent_revision_signals = revision_signals
        self.uncertainty_flags = uncertainty_flags

    def record_operator_execution(self, operator_name: str, is_success: bool, reward: float = 0.0) -> None:
        """Updates the statistics for a specific operator."""
        if operator_name not in self.operator_success_statistics:
            self.operator_success_statistics[operator_name] = OperatorStat()
        
        stat = self.operator_success_statistics[operator_name]
        stat.attempts += 1
        if is_success:
            stat.successes += 1
        stat.cumulative_reward += reward

    def transition_phase(self, new_phase: OptimizationPhase) -> None:
        """Transitions the optimization to a new phase."""
        self.current_phase = new_phase

    def to_json(self) -> str:
        """Serializes the compressed state for logging or LLM context injection."""
        state_dict = asdict(self)
        # Convert Enum to string for JSON serialization
        state_dict['current_phase'] = self.current_phase.value
        # Add computed properties for operators
        for op_name, stat in self.operator_success_statistics.items():
            state_dict['operator_success_statistics'][op_name]['success_rate'] = stat.success_rate
        return json.dumps(state_dict, indent=2, ensure_ascii=False)

# ==========================================
# 4. MVP State Initialization & Update Test
# ==========================================
if __name__ == "__main__":
    # 1. Initialize State (Simulating start of optimization)
    state = OptimizationState(
        active_constraints=[
            "Must be syntactically valid",
            "Latency < 200ms"
        ],
        remaining_budget=RemainingBudget(
            llm_calls=50, 
            sandbox_runs=100, 
            wall_clock_mins=60.0, 
            api_cost_usd=10.0
        ),
        current_phase=OptimizationPhase.FEASIBILITY
    )

    print("Initial State:")
    print(state.to_json())
    print("\n" + "="*50 + "\n")

    # 2. Simulate an update after a quick inner loop iteration
    # Use the new explicit interfaces instead of direct attribute manipulation
    
    # Consume budget
    state.consume_budget(llm_calls=1, sandbox_runs=1)
    
    # Record operator performance (Controller tried 'code_mutation' and failed)
    state.record_operator_execution(operator_name="code_mutation", is_success=False)
    
    # Update memory insights
    state.update_from_memory(
        errors=["IndexError on boundary condition array[-1]"],
        revision_signals=["Focus on array bounds checking in the next prompt"],
        uncertainty_flags=["Is the provided test case covering empty arrays?"]
    )

    # Found a working but slow solution, update archive and move to IMPROVEMENT phase
    new_incumbent = ArtifactPreview(
        artifact_id="art_001",
        score=0.45,
        is_feasible=True,
        summary_descriptor="Basic brute-force implementation"
    )
    state.update_from_archive(
        best_artifact=new_incumbent, 
        frontier_summary=["Exploring hash-map based approaches"]
    )
    state.transition_phase(OptimizationPhase.IMPROVEMENT)

    print("State after 1st Iteration (Moving to Improvement):")
    print(state.to_json())