import json
import sys
import importlib.util
from dataclasses import dataclass, asdict, field
from enum import Enum
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

# ==========================================
# 1. Query Type Definitions (Based on Diagram)
# ==========================================

class QueryType(str, Enum):
    """
    The 5 core query types defined in the system architecture.
    """
    ENVIRONMENT = "environment"  # Environment, dependencies, schema, APIs
    MEMORY = "memory"            # Historical failures, existing fixes, effective strategies
    ARTIFACT = "artifact"        # Current artifact structure, diff, hotspots
    KNOWLEDGE = "knowledge"      # Documentation, papers, specifications
    HUMAN = "human"              # High-cost clarification (triggered only when necessary)

# ==========================================
# 2. Data Structures for Query Execution
# ==========================================

@dataclass
class QueryResult:
    """
    Represents the evidence/information retrieved by the Query Engine.
    """
    query_type: QueryType
    original_query: str
    evidence: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        data = asdict(self)
        data['query_type'] = self.query_type.value
        return data

# ==========================================
# 2.5 Abstract Query Handlers (Interfaces)
# ==========================================

class BaseQueryHandler(ABC):
    """
    Abstract base class for all query handlers.
    Real implementations (e.g., hitting a vector DB, AST parser) should inherit from this.
    """
    @abstractmethod
    def execute(self, query_string: str) -> QueryResult:
        """Executes the query and returns the structured evidence."""
        pass


class EnvironmentSnapshotHandler(BaseQueryHandler):
    """Lightweight local environment probe used by the default MVP engine."""

    def execute(self, query_string: str) -> QueryResult:
        packages = []
        for pkg in ("numpy", "scipy", "torch", "cvxpy", "numba"):
            status = "available" if importlib.util.find_spec(pkg) else "missing"
            packages.append(f"{pkg}={status}")
        return QueryResult(
            query_type=QueryType.ENVIRONMENT,
            original_query=query_string,
            evidence=(
                "[Environment]\n"
                f"- Python: {sys.version.split()[0]}\n"
                f"- Packages: {', '.join(packages)}\n"
                "- Use only installed dependencies or standard-library fallbacks."
            ),
            metadata={"source": "local_runtime"},
        )


class KnowledgeHeuristicHandler(BaseQueryHandler):
    """Static, deterministic guidance for common optimization query intents."""

    def __init__(self, result_type: QueryType = QueryType.KNOWLEDGE):
        self.result_type = result_type

    def execute(self, query_string: str) -> QueryResult:
        q = query_string.lower()
        if any(k in q for k in ("numerical", "scipy", "l-bfgs", "basinhopping", "annealing")):
            evidence = (
                "[Knowledge]\n"
                "- For continuous constrained optimization, parameterize decisions as a flat vector.\n"
                "- Prefer deterministic multi-start search, then polish the best candidate with a local solver.\n"
                "- Keep the true metric separate from any smoothed or penalized surrogate loss.\n"
                "- Report the exact benchmark-facing output contract, not the internal loss."
            )
        elif any(k in q for k in ("diff", "artifact", "hotspot", "structure")):
            evidence = (
                "[Artifact]\n"
                "- Preserve the incumbent's working public API and output contract.\n"
                "- Make focused changes around the suspected bottleneck or failure point.\n"
                "- Avoid large rewrites unless the current approach is structurally blocked."
            )
        elif any(k in q for k in ("repair", "error", "fix", "failed", "trace")):
            evidence = (
                "[Repair]\n"
                "- Treat stderr, failed assertions, and benchmark diagnostics as hard evidence.\n"
                "- Apply the smallest change that resolves the root cause before attempting broader optimization.\n"
                "- Do not hide failures by returning placeholder metrics."
            )
        else:
            evidence = (
                "[Knowledge]\n"
                "- Improve the current artifact against the stated objective while preserving hard constraints.\n"
                "- Reuse proven parts of the incumbent and make changes that are measurable by the evaluator."
            )
        return QueryResult(
            query_type=self.result_type,
            original_query=query_string,
            evidence=evidence,
            metadata={"source": "built_in_heuristics"},
        )


class MemoryPointerHandler(BaseQueryHandler):
    """Signals that memory is supplied via ContextAssembler snapshots."""

    def execute(self, query_string: str) -> QueryResult:
        return QueryResult(
            query_type=QueryType.MEMORY,
            original_query=query_string,
            evidence=(
                "[Memory]\n"
                "Structured memory is included separately in the context bundle: "
                "operator statistics, recent revision signals, recurring motifs, and the global summary."
            ),
            metadata={"source": "memory_snapshot_pointer"},
        )


class HumanClarificationHandler(BaseQueryHandler):
    """Default non-blocking handler for high-cost human queries."""

    def execute(self, query_string: str) -> QueryResult:
        return QueryResult(
            query_type=QueryType.HUMAN,
            original_query=query_string,
            evidence=(
                "[Human Query]\n"
                "No blocking clarification was requested in this automated run. "
                "Proceed using the task description, constraints, and evaluator feedback."
            ),
            metadata={"source": "non_blocking_default"},
        )

# ==========================================
# 3. Query Engine Implementation
# ==========================================

class QueryEngine:
    """
    Query Engine that routes queries to registered handlers.
    Acts as a facade/dispatcher for various information retrieval subsystems.
    """
    
    def __init__(self, register_defaults: bool = True):
        # Registry mapping query types to their specific handlers
        self._handlers: Dict[QueryType, BaseQueryHandler] = {}
        if register_defaults:
            self.register_handler(QueryType.ENVIRONMENT, EnvironmentSnapshotHandler())
            self.register_handler(QueryType.KNOWLEDGE, KnowledgeHeuristicHandler())
            self.register_handler(QueryType.ARTIFACT, KnowledgeHeuristicHandler(QueryType.ARTIFACT))
            self.register_handler(QueryType.MEMORY, MemoryPointerHandler())
            self.register_handler(QueryType.HUMAN, HumanClarificationHandler())

    def register_handler(self, query_type: QueryType, handler: BaseQueryHandler) -> None:
        """Registers a concrete handler for a specific query type."""
        self._handlers[query_type] = handler

    def _classify_query(self, query_string: str) -> QueryType:
        """
        MVP Router: Uses simple keyword matching to classify the query.
        In a production system, this could be an LLM router or explicit typing.
        """
        query_lower = query_string.lower()
        if any(kw in query_lower for kw in ["env", "api", "schema", "dependency"]):
            return QueryType.ENVIRONMENT
        elif any(kw in query_lower for kw in ["history", "past", "strategy", "memory"]):
            return QueryType.MEMORY
        elif any(kw in query_lower for kw in [
            "diff", "structure", "hotspot", "artifact", "trace", "diagnostic", "failed case"
        ]):
            return QueryType.ARTIFACT
        elif any(kw in query_lower for kw in ["doc", "paper", "spec", "search"]):
            return QueryType.KNOWLEDGE
        elif any(kw in query_lower for kw in ["ask user", "clarify", "human"]):
            return QueryType.HUMAN
        else:
            # Default to Knowledge query if uncertain
            return QueryType.KNOWLEDGE

    def _execute_single_query(self, query_string: str, q_type: QueryType) -> QueryResult:
        """
        Routes the query to the appropriate registered handler.
        """
        handler = self._handlers.get(q_type)
        if not handler:
            # Fallback if no handler is registered for the requested type. This should
            # be rare because the MVP engine installs safe defaults.
            return QueryResult(
                query_type=q_type,
                original_query=query_string,
                evidence=(
                    f"[Unavailable Query Handler] No handler registered for {q_type.name}. "
                    "Proceed with archive and memory context."
                ),
                metadata={"error": "missing_handler"}
            )
            
        # Execute the real implementation
        try:
            return handler.execute(query_string)
        except Exception as e:
            return QueryResult(
                query_type=q_type,
                original_query=query_string,
                evidence=f"[Error] Handler execution failed: {str(e)}",
                metadata={"error": "execution_failure"}
            )

    def run(self, query_plan: List[str]) -> List[QueryResult]:
        """
        Main entry point for the Query Engine.
        Executes the query plan provided by the Optimization Controller.
        """
        results = []
        for query_str in query_plan:
            # 1. Classify
            q_type = self._classify_query(query_str)
            
            # 2. Execute (Fetch evidence)
            result = self._execute_single_query(query_str, q_type)
            
            # 3. Collect
            results.append(result)
            
        return results

# ==========================================
# 4. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    # --- Define some mock implementations of the interfaces for testing ---
    class KnowledgeDatabaseHandler(BaseQueryHandler):
        def execute(self, query_string: str) -> QueryResult:
            # In reality, this connects to a Vector DB or RAG pipeline
            return QueryResult(
                query_type=QueryType.KNOWLEDGE,
                original_query=query_string,
                evidence=f"Retrieved from docs matching '{query_string}': Use asyncio.wait_for.",
                metadata={"source": "vector_db", "confidence": 0.95}
            )

    class EnvironmentInspectorHandler(BaseQueryHandler):
        def execute(self, query_string: str) -> QueryResult:
            # In reality, this runs shell commands or inspects sys.modules
            return QueryResult(
                query_type=QueryType.ENVIRONMENT,
                original_query=query_string,
                evidence="Python 3.10. SQLAlchemy 2.0 detected in requirements.txt.",
                metadata={"source": "local_filesystem"}
            )

    # --- Initialize Engine and Register Handlers ---
    engine = QueryEngine()
    engine.register_handler(QueryType.KNOWLEDGE, KnowledgeDatabaseHandler())
    engine.register_handler(QueryType.ENVIRONMENT, EnvironmentInspectorHandler())
    # Note: We intentionally leave some un-registered to test the fallback mechanism

    print("Testing Query Engine Execution with Interfaces")
    
    # Simulating a query plan generated by the Optimization Controller
    sample_query_plan = [
        "Search documentation/fixes for: TimeoutError during API call",
        "Check current environment schema for database connections",
        "What strategies worked in past history for reducing latency?" # Unregistered!
    ]
    
    print("\nExecuting Query Plan:")
    for q in sample_query_plan:
        print(f"  - {q}")
        
    print("\n" + "="*50 + "\n")
    
    # Run the engine
    retrieved_evidence = engine.run(sample_query_plan)
    
    # Display results
    for idx, res in enumerate(retrieved_evidence):
        print(f"Result [{idx + 1}] - Type: {res.query_type.name}")
        print(f"Original: {res.original_query}")
        print(f"Evidence: {res.evidence}")
        if res.metadata:
            print(f"Metadata: {res.metadata}")
        print("-" * 30)
