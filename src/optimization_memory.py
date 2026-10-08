import json
import time
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from collections import defaultdict
from enum import Enum

# Import the LLM Client for advanced summarization
from artifact_synthesizer import BaseLLMClient

# ==========================================
# 0. Mock Data Structures (From Upstream)
# ==========================================
# Conceptually imported from controller, query_engine, synthesizer, harness, and evaluation_stack

@dataclass
class ActionPackage:
    operator_choice: str
    query_plan: List[str]
    branch_width: int

@dataclass
class QueryResult:
    query_type: str
    evidence: str

@dataclass
class GeneratedArtifact:
    artifact_id: str
    content: str
    
@dataclass
class ExecutionTrace:
    is_completed: bool
    stdout: str
    stderr: str

@dataclass
class EvaluationRecord:
    artifact_id: str
    failure_type: str
    diagnostics: str
    revision_signal: str
    is_passed: bool
    metrics: Dict[str, float]

# ==========================================
# 1. Memory Event Data Structures
# ==========================================

@dataclass
class MemoryEvent:
    """
    Represents a single, immutable log entry in the Optimization Memory.
    Captures the full context of a single artifact generation branch.
    Matches the 4 core histories from the architecture diagram.
    """
    event_id: int
    timestamp: float
    
    # 1. Action History
    operator: str
    query_plan: List[str]
    
    # 2. Observation History
    retrieved_evidence: List[str]
    trace_stdout: str
    trace_stderr: str
    
    # 3. Artifact History
    artifact_id: str
    content: str
    # A simple diff could be stored here if comparing to a parent artifact
    content_diff_summary: str 
    
    # 4. Evaluation History
    failure_type: str
    diagnostics: str
    revision_signal: str
    is_passed: bool
    metrics: Dict[str, float]

@dataclass
class OperatorStats:
    """Tracks the historical performance of an operator."""
    attempts: int = 0
    successes: int = 0
    total_score: float = 0.0
    
    @property
    def success_rate(self) -> float:
        return self.successes / self.attempts if self.attempts > 0 else 0.0
        
    @property
    def avg_score(self) -> float:
        return self.total_score / self.attempts if self.attempts > 0 else 0.0

# ==========================================
# 2. Optimization Memory Implementation
# ==========================================

class OptimizationMemory:
    """
    Module ⑪: Optimization Memory.
    Uses an append-only Event Log combined with Structured Indexes to avoid Vector DBs.
    Maintains operator statistics and distills higher-level heuristics.
    Now includes an LLM-driven summarization mechanism for long contexts.
    """
    
    def __init__(self, llm_client: Optional[BaseLLMClient] = None, summarize_every_n_events: int = 5):
        # --- LLM Summarization Setup ---
        self.llm_client = llm_client
        self.summarize_every_n = summarize_every_n_events
        self.last_summarized_index = 0
        self.global_llm_summary = "No summary generated yet."
        
        # --- Core Storage: The Append-Only Event Log ---
        self.event_log: List[MemoryEvent] = []
        
        # --- Structured Indexes (Pointers to Event IDs) ---
        # 1. Index by Operator (e.g., "local_revision" -> [0, 3, 4])
        self.operator_index: Dict[str, List[int]] = defaultdict(list)
        # 2. Index by Failure Type (e.g., "correctness_error" -> [1, 2])
        self.failure_index: Dict[str, List[int]] = defaultdict(list)
        # 3. Index by Artifact ID
        self.artifact_index: Dict[str, int] = {}
        
        # --- 5. Operator Statistics ---
        self.operator_stats: Dict[str, OperatorStats] = defaultdict(OperatorStats)
        
        # --- Advanced Layer (Distilled Knowledge) ---
        self.distilled_heuristics: List[str] = []
        self.reusable_templates: List[str] = []
        self.common_failure_motifs: Dict[str, int] = defaultdict(int) # motif -> count

    @staticmethod
    def _value_text(value: Any) -> str:
        """Normalize Enums and arbitrary objects into stable string keys."""
        if isinstance(value, Enum):
            return str(value.value)
        return str(value)

    @staticmethod
    def _attr(obj: Any, name: str, default: Any = None) -> Any:
        if isinstance(obj, dict):
            return obj.get(name, default)
        return getattr(obj, name, default)

    def write(self, 
              action: ActionPackage, 
              evidence: List[QueryResult], 
              artifacts: List[GeneratedArtifact], 
              traces: List[ExecutionTrace], 
              records: List[EvaluationRecord]) -> None:
        """
        Writes a batch of execution results into the memory log and updates all indexes.
        """
        evidence_texts = [self._attr(e, "evidence", str(e)) for e in evidence]
        operator_choice = self._attr(action, "operator_choice", "unknown_operator")
        query_plan = list(self._attr(action, "query_plan", []) or [])
        
        for art, trace, rec in zip(artifacts, traces, records):
            event_id = len(self.event_log)
            failure_type = self._value_text(self._attr(rec, "failure_type", "unknown"))
            metrics = self._attr(rec, "metrics", {}) or {}
            
            # 1. Create the structured Memory Event
            event = MemoryEvent(
                event_id=event_id,
                timestamp=time.time(),
                operator=operator_choice,
                query_plan=query_plan,
                retrieved_evidence=evidence_texts,
                trace_stdout=self._attr(trace, "stdout", ""),
                trace_stderr=self._attr(trace, "stderr", ""),
                artifact_id=self._attr(art, "artifact_id", ""),
                content=self._attr(art, "content", ""),
                content_diff_summary="Diff extraction simulated", # In a real system, compute difflib here
                failure_type=failure_type,
                diagnostics=self._attr(rec, "diagnostics", ""),
                revision_signal=self._attr(rec, "revision_signal", ""),
                is_passed=bool(self._attr(rec, "is_passed", False)),
                metrics=metrics
            )
            
            # 2. Append to Event Log
            self.event_log.append(event)
            
            # 3. Update Indexes
            self.operator_index[operator_choice].append(event_id)
            self.failure_index[failure_type].append(event_id)
            self.artifact_index[event.artifact_id] = event_id
            
            # 4. Update Operator Statistics
            stats = self.operator_stats[operator_choice]
            stats.attempts += 1
            if event.is_passed:
                stats.successes += 1
            score = metrics.get("score", 0.0)
            if isinstance(score, (int, float)) and score == score and score not in (float("inf"), -float("inf")):
                stats.total_score += float(score)
            
            # 5. Extract Meta-Insights (Advanced Layer)
            self._extract_insights_from_event(event)
            
        # 6. Trigger LLM Summarization if the threshold is reached
        self._trigger_llm_summarization_if_needed()

    def _extract_insights_from_event(self, event: MemoryEvent) -> None:
        """
        A lightweight rule-based distillation engine.
        In a full implementation, this could periodically call an LLM to summarize patterns.
        """
        if not event.is_passed:
            # Simple heuristic: If the same diagnostic occurs multiple times, it becomes a common motif
            error_signature = str(event.diagnostics)[:50] + "..." # Truncate for matching
            self.common_failure_motifs[error_signature] += 1
            
            if self.common_failure_motifs[error_signature] == 3:
                # Discovered a recurring failure motif!
                motif_warning = f"AVOID: The system repeatedly fails with '{error_signature}'."
                if motif_warning not in self.distilled_heuristics:
                    self.distilled_heuristics.append(motif_warning)

    def _trigger_llm_summarization_if_needed(self) -> None:
        """
        Checks if there are enough new events to trigger an LLM-based summarization
        to prevent the context from growing infinitely.
        """
        if not self.llm_client:
            return # Skip if no LLM client is configured
            
        unsummarized_count = len(self.event_log) - self.last_summarized_index
        if unsummarized_count >= self.summarize_every_n:
            print("\n[ LLM triggered to summarize Memory History...]")
            # 1. Gather unsummarized events
            recent_events = self.event_log[self.last_summarized_index:]
            event_texts = []
            for ev in recent_events:
                status = "SUCCESS" if ev.is_passed else f"FAILED ({ev.failure_type})"
                metric_text = self._format_metrics_compact(ev.metrics)
                event_texts.append(
                    f"Event {ev.event_id}: Operator={ev.operator}, Status={status}, "
                    f"Metrics={metric_text}, Signal={ev.revision_signal}"
                )
            
            recent_history_text = "\n".join(event_texts)
            
            # 2. Call LLM to update the global summary
            sys_instruction = (
                "You are an AI optimization historian managing the 'Optimization Memory'. "
                "Your task is to compress the history of exploration into a dense, actionable summary. "
                "Analyze the new events and integrate their lessons into the existing global summary. "
                "Focus on identifying winning strategies, dead-end approaches to avoid, and overall progress."
            )
            
            user_prompt = (
                f"--- Current Global Summary ---\n{self.global_llm_summary}\n\n"
                f"--- New Recent Events ---\n{recent_history_text}\n\n"
                "Please provide an updated, concise Global Summary (keep it under 200 words):"
            )
            
            try:
                responses = self.llm_client.generate(user_prompt, sys_instruction, n=1)
                self.global_llm_summary = responses[0].strip()
                self.last_summarized_index = len(self.event_log)
                print("[LLM Summary successfully updated]")
            except Exception as e:
                print(f"[Memory Summarization Error]: {e}")

    def retrieve_by_failure_type(self, failure_type: str, limit: int = 3) -> List[MemoryEvent]:
        """Query the log using the structured index."""
        event_ids = self.failure_index.get(failure_type, [])[-limit:]
        return [self.event_log[eid] for eid in event_ids]

    def snapshot(self) -> Dict[str, Any]:
        """
        Generates a summary of learned knowledge for the Context Assembler and Strategy Adapter.
        """
        # Format the operator stats for the prompt
        formatted_stats = {
            op: f"Success Rate: {stat.success_rate*100:.1f}%, Avg Score: {stat.avg_score:.2f}"
            for op, stat in self.operator_stats.items()
        }
        
        return {
            "total_events": len(self.event_log),
            "operator_performance": formatted_stats,
            "critical_rules": self.distilled_heuristics, # These go directly into the Prompt
            "recent_revision_signals": [e.revision_signal for e in self.event_log[-3:] if not e.is_passed],
            "recent_metric_observations": [
                f"event={e.event_id}, operator={e.operator}, passed={e.is_passed}, metrics={self._format_metrics_compact(e.metrics)}"
                for e in self.event_log[-5:]
                if e.metrics
            ],
            "global_llm_summary": self.global_llm_summary # Added LLM summary for Context Assembler
        }

    @staticmethod
    def _format_metrics_compact(metrics: Dict[str, Any], max_items: int = 6) -> str:
        if not metrics:
            return "{}"
        priority = [
            "runtime_score",
            "score",
            "combined_score",
            "success_rate",
            "max_kvpr",
            "execution_time",
        ]
        parts: List[str] = []
        seen = set()
        for key in priority:
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.6g}")
                seen.add(key)
        for key in sorted(metrics):
            if len(parts) >= max_items:
                break
            if key in seen:
                continue
            value = metrics.get(key)
            if isinstance(value, (int, float)):
                parts.append(f"{key}={float(value):.6g}")
        return "{" + ", ".join(parts) + "}"

# ==========================================
# 3. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    print("Initiating Optimization Memory (Event Log + Index + Summarization Mode)...")
    
    llm_client = None

    # Initialize with LLM client and set threshold to 3 events to trigger easily
    memory = OptimizationMemory(llm_client=llm_client, summarize_every_n_events=3)
    
    # --- Mock Data for Iteration 1 ---
    action_1 = ActionPackage("local_revision", ["query syntax"], 1)
    evidence_1 = [QueryResult("knowledge", "Use `try-except` for robustness.")]
    art_1 = GeneratedArtifact("art_001", "def func(): 1/0")
    trace_1 = ExecutionTrace(True, "", "ZeroDivisionError")
    rec_1 = EvaluationRecord("art_001", "feasibility_error", "ZeroDivisionError on line 1", "Add error handling.", False, {"score": 0.0})
    
    memory.write(action_1, evidence_1, [art_1], [trace_1], [rec_1])
    
    # --- Mock Data for Iteration 2 (Simulating repeated error) ---
    art_2 = GeneratedArtifact("art_002", "def func(): x = 1/0")
    rec_2 = EvaluationRecord("art_002", "feasibility_error", "ZeroDivisionError on line 1", "Still crashing.", False, {"score": 0.0})
    memory.write(action_1, evidence_1, [art_2], [trace_1], [rec_2])
    
    # --- Mock Data for Iteration 3 (Triggers Insight Distillation and LLM Summarization) ---
    art_3 = GeneratedArtifact("art_003", "def func(): return 1 / 0")
    rec_3 = EvaluationRecord("art_003", "feasibility_error", "ZeroDivisionError on line 1", "Stop dividing by zero.", False, {"score": 0.0})
    memory.write(action_1, evidence_1, [art_3], [trace_1], [rec_3])
    
    # --- Mock Data for Iteration 4 (Success with different operator) ---
    action_2 = ActionPackage("diverse_crossover", ["query robust patterns"], 1)
    art_4 = GeneratedArtifact("art_004", "def func(): return 0")
    trace_4 = ExecutionTrace(True, "Success", "")
    rec_4 = EvaluationRecord("art_004", "none", "All passed", "Great job.", True, {"score": 0.95})
    memory.write(action_2, [], [art_4], [trace_4], [rec_4])
    
    # --- Output Verification ---
    print("\n--- 1. Structured Index Query ---")
    feasibility_errors = memory.retrieve_by_failure_type("feasibility_error")
    print(f"Found {len(feasibility_errors)} 'feasibility_error' events via O(1) Index lookup.")
    for err in feasibility_errors:
        print(f"  - Event ID {err.event_id}: Operator '{err.operator}', Signal: '{err.revision_signal}'")
        
    print("\n--- 2. Snapshot (Export for Context Assembler) ---")
    snapshot = memory.snapshot()
    print(json.dumps(snapshot, indent=2))
