import json
import math
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# ==========================================
# 0. Mock Data Structures (From Upstream)
# ==========================================
# Imported conceptually from artifact_synthesizer and evaluation_stack

@dataclass
class GeneratedArtifact:
    artifact_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class EvaluationRecord:
    artifact_id: str
    metrics: Dict[str, float]
    failure_type: str
    diagnostics: str
    confidence: float
    revision_signal: str
    is_passed: bool

# ==========================================
# 1. Archive Data Structures
# ==========================================

@dataclass
class ArchivedItem:
    """
    A wrapper that binds the Artifact and its Evaluation Record together.
    Represents a fully processed candidate in the archive.
    """
    artifact: GeneratedArtifact
    record: EvaluationRecord
    parent_id: Optional[str] = None
    
    @property
    def artifact_id(self) -> str:
        return self.artifact.artifact_id

# ==========================================
# 2. Archive Manager Implementation
# ==========================================

class ArchiveManager:
    """
    Module ⑩: Archive Manager.
    Maintains the state of exploration including the incumbent, active frontier, 
    rejection set, and tracks the evolutionary lineage of artifacts.
    """
    
    def __init__(self, top_k: int = 5, objective_metric: str = "score", maximize: bool = True):
        self.top_k = top_k
        self.objective_metric = objective_metric
        self.maximize = maximize
        
        # --- The 3 Core Containers ---
        
        # 1. Best Artifact (The Incumbent)
        self.best_artifact: Optional[ArchivedItem] = None
        
        # 2. Top-K Active Set (Active Frontier / Pareto approximation)
        # Stored as a list sorted by the objective metric
        self.active_set: List[ArchivedItem] = []
        
        # 3. Rejection Archive (Failed artifacts, useful for negative constraints)
        self.rejection_archive: List[ArchivedItem] = []
        
        # --- Tracking ---
        # Lineage Graph: Maps parent_id -> List of child_ids
        self.lineage_graph: Dict[str, List[str]] = {}

    def update(self, artifacts: List[GeneratedArtifact], records: List[EvaluationRecord]) -> None:
        """
        Processes a batch of evaluated artifacts, decides admission, 
        and maintains the internal containers.
        """
        # Map artifacts to records
        record_map = {r.artifact_id: r for r in records}
        
        for artifact in artifacts:
            record = record_map.get(artifact.artifact_id)
            if not record:
                continue # Skip if no evaluation record exists
                
            parent_id = artifact.metadata.get("parent_id", "root")
            item = ArchivedItem(artifact=artifact, record=record, parent_id=parent_id)
            
            # 1. Update Lineage Graph
            if parent_id not in self.lineage_graph:
                self.lineage_graph[parent_id] = []
            self.lineage_graph[parent_id].append(item.artifact_id)
            
            # 2. Decide Admission
            self._decide_admission(item)

    def _decide_admission(self, item: ArchivedItem) -> None:
        """
        Routes the evaluated item into the correct container based on its performance.
        关键修复: 即使没通过所有 gates, 只要有有效的 objective score, 仍然记录到 active_set,
        这样 best_artifact 可以反映真实最优进度, 避免 incumbent 永远是 -inf。
        """
        score = item.record.metrics.get(self.objective_metric, None)
        has_valid_score = isinstance(score, (int, float)) and math.isfinite(float(score))
        
        # Route 1: 完全没有 score 的失败 (feasibility/syntax error) -> Rejection only
        if not has_valid_score:
            self.rejection_archive.append(item)
            return
        
        # Route 2: 有有效 score -> Admit to active set regardless of is_passed
        # (因为 gate3 的 0.7 threshold 对构造类问题可能不合理, 不能因此丢弃高分结果)
        self.active_set.append(item)
        
        # 同时如果没有 passed, 也记录到 rejection archive 供 LLM 参考反馈
        if not item.record.is_passed:
            self.rejection_archive.append(item)
        
        # Sort the active set by objective metric
        self.active_set.sort(
            key=lambda x: x.record.metrics.get(self.objective_metric, -float('inf') if self.maximize else float('inf')),
            reverse=self.maximize
        )
        
        # Trim to top-k
        if len(self.active_set) > self.top_k:
            self.active_set = self.active_set[:self.top_k]
            
        # Update Best Artifact (Incumbent)
        current_best = self.active_set[0]
        if self.best_artifact is None or self.best_artifact.artifact_id != current_best.artifact_id:
            self.best_artifact = current_best

    def get_best_content(self) -> str:
        """Returns the raw content of the current incumbent."""
        return self.best_artifact.artifact.content if self.best_artifact else ""

    def snapshot(self) -> Dict[str, Any]:
        """
        Generates a summary dictionary to be consumed by the Context Assembler.
        Exposes useful exemplars and current frontier status.
        """
        # Extract top exemplars for the 'recombination' operator
        top_k_exemplars = [item.artifact.content for item in self.active_set]
        top_metric_summaries = [
            {
                "artifact_id": item.artifact_id,
                "is_passed": item.record.is_passed,
                "metrics": self._compact_metrics(item.record.metrics),
            }
            for item in self.active_set[: self.top_k]
        ]
        
        # Extract common failure diagnostics from the rejection archive for negative prompts
        recent_failures = [item.record.diagnostics for item in self.rejection_archive[-3:]]
        
        return {
            "best_artifact_id": self.best_artifact.artifact_id if self.best_artifact else None,
            "best_score": self.best_artifact.record.metrics.get(self.objective_metric) if self.best_artifact else None,
            "best_metrics": self._compact_metrics(self.best_artifact.record.metrics) if self.best_artifact else {},
            "active_frontier_size": len(self.active_set),
            "top_k_exemplars": top_k_exemplars,
            "top_metric_summaries": top_metric_summaries,
            "recent_rejection_diagnostics": recent_failures
        }

    @staticmethod
    def _compact_metrics(metrics: Dict[str, Any], max_items: int = 8) -> Dict[str, float]:
        """Keep a stable, compact view of numeric evaluator metrics."""
        compact: Dict[str, float] = {}
        priority = [
            "runtime_score",
            "score",
            "combined_score",
            "success_rate",
            "max_kvpr",
            "execution_time",
            "accuracy",
            "reward",
            "value",
        ]
        for key in priority:
            value = metrics.get(key)
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                compact[key] = float(value)
        for key, value in sorted(metrics.items()):
            if len(compact) >= max_items:
                break
            if key in compact:
                continue
            if isinstance(value, (int, float)) and math.isfinite(float(value)):
                compact[key] = float(value)
        return compact

    def print_lineage(self) -> str:
        """Helper to visualize the evolutionary tree."""
        return json.dumps(self.lineage_graph, indent=2)

# ==========================================
# 3. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    print("🗄️ Initiating Archive Manager...")
    
    # Initialize aiming to maximize 'accuracy'
    manager = ArchiveManager(top_k=2, objective_metric="accuracy", maximize=True)
    
    # --- Simulated Iteration 1 ---
    # Generated from an initial prompt
    art_1 = GeneratedArtifact("art_001", "def fast_func(): return True", {"parent_id": "root", "operator": "initial"})
    rec_1 = EvaluationRecord("art_001", {"accuracy": 0.6}, "none", "Passed.", 1.0, "Good baseline.", True)
    
    art_2 = GeneratedArtifact("art_002", "def fast_func(): return 1/0", {"parent_id": "root", "operator": "initial"})
    rec_2 = EvaluationRecord("art_002", {}, "feasibility_error", "ZeroDivisionError", 1.0, "Fix crash.", False)
    
    manager.update([art_1, art_2], [rec_1, rec_2])
    
    print("\n--- After Iteration 1 ---")
    print(f"Best Artifact ID: {manager.best_artifact.artifact_id} (Score: {manager.best_artifact.record.metrics['accuracy']})")
    print(f"Active Set Size: {len(manager.active_set)}")
    print(f"Rejection Archive Size: {len(manager.rejection_archive)}")
    
    # --- Simulated Iteration 2 ---
    # Controller decides to mutate art_001
    art_3 = GeneratedArtifact("art_003", "def fast_func(): return True # optimized", {"parent_id": "art_001", "operator": "local_revision"})
    rec_3 = EvaluationRecord("art_003", {"accuracy": 0.85}, "none", "Passed.", 1.0, "Great improvement.", True)
    
    # Another branch that failed
    art_4 = GeneratedArtifact("art_004", "def fast_func(): pass", {"parent_id": "art_001", "operator": "local_revision"})
    rec_4 = EvaluationRecord("art_004", {"accuracy": 0.1}, "objective_suboptimal", "Score too low.", 1.0, "Needs better logic.", False)
    
    # A new orthogonal idea
    art_5 = GeneratedArtifact("art_005", "def fast_func(): return 'Valid'", {"parent_id": "root", "operator": "diverse_crossover"})
    rec_5 = EvaluationRecord("art_005", {"accuracy": 0.75}, "none", "Passed.", 1.0, "Solid alternative.", True)
    
    manager.update([art_3, art_4, art_5], [rec_3, rec_4, rec_5])
    
    print("\n--- After Iteration 2 ---")
    print(f"New Best Artifact ID: {manager.best_artifact.artifact_id} (Score: {manager.best_artifact.record.metrics['accuracy']})")
    
    # Since top_k=2, art_001 (0.6) should be evicted from the active set, leaving art_003 (0.85) and art_005 (0.75)
    active_ids = [item.artifact_id for item in manager.active_set]
    print(f"Active Set (Top 2): {active_ids}")
    
    print(f"Total Rejections: {len(manager.rejection_archive)}")
    
    print("\n--- Lineage Graph (Evolution Tree) ---")
    print(manager.print_lineage())
    
    print("\n--- Snapshot for Context Assembler ---")
    snapshot = manager.snapshot()
    print(f"Extracted {len(snapshot['top_k_exemplars'])} Exemplars.")
    print(f"Extracted {len(snapshot['recent_rejection_diagnostics'])} Recent Failure Diagnostics.")
