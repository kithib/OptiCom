import json
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

# ==========================================
# 1. Output Structure Definition
# ==========================================

@dataclass
class ContextBundle:
    """
    The final assembled package ready to be sent to the LLM (Synthesizer).
    Contains the raw formatted prompt string and metadata.
    """
    formatted_prompt: str
    estimated_tokens: int
    metadata: Dict[str, Any]

# ==========================================
# 2. Context Assembler Implementation
# ==========================================

class ContextAssembler:
    """
    Module ⑤: Context Assembler.
    Transforms raw information into an LLM-usable Context Bundle using a 4-step pipeline:
    Select -> Compress -> Order -> Frame.
    """
    
    def __init__(self, max_context_length: int = 8000, artifact_max_chars: int = 4000, evidence_max_chars: int = 1500):
        """
        Args:
            max_context_length: 估算的总上下文 token 上限 (仅用于 metadata.truncated 标记)。
            artifact_max_chars: 传给 LLM 的 CURRENT BEST ARTIFACT 压缩上限 (字符数)。
                之前默认 2000, 对数值优化任务 (>180 行代码) 会被拦腰砍半,
                导致 LLM 看不到 loss_function / 起点生成等关键细节, 重写时丢信息。
                默认调大到 4000, 可通过 yaml `context.artifact_max_chars` 覆盖。
            evidence_max_chars: 检索证据块的压缩上限 (字符数), 默认 1500。
        """
        self.max_context_length = max_context_length
        self.artifact_max_chars = max(512, int(artifact_max_chars))
        self.evidence_max_chars = max(256, int(evidence_max_chars))

    def build(self, 
              state: Any,              # Optimization State
              query_results: List[Any], # Results from Query Engine
              archive_snapshot: Dict,   # Summary of past explored candidates
              memory_snapshot: Dict,    # Learned heuristics and rules
              action_package: Any       # Instructions from Controller
             ) -> ContextBundle:
        """Main entry point representing the 4-step assembly pipeline."""
        
        # Step 1: SELECT - Filter out noise and pick most relevant pieces
        selected_evidence = self._select(query_results, memory_snapshot)
        selected_failures = state.get("unresolved_errors", [])[:3] # Keep top 3 errors
        selected_exemplars = archive_snapshot.get("top_k_exemplars", [])
        
        # 关键修复: 提取 incumbent 的 best_score 与 scoring history, 真正传给 LLM 指导爬山
        best_score = archive_snapshot.get("best_score")
        best_artifact_id = archive_snapshot.get("best_artifact_id")
        best_metrics = archive_snapshot.get("best_metrics") or {}
        top_metric_summaries = archive_snapshot.get("top_metric_summaries") or []
        recent_failures_from_archive = archive_snapshot.get("recent_rejection_diagnostics", [])
        
        # Step 2: COMPRESS - Truncate or summarize to fit context window
        compressed_artifact = self._compress_text(state.get("current_artifact", "None"), max_chars=self.artifact_max_chars)
        compressed_evidence = self._compress_text("\n".join(selected_evidence), max_chars=self.evidence_max_chars)
        
        # Step 3 & 4: ORDER & FRAME - Arrange blocks logically and wrap in XML/Markdown
        formatted_prompt = self._order_and_frame(
            objective=state.get("objective", "Unknown Objective"),
            constraints=state.get("active_constraints", []),
            artifact=compressed_artifact,
            evidence=compressed_evidence,
            failures=selected_failures + recent_failures_from_archive[:2],
            exemplars=selected_exemplars,
            operator_instruction={
                "operator_choice": action_package.get("operator_choice", "standard_generation"),
                "instruction_template": action_package.get("operator_instruction", ""),
            },
            baseline_profile=action_package.get("baseline_profile"),
            best_score=best_score,
            best_artifact_id=best_artifact_id,
            best_metrics=best_metrics,
            top_metric_summaries=top_metric_summaries,
        )
        
        # Estimate tokens (simple mock: 1 token ~= 4 characters)
        estimated_tokens = len(formatted_prompt) // 4

        # 关键: 把未压缩的 incumbent 原文塞到 metadata 里, 供 diff_based_rewrite
        # 这类需要完整原文做 patch 的 operator 下游取用。
        raw_incumbent = state.get("current_artifact") or ""

        return ContextBundle(
            formatted_prompt=formatted_prompt,
            estimated_tokens=estimated_tokens,
            metadata={
                "truncated": estimated_tokens > self.max_context_length,
                "incumbent_code": raw_incumbent,
                "incumbent_artifact_id": best_artifact_id,
            },
        )

    # --- Pipeline Internal Methods ---

    def _select(self, query_results: List[Any], memory_snapshot: Dict) -> List[str]:
        """Step 1: Selects the most critical information based on current context scope."""
        evidence = []
        # Extract evidence from query results
        for res in query_results:
            if isinstance(res, dict) and "evidence" in res:
                evidence.append(res["evidence"])
            elif hasattr(res, "evidence"):
                evidence.append(res.evidence)
                
        # Add high-priority learned rules from memory
        if "critical_rules" in memory_snapshot:
            evidence.extend(memory_snapshot["critical_rules"])
        global_summary = memory_snapshot.get("global_llm_summary")
        if global_summary and global_summary != "No summary generated yet.":
            evidence.append(f"[Memory Summary]\n{global_summary}")
        recent_signals = memory_snapshot.get("recent_revision_signals") or []
        if recent_signals:
            evidence.append(
                "[Recent Revision Signals]\n"
                + "\n".join(f"- {signal}" for signal in recent_signals[:3])
            )
        recent_metrics = memory_snapshot.get("recent_metric_observations") or []
        if recent_metrics:
            evidence.append(
                "[Recent Metric Observations]\n"
                + "\n".join(f"- {metric}" for metric in recent_metrics[:5])
            )
        operator_perf = memory_snapshot.get("operator_performance") or {}
        if operator_perf:
            evidence.append(
                "[Operator Performance]\n"
                + "\n".join(f"- {op}: {summary}" for op, summary in operator_perf.items())
            )
            
        return evidence

    def _compress(self, text: str, max_chars: int) -> str:
        """Alias for _compress_text for semantic alignment with the 4-step flow."""
        return self._compress_text(text, max_chars)

    def _compress_text(self, text: str, max_chars: int) -> str:
        """Step 2: Compresses long strings to prevent context window overflow."""
        if not text:
            return "None"
        if len(text) <= max_chars:
            return text
        # Keep the beginning and the end, truncate the middle
        half = max_chars // 2
        return text[:half] + "\n\n... [CONTENT TRUNCATED FOR LENGTH] ...\n\n" + text[-half:]

    def _order_and_frame(self, 
                         objective: str, 
                         constraints: List[str], 
                         artifact: str, 
                         evidence: str, 
                         failures: List[str], 
                         exemplars: List[str], 
                         operator_instruction: str,
                         baseline_profile: Optional[Dict[str, Any]] = None,
                         best_score: Optional[float] = None,
                         best_artifact_id: Optional[str] = None,
                         best_metrics: Optional[Dict[str, Any]] = None,
                         top_metric_summaries: Optional[List[Dict[str, Any]]] = None) -> str:
        """Step 3 & 4: Orders the blocks logically and frames them with Markdown headers."""
        
        blocks = []
        
        # 1. Objective Block
        blocks.append("### 1. OBJECTIVE\n" + objective)

        if baseline_profile:
            operators = ", ".join(baseline_profile.get("operators", [])) or "none"
            blocks.append(
                "### 1.5 BASELINE CONFIGURATION\n"
                f"- System: {baseline_profile.get('display_name', baseline_profile.get('name', 'unknown'))}\n"
                f"- Artifact space: {baseline_profile.get('artifact_space', 'unknown')}\n"
                f"- Query policy: {baseline_profile.get('query_policy', 'unknown')}\n"
                f"- Operators: {operators}\n"
                f"- Evaluation protocol: {baseline_profile.get('evaluation_protocol', 'unknown')}\n"
                f"- Memory policy: {baseline_profile.get('memory_policy', 'unknown')}\n"
                f"- Strategy policy: {baseline_profile.get('strategy_policy', 'unknown')}"
            )

        # 2. Constraints Block
        constraints_str = "\n".join([f"- {c}" for c in constraints]) if constraints else "- None"
        blocks.append("### 2. CONSTRAINTS\n" + constraints_str)
        
        # 关键修复: 2.5 块 - 当前最优分数 (让 LLM 明确知道要超越的目标)
        if best_score is not None and best_score != -float('inf'):
            blocks.append(
                "### 2.5 CURRENT BEST SCORE (Incumbent to beat)\n"
                f"- Best artifact id: {best_artifact_id}\n"
                f"- Best objective score so far: {best_score:.6f}\n"
                "- Your new candidate MUST aim for a STRICTLY HIGHER score than this. "
                "Do NOT regress: copy the strengths of the incumbent and only change what can plausibly improve the score."
            )

        metric_lines = self._format_metric_breakdown(
            best_metrics or {},
            top_metric_summaries or [],
        )
        if metric_lines:
            blocks.append(
                "### 2.6 EVALUATOR METRIC BREAKDOWN\n"
                + metric_lines
                + "\nUse these native evaluator metrics to infer trade-offs. "
                "Optimize the primary score as implemented by the evaluator, not a proxy metric."
            )
        
        # 3. Current Artifact
        blocks.append("### 3. CURRENT BEST ARTIFACT (modify and improve this)\n```\n" + artifact + "\n```")
        
        # 4. Critical Evidence (from Query Engine & Memory)
        blocks.append("### 4. CRITICAL EVIDENCE\n" + (evidence if evidence else "None retrieved."))
        
        # 5. Recent Failures
        failures_str = "\n".join([f"- {f}" for f in failures]) if failures else "- No recent failures."
        blocks.append("### 5. RECENT FAILURES & ERRORS (avoid repeating these)\n" + failures_str)
        
        # 6. Selected Exemplars (from Archive)
        if exemplars and len(exemplars) > 1:
            # 跳过 index 0 (因为 exemplars[0] 已经作为 current artifact 展示过)
            other_exemplars = exemplars[1:3]
            ex_str = "\n---\n".join([f"Exemplar {i+1}:\n{ex}" for i, ex in enumerate(other_exemplars)])
            blocks.append("### 6. OTHER TOP-K EXEMPLARS (alternative good approaches)\n" + ex_str)
        
        # 7. Operator Instruction (from Controller)
        explicit_instruction = ""
        if isinstance(operator_instruction, dict):
            explicit_instruction = operator_instruction.get("instruction_template", "")
            operator_instruction = operator_instruction.get("operator_choice", "standard_generation")
        blocks.append("### 7. OPERATOR INSTRUCTION\n" + 
                      f"Action requested: {operator_instruction}\n" +
                      (f"{explicit_instruction}\n" if explicit_instruction else "") +
                      "Please apply the above instruction to modify the CURRENT ARTIFACT. "
                      "Output ONLY the complete updated artifact in the target format — NO markdown fences, NO prose, NO explanations.")
        
        # Join all blocks with clear separators
        return "\n\n=========================================\n\n".join(blocks)

    @staticmethod
    def _format_metric_breakdown(
        best_metrics: Dict[str, Any],
        top_metric_summaries: List[Dict[str, Any]],
    ) -> str:
        lines: List[str] = []
        if best_metrics:
            metric_str = ", ".join(
                f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                for k, v in best_metrics.items()
            )
            lines.append(f"- Current best metrics: {metric_str}")
        if top_metric_summaries:
            lines.append("- Top frontier metrics:")
            for item in top_metric_summaries[:4]:
                metrics = item.get("metrics") or {}
                metric_str = ", ".join(
                    f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
                    for k, v in metrics.items()
                )
                status = "passed" if item.get("is_passed") else "not_passed"
                lines.append(f"  - {item.get('artifact_id')}: {status}; {metric_str}")
        return "\n".join(lines)

# ==========================================
# 3. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    assembler = ContextAssembler()

    # MOCK INPUTS (Simulating data from other modules)
    mock_state = {
        "objective": "Optimize the sorting function for maximum speed on nearly-sorted arrays.",
        "active_constraints": ["Must maintain O(1) auxiliary space", "Must not use external libraries"],
        "current_artifact": "def sort_arr(arr):\n    return sorted(arr)  # Slow for this specific edge case",
        "unresolved_errors": ["MemoryError: auxiliary space exceeded in previous attempt"]
    }
    
    mock_query_results = [
        {"evidence": "[Knowledge] Insertion sort performs optimally O(N) on nearly-sorted arrays."}
    ]
    
    mock_archive = {
        "top_k_exemplars": ["def insertion_sort(arr):\n    # standard implementation..."]
    }
    
    mock_memory = {
        "critical_rules": ["[Memory] Never use recursive QuickSort here, it triggers RecursionError."]
    }
    
    mock_action_package = {
        "operator_choice": "Apply 'diverse_crossover' using the logic from the exemplar while keeping space complexity strictly O(1)."
    }

    print("Building Context Bundle...")
    
    # Run the assembly pipeline
    bundle = assembler.build(
        state=mock_state,
        query_results=mock_query_results,
        archive_snapshot=mock_archive,
        memory_snapshot=mock_memory,
        action_package=mock_action_package
    )

    print("\n" + "*"*60)
    print("FINAL LLM PROMPT (Context Bundle):")
    print("*"*60 + "\n")
    print(bundle.formatted_prompt)
    print("\n" + "*"*60)
    print(f"Token Estimate: ~{bundle.estimated_tokens} tokens")
