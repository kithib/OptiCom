import re
import uuid
import json
from dataclasses import dataclass, field
from typing import List, Any, Dict, Optional
from abc import ABC, abstractmethod

# LLM 客户端抽象层统一下沉到 llm_clients 模块。
from llm_clients import BaseLLMClient, MockLLMClient


# ==========================================
# 1. Output Data Structure
# ==========================================

@dataclass
class GeneratedArtifact:
    """
    Represents a newly generated artifact from the LLM.
    It contains the raw generated content and tracing metadata.
    """
    artifact_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

# ==========================================
# 2. LLM Client Interface (see llm_clients.py for full abstraction)
# ==========================================
# NOTE: 具体后端实现位于 llm_clients.py。

# ==========================================
# 3. Artifact Synthesizer Implementation
# ==========================================

class ArtifactSynthesizer:
    """
    Module ⑦: Artifact Synthesizer.
    A thin wrapper that combines the Operator instruction and Context Bundle,
    calls the LLM, and outputs raw GeneratedArtifacts.
    """
    
    def __init__(self, llm_client: BaseLLMClient):
        # Inject the real LLM client
        self.llm_client = llm_client

    def generate(self, operator: Any, context: Any, n: int = 1) -> List[GeneratedArtifact]:
        """
        The core thin-layer generation method.
        
        Args:
            operator: The selected Operator (contains instruction_template).
            context: The ContextBundle (contains formatted_prompt).
            n: branch_width (how many candidates to generate).
            
        Returns:
            List of GeneratedArtifact instances.
        """
        # 1. Extract the meta-instruction and the context
        system_instruction = operator.instruction_template
        user_prompt = context.formatted_prompt
        
        # 2. Call the underlying LLM (This is where the magic/black-box happens)
        raw_responses = self.llm_client.generate(
            prompt=user_prompt,
            system_instruction=system_instruction,
            n=n
        )
        
        # 3. Wrap the raw text into tracked Artifact objects
        generated_artifacts = []
        # 若算子是 diff_based_rewrite, 尝试把 diff 应用到未压缩的 incumbent 原文。
        # 如果 LLM 确实返回了 unified diff 但应用失败, 标记为不可执行产物;
        # 下游 harness 会直接拒绝它, 避免把 raw diff 当成 Python/C++ 代码运行。
        is_diff_op = getattr(operator, "name", "") == "diff_based_rewrite"
        incumbent_code = ""
        parent_id = None
        ctx_meta = getattr(context, "metadata", {}) or {}
        parent_id = ctx_meta.get("incumbent_artifact_id")
        if is_diff_op:
            incumbent_code = ctx_meta.get("incumbent_code") or ""

        for response in raw_responses:
            content = response
            applied_from_diff = False
            diff_apply_error = ""
            if is_diff_op and incumbent_code:
                patched = _apply_unified_diff(incumbent_code, response)
                if patched is not None and patched != incumbent_code:
                    content = patched
                    applied_from_diff = True
                elif _looks_like_unified_diff(response):
                    diff_apply_error = (
                        "diff_based_rewrite returned a unified diff, but the patch "
                        "could not be applied to the current incumbent."
                    )

            artifact = GeneratedArtifact(
                artifact_id=f"art_{uuid.uuid4().hex[:8]}",
                content=content,
                metadata={
                    "parent_operator": operator.name,
                    "parent_id": parent_id or "root",
                    "branch_width": n,
                    "applied_from_diff": applied_from_diff,
                    "diff_apply_error": diff_apply_error,
                },
            )
            generated_artifacts.append(artifact)

        return generated_artifacts


def _looks_like_unified_diff(text: str) -> bool:
    """Heuristic guard for LLM patch output."""
    if not text:
        return False
    stripped = text.strip()
    stripped = re.sub(r"^```(?:diff|patch)?\s*\n", "", stripped)
    stripped = re.sub(r"\n```\s*$", "", stripped)
    lines = stripped.splitlines()
    has_hunk = any(line.startswith("@@") for line in lines)
    has_headers = (
        any(line.startswith("--- ") for line in lines)
        and any(line.startswith("+++ ") for line in lines)
    )
    has_git_header = any(line.startswith("diff --git ") for line in lines)
    return has_hunk and (has_headers or has_git_header)


def _apply_unified_diff(source_code: str, diff_text: str) -> Optional[str]:
    """
    Minimal unified-diff applier (single-file, forward-only).

    设计要点:
    - 只支持 `--- a/xxx` / `+++ b/xxx` 开头 + 标准 `@@` hunk 的简单 diff。
    - 允许起点偏差: 用 hunk 里"上下文+删除"行在 source 中按子串匹配定位,
      容忍 LLM 输出的 start-line 号偏移几行。
    - 匹配失败立刻返回 None, 调用方回退到原文。
    """
    if not diff_text or "@@" not in diff_text:
        return None

    # 去掉 markdown 残留 (尽量兼容)
    text = diff_text.strip()
    text = re.sub(r"^```(?:diff|patch)?\s*\n", "", text)
    text = re.sub(r"\n```\s*$", "", text)

    lines = text.splitlines()
    # 找第一个 hunk header
    idx = 0
    while idx < len(lines) and not lines[idx].startswith("@@"):
        idx += 1
    if idx >= len(lines):
        return None

    hunks = []  # list of list-of-lines (without the @@ header itself)
    current: List[str] = []
    for line in lines[idx:]:
        if line.startswith("@@"):
            if current:
                hunks.append(current)
            current = []
        else:
            current.append(line)
    if current:
        hunks.append(current)
    if not hunks:
        return None

    source_lines = source_code.splitlines(keepends=False)

    for hunk in hunks:
        # 构造"原文侧"序列 (上下文 + 删除) 与 "新文侧" 序列 (上下文 + 新增)
        old_block: List[str] = []
        new_block: List[str] = []
        for raw in hunk:
            if not raw:
                # 空行当作上下文空行
                old_block.append("")
                new_block.append("")
                continue
            tag, body = raw[0], raw[1:]
            if tag == " ":
                old_block.append(body)
                new_block.append(body)
            elif tag == "-":
                old_block.append(body)
            elif tag == "+":
                new_block.append(body)
            elif tag == "\\":
                # "\ No newline at end of file" —— 忽略
                continue
            else:
                # 非预期前缀, 保守失败
                return None

        if not old_block:
            # 纯新增 hunk: 放到文件末尾
            source_lines.extend(new_block)
            continue

        found_at = _find_block(source_lines, old_block)
        if found_at >= 0:
            source_lines = (
                source_lines[:found_at]
                + new_block
                + source_lines[found_at + len(old_block):]
            )
            continue

        span = _single_change_span(hunk)
        if span is None:
            return None
        removed_lines, added_lines = span
        found_at = _find_block(source_lines, removed_lines)
        if found_at < 0:
            return None
        source_lines = (
            source_lines[:found_at]
            + added_lines
            + source_lines[found_at + len(removed_lines):]
        )

    return "\n".join(source_lines) + ("\n" if source_code.endswith("\n") else "")


def _find_block(lines: List[str], needle: List[str]) -> int:
    """Find a block exactly first, then with trailing-whitespace tolerance."""
    if not needle:
        return -1
    limit = len(lines) - len(needle) + 1
    for start in range(max(0, limit)):
        if lines[start:start + len(needle)] == needle:
            return start
    stripped_needle = [line.rstrip() for line in needle]
    for start in range(max(0, limit)):
        if [line.rstrip() for line in lines[start:start + len(needle)]] == stripped_needle:
            return start
    return -1


def _single_change_span(hunk: List[str]) -> Optional[tuple[List[str], List[str]]]:
    """
    Extract one contiguous +/- edit span from a hunk.

    This is a conservative fallback for LLM diffs whose context lines drifted but
    whose actual deleted line(s) still uniquely identify the intended edit.
    """
    spans: List[tuple[List[str], List[str]]] = []
    removed: List[str] = []
    added: List[str] = []
    in_change = False

    def flush() -> None:
        nonlocal removed, added, in_change
        if in_change:
            spans.append((removed, added))
            removed = []
            added = []
            in_change = False

    for raw in hunk:
        if not raw:
            flush()
            continue
        tag, body = raw[0], raw[1:]
        if tag == "-":
            in_change = True
            removed.append(body)
        elif tag == "+":
            in_change = True
            added.append(body)
        elif tag == " ":
            flush()
        elif tag == "\\":
            continue
        else:
            return None
    flush()

    if len(spans) != 1:
        return None
    removed_lines, added_lines = spans[0]
    if not removed_lines or not added_lines:
        return None
    return removed_lines, added_lines

# ==========================================
# 4. MVP Test Scenarios
# ==========================================
if __name__ == "__main__":
    # To test this, we mock the inputs that would normally come from
    # the Operator Library and Context Assembler.
    class MockOperator:
        name = "recombination"
        instruction_template = "Analyze the CURRENT ARTIFACT alongside the SELECTED EXEMPLARS. Extract strengths and cross-pollinate."
        
    class MockContextBundle:
        formatted_prompt = "### 1. OBJECTIVE\nOptimize speed...\n### 3. CURRENT ARTIFACT\ndef slow_func(): pass"
        
    client = MockLLMClient(canned_response="def optimized():\n    return True")
    synthesizer = ArtifactSynthesizer(llm_client=client)

    print("Testing Artifact Synthesizer with a local mock client")
    print("Simulating Controller requesting branch_width = 3...\n")

    op = MockOperator()
    ctx = MockContextBundle()
    branch_width = 3

    artifacts = synthesizer.generate(operator=op, context=ctx, n=branch_width)
    for art in artifacts:
        print(f"[{art.artifact_id}] Metadata: {art.metadata}")
        print(art.content)
        print("-" * 50)
