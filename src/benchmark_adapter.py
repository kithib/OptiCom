"""
benchmark_adapter.py
===================

Benchmark 目录约定:
    <benchmark_root>/
        <category>/<task_name>/
            ├── config.yaml           # 含 prompt.system_message / num_islands / ...
            ├── initial_program.py    # (或 .cpp) LLM 改进起点
            ├── evaluator.py          # 暴露顶级函数 evaluate(program_path) -> dict[str, float]
            └── (optional) evaluator/evaluator.py  # ADRS 系列的子目录结构

对接方案:
    1. `BenchmarkRegistry.discover(root)` 扫描所有 task 目录、返回 BenchmarkTask 列表;
    2. `BenchmarkTask.load()` 读取 config / initial program;
    3. `BenchmarkTaskHarness` 替换 PythonCodeHarness:
       - 把 LLM 生成的 artifact.content 写入临时文件
       - 动态 import 该 benchmark 的 evaluator 并调用 evaluate(path)
       - 解析返回的 metrics dict, 写入 trace.metrics["score"]
       - `score = metrics.get("combined_score") or metrics.get("score") or fallback_metric`
    4. 默认 objective = "combined_score" (maximize)

用法:
    from benchmark_adapter import BenchmarkRegistry, BenchmarkTaskHarness

    reg = BenchmarkRegistry("/mnt/.../skydiscover/benchmarks")
    task = reg.get("circle_packing")
    harness = BenchmarkTaskHarness(task=task, timeout_seconds=120)
    # harness 可直接传入 ExecutionHarness(adapter=harness)
"""

from __future__ import annotations

import os
import sys
import time
import uuid
import pickle
import traceback
import tempfile
import importlib.util
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable

import yaml  # type: ignore


# ==========================================
# 1. BenchmarkTask Data Class
# ==========================================

@dataclass
class BenchmarkTask:
    """描述一个 benchmark 任务的所有元数据。"""

    task_id: str                   # e.g. "circle_packing"
    category: str                  # e.g. "math" / "ADRS" / "algorithm"
    root_dir: Path                 # 任务目录绝对路径
    initial_program_path: Path     # 初始程序文件
    evaluator_module_path: Path    # evaluator.py 路径
    language: str = "python"       # "python" / "cpp" / ...
    config: Dict[str, Any] = field(default_factory=dict)

    # ---- Convenience -----------------------------------------------
    @property
    def full_name(self) -> str:
        return f"{self.category}/{self.task_id}"

    def get_system_message(self) -> str:
        """从 config.yaml 中读取 prompt.system_message 作为 task description。"""
        prompt_cfg = self.config.get("prompt", {}) or {}
        msg = prompt_cfg.get("system_message") or prompt_cfg.get("system") or ""
        return str(msg).strip()

    def get_initial_program_code(self) -> str:
        """读取初始程序代码，作为 LLM 的 'current artifact'。"""
        if not self.initial_program_path.exists():
            return ""
        return self.initial_program_path.read_text(encoding="utf-8", errors="ignore")

    def get_evaluator_scoring_summary(self, max_chars: int = 2600) -> str:
        """
        Extract a compact source excerpt around evaluator scoring logic.

        LLM optimization often fails when the natural-language objective differs
        from the actual leaderboard metric. This keeps the prompt metric-aware
        without dumping the entire evaluator into context.
        """
        if not self.evaluator_module_path.exists():
            return ""
        try:
            lines = self.evaluator_module_path.read_text(
                encoding="utf-8", errors="ignore"
            ).splitlines()
        except OSError:
            return ""

        keywords = (
            "combined_score",
            "score",
            "success_rate",
            "avg_",
            "mean(",
            "max_kvpr",
            "target_ratio",
            "accuracy",
            "reward",
            "validity",
            "successful_runs",
            "all_kvpr",
            "return {",
            "except TimeoutError",
            "continue",
        )
        keep: set[int] = set()
        for idx, line in enumerate(lines):
            lowered = line.lower()
            if any(keyword.lower() in lowered for keyword in keywords):
                for j in range(max(0, idx - 3), min(len(lines), idx + 5)):
                    keep.add(j)

        if not keep:
            return ""

        chunks: List[str] = []
        previous = -2
        for idx in sorted(keep):
            if idx != previous + 1 and chunks:
                chunks.append("...")
            chunks.append(f"{idx + 1:04d}: {lines[idx]}")
            previous = idx

        summary = "\n".join(chunks).strip()
        if len(summary) > max_chars:
            half = max_chars // 2
            summary = (
                summary[:half]
                + "\n... [EVALUATOR SCORING EXCERPT TRUNCATED] ...\n"
                + summary[-half:]
            )
        return summary

    def get_task_description(self) -> str:
        """
        汇总 system_message + 初始程序 + 语言提示，生成一段完整的任务描述，
        喂给我们的 Agent 框架作为 task_description。
        """
        sys_msg = self.get_system_message()
        initial = self.get_initial_program_code()
        pieces: List[str] = []
        if sys_msg:
            pieces.append(f"## Task Description (from benchmark {self.full_name})\n{sys_msg}")
        if initial:
            fence_lang = "text" if self.language.lower() == "text" else (
                "python" if self.language.lower() == "python" else self.language
            )
            pieces.append(
                f"## Initial Artifact (improve this)\n```{fence_lang}\n{initial}\n```"
            )
        scoring_summary = self.get_evaluator_scoring_summary()
        if scoring_summary:
            pieces.append(
                "## Evaluator Scoring Contract (source excerpt)\n"
                "The leaderboard score is determined by this evaluator logic. "
                "Optimize the primary metric exactly as implemented, while preserving the public API.\n"
                f"```python\n{scoring_summary}\n```"
            )
        if self.language.lower() == "text":
            pieces.append(
                "## Output Contract\n"
                "Return only the improved prompt text. Preserve the same input/output interface and required fields. "
                "DO NOT output markdown fences, explanations, code, or extra metadata."
            )
        else:
            pieces.append(
                "## Output Contract\n"
                "Return a single self-contained runnable program that PRESERVES the public API "
                "(function names, signatures) of the initial program. DO NOT output markdown fences, "
                "DO NOT output prose — only valid source code of the target language."
            )
        return "\n\n".join(pieces)


# ==========================================
# 2. BenchmarkRegistry — 目录扫描
# ==========================================

class BenchmarkRegistry:
    """
    扫描 benchmarks/ 目录, 自动发现所有 BenchmarkTask.

    支持的布局（任意深度，最多向下递归 MAX_DEPTH 层）::
        root/arc_benchmark/evaluator/evaluator.py + initial_program.py
            └── 1 层：task 直接躺在 root 下（category 回退为 root basename）
        root/math/circle_packing/evaluator.py + initial_program.py
            └── 2 层：root/<category>/<task>
        root/ale_bench/ale-bench-lite-problems/ahc008/evaluator.py + initial_program.cpp
            └── 3 层：root/<category>/<subcollection>/<task>
        root/image_gen/sky_festival/evaluator.py  (没有 initial_program 也允许)
        root/prompt_optimization/hotpot_qa/evaluator.py + initial_prompt.txt
    """

    # 常见的初始程序文件名（优先级从高到低）
    INITIAL_PROGRAM_NAMES = [
        "initial_program.py",
        "initial_program.cpp",
        "initial_program.cc",
        "initial_program.cxx",
        "initial_program.c",
        "initial_program.rs",
        "initial_program.go",
        "initial_program.java",
        "initial_program.ts",
        "initial_program.js",
        "initial_program.yaml",
        "initial_program.yml",
        "initial_program.json",
        "initial_program.txt",
        "initial_prompt.txt",
        "initial_prompt.md",
        "initial.py",
        "initial.cpp",
    ]

    # evaluator.py 可能出现的位置（相对于 task_dir）
    EVALUATOR_CANDIDATES = [
        Path("evaluator.py"),
        Path("evaluator") / "evaluator.py",
    ]

    # 递归扫描的最大深度（benchmarks/<d1>/<d2>/<d3>）
    MAX_DEPTH = 3

    # 扫描时跳过的目录（噪声）
    SKIP_DIR_NAMES = {
        "__pycache__", ".git", ".ipynb_checkpoints",
        "evaluator",          # 已在候选路径里单独处理
        "codebase",           # math/circle_packing/codebase 等参考资料
        "ale_agent_best",     # ale_bench 的参考最佳解
    }

    def __init__(self, benchmarks_root: str | os.PathLike):
        self.root = Path(benchmarks_root).resolve()
        if not self.root.exists():
            raise ValueError(f"Benchmark root not found: {self.root}")
        self._tasks: Dict[str, BenchmarkTask] = {}
        self._discover_all()

    # ---------- Public API ----------
    def list_tasks(self) -> List[str]:
        return sorted(self._tasks.keys())

    def list_by_category(self) -> Dict[str, List[str]]:
        out: Dict[str, List[str]] = {}
        for tid, task in self._tasks.items():
            out.setdefault(task.category, []).append(tid)
        for k in out:
            out[k].sort()
        return out

    def get(self, task_id: str) -> BenchmarkTask:
        if task_id not in self._tasks:
            # 友好的错误提示
            candidates = [t for t in self._tasks if task_id.lower() in t.lower()]
            hint = f" Did you mean: {candidates}?" if candidates else ""
            raise KeyError(
                f"Benchmark task '{task_id}' not found.{hint} "
                f"Available: {self.list_tasks()[:10]}..."
            )
        return self._tasks[task_id]

    # ---------- Internal ----------
    def _find_evaluator(self, task_dir: Path) -> Optional[Path]:
        for rel in self.EVALUATOR_CANDIDATES:
            p = task_dir / rel
            if p.exists():
                return p
        return None

    def _find_initial_program(self, task_dir: Path) -> Optional[Path]:
        for name in self.INITIAL_PROGRAM_NAMES:
            p = task_dir / name
            if p.exists():
                return p
        # 某些任务会把 initial 放在 initial/ / program/ / src/ 子目录
        for sub in ("initial", "program", "src"):
            subdir = task_dir / sub
            if subdir.exists() and subdir.is_dir():
                for name in self.INITIAL_PROGRAM_NAMES:
                    p = subdir / name
                    if p.exists():
                        return p
        return None

    def _discover_all(self) -> None:
        """递归扫描 root，最多向下 MAX_DEPTH 层，寻找含 evaluator.py 的目录。"""
        self._walk(self.root, category=None, depth=0)

    def _walk(self, cur: Path, category: Optional[str], depth: int) -> None:
        if depth > self.MAX_DEPTH:
            return
        # 1) 当前目录本身是否就是一个 task?
        eval_path = self._find_evaluator(cur)
        if eval_path is not None and cur != self.root:
            # 决定 category: 如果还没定，用 cur 的直接父目录名；如果父目录就是 root，则 category=root basename
            cat = category
            if cat is None:
                cat = self.root.name if cur.parent == self.root else cur.parent.name
            task = self._try_build_task(cur, category=cat, eval_path=eval_path)
            if task is not None:
                # 决定 registry key:
                #   - 通常直接用 task.task_id (即目录名)
                #   - 如果目录名纯数字 (例如 heilbronn_convex/13) 或者已经冲突, 用父目录名作为前缀
                key = task.task_id
                parent_name = cur.parent.name
                if (key.isdigit() or key in self._tasks) and parent_name and parent_name != cat:
                    key = f"{parent_name}_{task.task_id}"
                if key in self._tasks:
                    key = f"{task.category}_{key}"
                self._tasks[key] = task
            # 不要再往下挖 —— task 子目录里的 evaluator/ 不算新 task
            return

        # 2) 否则把自己当作 category / 中间层，继续往下扫
        try:
            children = list(cur.iterdir())
        except OSError:
            return
        for child in children:
            if not child.is_dir():
                continue
            if child.name.startswith(".") or child.name in self.SKIP_DIR_NAMES:
                continue
            # category 策略：第 0 层（root 下一层）子目录名视为 category；更深层沿用已有 category
            next_category = category
            if depth == 0:
                next_category = child.name
            self._walk(child, category=next_category, depth=depth + 1)

    def _try_build_task(self, task_dir: Path,
                        category: str,
                        eval_path: Path) -> Optional[BenchmarkTask]:
        """根据已定位到的 evaluator 构建 BenchmarkTask。允许 initial_program 缺失。"""
        # 1) 定位 initial_program (可缺失)
        init_path = self._find_initial_program(task_dir)

        # 2) 读取 config.yaml (可选；名称也可能是 config_xxx.yaml)
        cfg: Dict[str, Any] = {}
        config_candidates = [task_dir / "config.yaml", task_dir / "config.yml"]
        config_candidates.extend(sorted(task_dir.glob("config_*.yaml")))
        config_candidates.extend(sorted(task_dir.glob("config_*.yml")))
        seen: set[Path] = set()
        for p in config_candidates:
            if p in seen:
                continue
            seen.add(p)
            if not p.exists():
                continue
            try:
                cfg = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
            except Exception as e:  # noqa: BLE001
                print(f"[BenchmarkRegistry] Warn: failed to parse {p}: {e}")
                continue
            break

        # 3) 推导语言
        if init_path is None:
            language = "python"  # 默认
        elif init_path.suffix in (".py",):
            language = "python"
        elif init_path.suffix in (".txt", ".md"):
            language = "text"
        else:
            language = init_path.suffix.lstrip(".") or "python"

        return BenchmarkTask(
            task_id=task_dir.name,
            category=category,
            root_dir=task_dir,
            # 允许 initial_program 缺失 —— 用一个不存在的哨兵路径，get_initial_program_code() 会返回 ""
            initial_program_path=init_path if init_path is not None else (task_dir / "__no_initial_program__"),
            evaluator_module_path=eval_path,
            language=language,
            config=cfg,
        )


# ==========================================
# 3. Evaluator Loader
# ==========================================

class BenchmarkEvaluatorLoader:
    """
    动态 import 每个 benchmark 的 evaluator.py 并调用其 evaluate(program_path).

    为了隔离不同 benchmark 的顶层 import，这里每次使用独立模块名 + 专属 spec。
    同时会把 task 所在目录临时加入 sys.path，以便 evaluator 能 import 自己的工具模块。
    """

    @staticmethod
    def load(task: BenchmarkTask) -> Callable[[str], Dict[str, Any]]:
        module_name = f"_bench_eval_{task.category}_{task.task_id}_{uuid.uuid4().hex[:6]}"
        spec = importlib.util.spec_from_file_location(
            module_name, str(task.evaluator_module_path)
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot load evaluator from {task.evaluator_module_path}")

        # 把 task 目录 + evaluator 所在目录加到 sys.path, 支持 relative import
        extra_paths = [
            str(task.root_dir),
            str(task.evaluator_module_path.parent),
        ]
        for p in extra_paths:
            if p not in sys.path:
                sys.path.insert(0, p)

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        if not hasattr(module, "evaluate"):
            raise AttributeError(
                f"Evaluator for {task.full_name} does not expose a top-level `evaluate` function."
            )
        return getattr(module, "evaluate")


# ==========================================
# 4. Harness: 对接到我们的 ExecutionHarness
# ==========================================

# 延迟 import 以避免循环依赖
try:
    from execution_harness import BaseHarness, ExecutionTrace  # type: ignore
    from artifact_synthesizer import GeneratedArtifact  # type: ignore
except ImportError:
    BaseHarness = object  # type: ignore
    ExecutionTrace = None  # type: ignore
    GeneratedArtifact = None  # type: ignore


class BenchmarkTaskHarness(BaseHarness):  # type: ignore[misc]
    """
    把 LLM 生成的 artifact 通过 benchmark 自己的 evaluator 进行评估,
    产出完整的 ExecutionTrace (包含数值 score)。
    """

    METRIC_PRIORITIES = [
        "combined_score",   # skydiscover 大多数 task 的首选
        "score",
        "sum_radii",        # circle packing 等
        "target_ratio",
        "value",
        "accuracy",
        "reward",
    ]

    def __init__(self,
                 task: BenchmarkTask,
                 timeout_seconds: int = 300,
                 objective_metric: Optional[str] = None,
                 python_executable: str = "python3",
                 isolate_evaluator: bool = True):
        self.task = task
        self.timeout_seconds = timeout_seconds
        self.objective_metric = objective_metric
        self.python_executable = python_executable
        self.isolate_evaluator = isolate_evaluator
        self._last_evaluator_stdout = ""
        self._last_evaluator_stderr = ""
        # 默认用子进程隔离 evaluator，避免 evaluator/候选程序超时拖住主循环。
        self._evaluate_fn = None if isolate_evaluator else BenchmarkEvaluatorLoader.load(task)

    def run(self, artifacts):  # type: ignore[override]
        traces = []
        for artifact in artifacts:
            rejection_reason = self._artifact_rejection_reason(artifact)
            if rejection_reason:
                trace = ExecutionTrace(
                    artifact_id=artifact.artifact_id,
                    is_completed=True,
                    return_code=1,
                    stdout="[benchmark metrics] {'score': -inf}",
                    stderr=rejection_reason,
                    execution_time_ms=0.0,
                    reproduction_command="not executed: artifact rejected before evaluator",
                )
                trace.metrics["score"] = -float("inf")
                traces.append(trace)
                continue

            suffix = self.task.initial_program_path.suffix or ".py"
            content = self._clean_llm_output(artifact.content)
            syntax_error = self._python_syntax_error(content, suffix=suffix)
            if syntax_error:
                trace = ExecutionTrace(
                    artifact_id=artifact.artifact_id,
                    is_completed=True,
                    return_code=1,
                    stdout="[benchmark metrics] {'score': -inf}",
                    stderr=syntax_error,
                    execution_time_ms=0.0,
                    reproduction_command="not executed: Python syntax precheck failed",
                )
                trace.metrics["score"] = -float("inf")
                traces.append(trace)
                continue

            # 1) 写到临时文件 (必须使用目标语言的后缀)
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=suffix, delete=False
            ) as fp:
                fp.write(content)
                program_path = fp.name

            start = time.time()
            metrics: Dict[str, Any] = {}
            stdout_str = ""
            stderr_str = ""
            return_code = 0
            is_completed = True

            try:
                result = self._invoke_evaluator(program_path)
                if self._last_evaluator_stdout:
                    stdout_str = self._last_evaluator_stdout.strip()
                if self._last_evaluator_stderr:
                    stderr_str = self._last_evaluator_stderr.strip()
                # 允许 evaluator 返回非 dict (例如 float), 做兼容
                if isinstance(result, dict):
                    metrics = self._extract_numeric_metrics(result)
                    # 保留额外字符串信息, 便于诊断
                    notes = self._format_result_notes(result)
                    if notes:
                        stdout_str = f"{stdout_str}\n{notes}".strip() if stdout_str else notes
                    evaluator_error = self._extract_evaluator_error(result)
                    if evaluator_error:
                        return_code = 1
                        stderr_str = (
                            f"{stderr_str}\n[Benchmark Evaluator Error] {evaluator_error}"
                        ).strip()
                elif isinstance(result, (int, float)):
                    metrics = {"score": float(result)}
                else:
                    stdout_str = str(result)
                # 抽取主要 score
                score = self._pick_score(metrics)
                metrics["score"] = score
            except subprocess.TimeoutExpired:
                is_completed = False
                return_code = -1
                stderr_str = f"[Timeout] Benchmark evaluator exceeded {self.timeout_seconds}s."
            except Exception as e:  # noqa: BLE001
                is_completed = False
                return_code = 1
                stderr_str = f"[Benchmark Evaluator Error] {e}\n{traceback.format_exc()[:2000]}"
            finally:
                exec_time_ms = (time.time() - start) * 1000
                try:
                    os.unlink(program_path)
                except OSError:
                    pass

            trace = ExecutionTrace(
                artifact_id=artifact.artifact_id,
                is_completed=is_completed,
                return_code=return_code,
                stdout=stdout_str or f"[benchmark metrics] {metrics}",
                stderr=stderr_str,
                execution_time_ms=exec_time_ms,
                reproduction_command=f"benchmark evaluate({program_path})",
            )
            trace.metrics.update(metrics)
            traces.append(trace)
        return traces

    # ---------- Helpers ----------
    @staticmethod
    def _artifact_rejection_reason(artifact: Any) -> str:
        """Return a pre-execution rejection reason attached by upstream modules."""
        metadata = getattr(artifact, "metadata", {}) or {}
        diff_error = metadata.get("diff_apply_error")
        if isinstance(diff_error, str) and diff_error.strip():
            return f"[DiffApplyError] {diff_error.strip()}"
        return ""

    def _python_syntax_error(self, content: str, *, suffix: str) -> str:
        """Compile Python artifacts before running the expensive benchmark evaluator."""
        language = str(getattr(self.task, "language", "") or "").lower()
        is_python = suffix == ".py" or language in {"python", "py"}
        if not is_python:
            return ""
        try:
            compile(content, "<candidate>", "exec")
        except SyntaxError as exc:
            message = exc.msg or "invalid syntax"
            line = exc.lineno or 0
            return f"[SyntaxPrecheck] line {line}: {message}"
        return ""

    @staticmethod
    def _extract_numeric_metrics(result: Dict[str, Any]) -> Dict[str, Any]:
        """Collect metrics from both direct and wrapper-style evaluator outputs."""
        metrics: Dict[str, Any] = {}
        preserve_json_keys = {"coeffs", "metric_version"}

        nested_metrics = result.get("metrics")
        if isinstance(nested_metrics, dict):
            for key, value in nested_metrics.items():
                if isinstance(value, bool):
                    metrics[key] = float(value)
                elif isinstance(value, (int, float)):
                    metrics[key] = float(value)
                elif key in preserve_json_keys and BenchmarkTaskHarness._is_small_json_metric(value):
                    metrics[key] = value

        for key, value in result.items():
            if key in {"metrics", "artifacts"}:
                continue
            if isinstance(value, bool):
                metrics[key] = float(value)
            elif isinstance(value, (int, float)):
                metrics[key] = float(value)
            elif key in preserve_json_keys and BenchmarkTaskHarness._is_small_json_metric(value):
                metrics[key] = value
        return metrics

    @staticmethod
    def _is_small_json_metric(value: Any, max_items: int = 256) -> bool:
        if value is None or isinstance(value, (str, bool, int, float)):
            return True
        if isinstance(value, list):
            if len(value) > max_items:
                return False
            return all(BenchmarkTaskHarness._is_small_json_metric(v, max_items) for v in value)
        if isinstance(value, dict):
            if len(value) > max_items:
                return False
            return all(
                isinstance(k, str) and BenchmarkTaskHarness._is_small_json_metric(v, max_items)
                for k, v in value.items()
            )
        return False

    @staticmethod
    def _format_result_notes(result: Dict[str, Any]) -> str:
        """Keep non-numeric evaluator metadata visible in trace output."""
        notes: List[str] = []

        for key, value in result.items():
            if key in {"metrics", "artifacts"}:
                continue
            if isinstance(value, str):
                notes.append(f"{key}: {value}")

        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict):
            for key, value in artifacts.items():
                if isinstance(value, (str, int, float, bool)):
                    notes.append(f"{key}: {value}")

        return "\n".join(notes)

    @staticmethod
    def _extract_evaluator_error(result: Dict[str, Any]) -> str:
        """Return a hard evaluator failure reason encoded in a result dict, if any."""
        status = str(result.get("status") or "").lower()
        error = result.get("error")
        if isinstance(error, str) and error.strip():
            return error.strip()

        artifacts = result.get("artifacts")
        if isinstance(artifacts, dict):
            artifact_error = artifacts.get("error")
            if isinstance(artifact_error, str) and artifact_error.strip():
                traceback_text = artifacts.get("traceback")
                if isinstance(traceback_text, str) and traceback_text.strip():
                    return f"{artifact_error.strip()}\n{traceback_text.strip()[-2000:]}"
                return artifact_error.strip()

        if status == "error":
            return "Evaluator returned status=error without a detailed message."
        return ""

    def _invoke_evaluator(self, program_path: str) -> Any:
        """
        直接在本进程内调用 evaluate(). benchmark 内部通常会再用 subprocess 跑 program,
        但外层 evaluator 本身也可能 hang (ALE-Bench session / frontier-cs judge)。
        这里用 signal.SIGALRM 做一层 wall-clock 兜底 (仅 POSIX 主线程有效)。
        """
        self._last_evaluator_stdout = ""
        self._last_evaluator_stderr = ""
        if self.isolate_evaluator:
            return self._invoke_evaluator_subprocess(program_path)

        # POSIX 主线程用 SIGALRM 做硬 timeout；非主线程/Windows 下 fallback 为裸调用
        try:
            import signal, threading
            if threading.current_thread() is threading.main_thread() and hasattr(signal, "SIGALRM"):
                def _on_timeout(signum, frame):
                    raise subprocess.TimeoutExpired(
                        cmd=f"evaluate({program_path})",
                        timeout=self.timeout_seconds,
                    )
                old_handler = signal.signal(signal.SIGALRM, _on_timeout)
                signal.alarm(int(self.timeout_seconds))
                try:
                    return self._evaluate_fn(program_path)
                finally:
                    signal.alarm(0)
                    signal.signal(signal.SIGALRM, old_handler)
        except (ImportError, ValueError):
            pass
        # fallback: 无 signal 支持就直接调用 (evaluator 自身 subprocess 仍有超时)
        return self._evaluate_fn(program_path)

    def _invoke_evaluator_subprocess(self, program_path: str) -> Any:
        """
        Import and run evaluator.evaluate(program_path) in a separate process group.

        This is intentionally heavier than a direct Python call, but it gives the
        outer framework a reliable hard timeout: if the evaluator or the candidate
        program hangs, the whole evaluator process group is killed and the next
        candidate/iteration can continue.
        """
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as result_fp:
            result_path = result_fp.name

        worker = r"""
import importlib.util
import pickle
import sys
import traceback

task_root, evaluator_path, program_path, result_path = sys.argv[1:5]
sys.path.insert(0, task_root)
sys.path.insert(0, __import__("os").path.dirname(evaluator_path))

payload = {"ok": False, "result": None, "error": "", "traceback": ""}
try:
    spec = importlib.util.spec_from_file_location("_oa_isolated_evaluator", evaluator_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load evaluator from {evaluator_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["_oa_isolated_evaluator"] = module
    spec.loader.exec_module(module)
    if not hasattr(module, "evaluate"):
        raise AttributeError("Evaluator does not expose evaluate(program_path)")
    payload["result"] = module.evaluate(program_path)
    payload["ok"] = True
except BaseException as exc:
    payload["error"] = str(exc)
    payload["traceback"] = traceback.format_exc()
finally:
    with open(result_path, "wb") as fp:
        pickle.dump(payload, fp)
"""

        cmd = [
            self.python_executable,
            "-c",
            worker,
            str(self.task.root_dir),
            str(self.task.evaluator_module_path),
            program_path,
            result_path,
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=(os.name == "posix"),
        )
        try:
            stdout, stderr = process.communicate(timeout=self.timeout_seconds)
        except subprocess.TimeoutExpired as exc:
            self._terminate_process_tree(process)
            stdout, stderr = process.communicate()
            self._last_evaluator_stdout = stdout or ""
            self._last_evaluator_stderr = stderr or ""
            try:
                os.unlink(result_path)
            except OSError:
                pass
            raise subprocess.TimeoutExpired(
                cmd=f"evaluate({program_path})",
                timeout=self.timeout_seconds,
                output=stdout,
                stderr=stderr,
            ) from exc

        self._last_evaluator_stdout = stdout or ""
        self._last_evaluator_stderr = stderr or ""

        try:
            with open(result_path, "rb") as fp:
                payload = pickle.load(fp)
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Evaluator subprocess did not produce a result file. "
                f"Return code={process.returncode}.\n"
                f"STDOUT:\n{self._last_evaluator_stdout[-2000:]}\n"
                f"STDERR:\n{self._last_evaluator_stderr[-2000:]}"
            ) from exc
        finally:
            try:
                os.unlink(result_path)
            except OSError:
                pass

        if process.returncode != 0 and not payload.get("ok"):
            raise RuntimeError(
                f"Evaluator subprocess exited with code {process.returncode}.\n"
                f"STDOUT:\n{self._last_evaluator_stdout[-2000:]}\n"
                f"STDERR:\n{self._last_evaluator_stderr[-2000:]}"
            )
        if not payload.get("ok"):
            raise RuntimeError(
                f"{payload.get('error', 'unknown evaluator error')}\n"
                f"{payload.get('traceback', '')[-2000:]}"
            )
        return payload.get("result")

    @staticmethod
    def _terminate_process_tree(process: subprocess.Popen) -> None:
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                import signal
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            pass
        except Exception:
            process.kill()

    def _pick_score(self, metrics: Dict[str, float]) -> float:
        if self.objective_metric and self.objective_metric in metrics:
            return metrics[self.objective_metric]
        for key in self.METRIC_PRIORITIES:
            if key in metrics:
                return metrics[key]
        # 最后兜底: 取数值型指标里最大者。完全没有数值指标时必须标为无效，
        # 否则 evaluator 的空输出会被误当作 score=0 的可归档候选。
        numeric = [v for v in metrics.values() if isinstance(v, (int, float))]
        return float(max(numeric)) if numeric else -float("inf")

    @staticmethod
    def _clean_llm_output(content: str) -> str:
        """
        去掉 LLM 常见的 markdown fence。支持任意语言标签 (```python / ```cpp / ```rust ...)。
        历史实现里 `lstrip("python\\n")` 会误把 'cpp' / 'cc' 开头的代码第一行砍坏,
        这里改成显式识别开头的 language tag 行并裁掉整行。
        """
        import re as _re
        # 最常见: ```<lang>\n...\n```  直接用正则匹配第一个完整的代码块
        m = _re.search(r"```[ \t]*([A-Za-z0-9_+\-]*)[ \t]*\n(.*?)\n```", content, _re.DOTALL)
        if m:
            return m.group(2).strip()
        # 退化: 只有单侧 fence, 尽量保留内容
        if "```" in content:
            parts = content.split("```")
            if len(parts) >= 2:
                body = parts[1]
                first_nl = body.find("\n")
                if first_nl != -1:
                    head = body[:first_nl].strip()
                    # 若第一行是光秃秃的 language tag (只由字母/数字/+/-/_ 组成), 裁掉
                    if head and _re.fullmatch(r"[A-Za-z0-9_+\-]+", head):
                        body = body[first_nl + 1:]
                return body.strip()
        return content


# ==========================================
# 5. CLI Helper (枚举 benchmark)
# ==========================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default=str(Path(__file__).resolve().parents[1] / "benchmarks"),
    )
    parser.add_argument("--show", default=None, help="show task description by id")
    args = parser.parse_args()

    reg = BenchmarkRegistry(args.root)
    if args.show:
        task = reg.get(args.show)
        print("=== TASK:", task.full_name)
        print("language:", task.language)
        print("root:", task.root_dir)
        print("-----")
        print(task.get_task_description()[:2000])
    else:
        by_cat = reg.list_by_category()
        for cat, ts in by_cat.items():
            print(f"[{cat}] ({len(ts)})")
            for t in ts:
                print(f"  - {t}")
