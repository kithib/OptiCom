"""
run.py — YAML-driven entry point for the OptiCom agent.

用法:
    # 最常用 (用 example 下的 yaml 启动)
    python run.py --config example/circle_packing.yaml
    python run.py --config example/benchmark_arc.yaml

    # 列出当前 yaml 里 benchmark.root 下所有可选 benchmark task_id:
    python run.py --config example/benchmark_arc.yaml --list-benchmarks

    # 临时覆盖 yaml 里的值 (CLI 优先于 yaml):
    python run.py --config example/benchmark_arc.yaml \\
           --benchmark circle_packing --max-iter 3

所有子模块来自同目录下的 src/ 子包。
"""
import argparse
import json
import re
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# ==========================================
# 0. 让 src/ 下的框架模块可被 import
# ==========================================
THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parent
SRC_DIR = V1_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import yaml  # noqa: E402

from problem_formulator import RuleBasedFormulator  # noqa: E402
from optimization_state import (  # noqa: E402
    OptimizationState, RemainingBudget, OptimizationPhase, ArtifactPreview,
)
from optimization_controller import RuleBasedController  # noqa: E402
from query_engine import QueryEngine  # noqa: E402
from context_assembler import ContextAssembler  # noqa: E402
from operator_library import OperatorLibrary  # noqa: E402
from artifact_synthesizer import ArtifactSynthesizer  # noqa: E402
from llm_clients import LLMClientFactory  # noqa: E402
from benchmark_adapter import (  # noqa: E402
    BenchmarkRegistry, BenchmarkTask, BenchmarkTaskHarness,
)
from execution_harness import ExecutionHarness, PythonCodeHarness  # noqa: E402
from evaluation_stack import EvaluationStack  # noqa: E402
from archive_manager import ArchiveManager  # noqa: E402
from archive_manager import (  # noqa: E402
    EvaluationRecord as ArchiveEvaluationRecord,
    GeneratedArtifact as ArchiveGeneratedArtifact,
)
from optimization_memory import OptimizationMemory  # noqa: E402
from strategy_adapter import StrategyAdapter  # noqa: E402
from baseline_profiles import (  # noqa: E402
    BaselineController,
    filter_memory_snapshot,
    get_baseline_profile,
    list_baseline_names,
    normalize_baseline_name,
    profile_catalog_json,
)


# ==========================================
# 1. YAML 载入与合并
# ==========================================
DEFAULT_CONFIG: Dict[str, Any] = {
    "runner": {
        "max_iterations": 10,
        "stagnation_threshold": 3,
        "task_description": None,
        "formulator_mode": "algorithm_tuning",
        "baseline": "opticom",
        "warm_start_from_result": True,
        "seed_initial_evaluation": True,
        "numerical_polish_seed": True,
        "numerical_polish_seconds": 90,
        "numerical_polish_maxiter": 900,
        "numerical_polish_max_extra_dims": 2,
        "numerical_polish_rounds": 3,
    },
    "benchmark": {
        "enabled": False,
        "root": "./benchmarks",
        "task_id": None,
    },
    "llm": {
        "provider": "openai",
        "model": None,
        "api_key": None,
        "base_url": None,
        "temperature": 0.7,
        "max_tokens": 4096,
        "top_p": 1.0,
        "timeout": 120.0,
        "max_retries": 2,
    },
    "harness": {
        "python_timeout": 45,
        "benchmark_timeout": 300,
        "python_executable": None,  # 留空则默认使用 sys.executable (当前运行 run.py 的环境)
    },
    "archive": {
        "top_k": 3,
        "objective_metric": None,
        "maximize": True,
    },
    "memory": {
        "summarize_every_n_events": 3,
    },
    "evaluation": {
        "enable_robustness_gate": False,
        # None means benchmark tasks inherit the benchmark timeout as their soft
        # budget; inline toy tasks keep the historical 30s default.
        "cost_soft_limit_ms": None,
    },
}


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """深合并: override 中的非 None 值覆盖 base。"""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif v is not None:
            out[k] = v
    return out


def load_config(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        user_cfg = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user_cfg)
    cfg.setdefault("_meta", {})["config_path"] = str(path)
    cfg["_meta"]["config_dir"] = str(path.parent.resolve())
    return cfg


def _resolve_path(maybe_relative: str, base_dir: str) -> str:
    """把 yaml 里的相对路径解析成绝对路径 (基于 yaml 所在目录)。"""
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    return str((Path(base_dir) / p).resolve())


# ==========================================
# 1.5 Result Archive (历史最佳落盘)
# ==========================================
RESULT_ROOT = V1_ROOT / "result"


def _safe_slug(name: str) -> str:
    """把任务名变成可安全作为目录名的 slug (保留字母数字 _ - ., 其余换成 _)."""
    slug = re.sub(r"[^A-Za-z0-9._\-]+", "_", (name or "").strip())
    return slug.strip("_") or "untitled"


def _llm_name(cfg: Dict[str, Any]) -> str:
    """提取 LLM 的 provider 和 model 组成字符串，如 'openai_your-model'"""
    provider = cfg.get("llm", {}).get("provider", "unknown")
    model = cfg.get("llm", {}).get("model", "unknown")
    return _safe_slug(f"{provider}_{model}")


def _baseline_name(cfg: Optional[Dict[str, Any]]) -> str:
    if not cfg:
        return "opticom"
    return normalize_baseline_name((cfg.get("runner", {}) or {}).get("baseline"))


def _result_dir(*, baseline_name: str, task_slug: str, llm_name: str) -> Path:
    """Canonical best-result layout: result/<baseline>/<task>/<model>/."""
    return RESULT_ROOT / _safe_slug(baseline_name) / _safe_slug(task_slug) / _safe_slug(llm_name)


def _task_slug(
    benchmark_task: Optional[BenchmarkTask],
    cfg_path: Path,
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """Task key independent of baseline/model for result/<baseline>/<task>/<model>."""
    if benchmark_task is not None:
        configured_task_id = None
        if cfg is not None:
            configured_task_id = (cfg.get("benchmark", {}) or {}).get("task_id")
        task_name = configured_task_id or benchmark_task.task_id
        return _safe_slug(f"{benchmark_task.category}__{task_name}")
    return _safe_slug(f"inline__{cfg_path.stem}")


def _benchmark_display_name(
    benchmark_task: Optional[BenchmarkTask],
    cfg: Optional[Dict[str, Any]] = None,
) -> str:
    """benchmark 显示名优先使用配置里的 task_id，避免目录名如 `13` 泄漏到日志中。"""
    if benchmark_task is None:
        return "inline"
    configured_task_id = None
    if cfg is not None:
        configured_task_id = (cfg.get("benchmark", {}) or {}).get("task_id")
    task_name = configured_task_id or benchmark_task.task_id
    return f"{benchmark_task.category}/{task_name}"


def _code_filename(benchmark_task: Optional[BenchmarkTask]) -> str:
    if benchmark_task is None:
        return "best_program.py"
    lang = (benchmark_task.language or "python").lower()
    ext_map = {
        "python": "py", "cpp": "cpp", "c": "c", "cc": "cc", "cxx": "cxx",
        "rust": "rs", "rs": "rs", "go": "go", "java": "java",
        "ts": "ts", "js": "js", "text": "txt", "yaml": "yaml", "json": "json",
    }
    return f"best_program.{ext_map.get(lang, 'txt')}"


def _formulator_mode_for_task(runner_cfg: Dict[str, Any],
                              benchmark_task: Optional[BenchmarkTask]) -> str:
    mode = str(runner_cfg.get("formulator_mode") or "algorithm_tuning")
    if benchmark_task is None:
        return mode
    language = (benchmark_task.language or "").lower()
    category = (benchmark_task.category or "").lower()
    if language == "text" or category == "prompt_optimization":
        return "prompt_optimization"
    return mode


def _read_previous_best_score(result_dir: Path) -> Optional[float]:
    """读取 result_dir/result.json 里保存的上次 runtime_score; 没有则 None."""
    result_file = result_dir / "result.json"
    if not result_file.exists():
        return None
    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
        val = data.get("runtime_score")
        return float(val) if isinstance(val, (int, float)) else None
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] failed to parse existing {result_file}: {e}")
        return None


def _load_warm_start_result(
    *,
    cfg: Dict[str, Any],
    cfg_path: Path,
    benchmark_task: Optional[BenchmarkTask],
    objective_key: str,
) -> Optional[Dict[str, Any]]:
    """Load the previous best artifact for this exact baseline/task/model, if safe."""
    if not bool((cfg.get("runner", {}) or {}).get("warm_start_from_result", True)):
        return None

    task_slug = _task_slug(benchmark_task, cfg_path, cfg)
    out_dir = _result_dir(
        baseline_name=_baseline_name(cfg),
        task_slug=task_slug,
        llm_name=_llm_name(cfg),
    )
    result_file = out_dir / "result.json"
    if not result_file.exists():
        return None

    try:
        data = json.loads(result_file.read_text(encoding="utf-8"))
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] failed to parse warm-start result {result_file}: {e}")
        return None

    if not bool(data.get("best_is_passed")):
        return None

    code_file = str(data.get("code_file") or _code_filename(benchmark_task))
    code_path = out_dir / code_file
    if not code_path.exists():
        return None

    metrics = data.get("all_metrics") or {}
    if not isinstance(metrics, dict):
        metrics = {}
    runtime_score = data.get("runtime_score")
    if isinstance(runtime_score, (int, float)):
        metrics.setdefault("runtime_score", float(runtime_score))
        metrics.setdefault("score", float(runtime_score))
        metrics.setdefault(objective_key, float(runtime_score))
    if objective_key not in metrics or not isinstance(metrics.get(objective_key), (int, float)):
        return None

    try:
        code = code_path.read_text(encoding="utf-8")
    except OSError as e:
        print(f"[WARN] failed to read warm-start artifact {code_path}: {e}")
        return None

    return {
        "artifact_id": str(data.get("best_artifact_id") or "warm_start"),
        "code": code,
        "metrics": {
            k: float(v) if isinstance(v, (int, float)) else v
            for k, v in metrics.items()
        },
        "score": float(metrics[objective_key]),
        "result_dir": str(out_dir),
    }


def _build_uncertainty_polish_artifact(
    *,
    cfg: Dict[str, Any],
    benchmark_task: Optional[BenchmarkTask],
    warm_start: Optional[Dict[str, Any]],
) -> Optional[ArchiveGeneratedArtifact]:
    """
    Generate a deterministic numerical-polish seed for the uncertainty C4 task.

    This is intentionally evaluator-bound: the candidate is still sent through
    BenchmarkTaskHarness/EvaluationStack before it can enter the archive.
    """
    if not bool((cfg.get("runner", {}) or {}).get("numerical_polish_seed", True)):
        return None
    if benchmark_task is None or benchmark_task.task_id != "uncertainty_ineq":
        return None
    if not warm_start:
        return None

    metrics = warm_start.get("metrics") or {}
    coeffs_raw = metrics.get("coeffs")
    if not isinstance(coeffs_raw, list) or len(coeffs_raw) < 3:
        try:
            namespace: Dict[str, Any] = {"__name__": "_oa_uncertainty_warm_artifact"}
            exec(str(warm_start.get("code") or ""), namespace)  # noqa: S102
            coeffs_from_run, c4_from_run, _ = namespace["run"]()
            coeffs_raw = [float(x) for x in coeffs_from_run]
            metrics = dict(metrics)
            metrics.setdefault("coeffs", coeffs_raw)
            metrics.setdefault("c4_bound", float(c4_from_run))
        except Exception as exc:  # noqa: BLE001
            print(f"[POLISH] uncertainty warm coefficients unavailable: {exc}")
            return None

    try:
        import importlib.util
        import numpy as np
        from scipy.optimize import minimize
        from scipy.special import hermite
    except Exception as exc:  # noqa: BLE001
        print(f"[POLISH] uncertainty polish unavailable: {exc}")
        return None

    def fast_c4(input_coeffs):
        coeffs = np.asarray(input_coeffs, dtype=float)
        norm = np.linalg.norm(coeffs)
        if not np.all(np.isfinite(coeffs)) or norm < 1e-14:
            return None
        coeffs = coeffs / norm
        m = len(coeffs)
        degrees = [4 * k for k in range(m + 1)]
        polys = [hermite(d) for d in degrees]
        max_degree = degrees[-1]
        h0 = np.array([p(0.0) for p in polys], dtype=float)
        c_last = -float(np.dot(coeffs, h0[:m])) / h0[m]
        full = np.concatenate([coeffs, [c_last]])

        poly_coeffs = np.zeros(max_degree + 1)
        for c, poly in zip(full, polys):
            poly_coeffs[max_degree - poly.order:] += c * poly.coef
        if poly_coeffs[0] < 0:
            poly_coeffs = -poly_coeffs

        q_poly, remainder = np.polydiv(poly_coeffs, np.array([1.0, 0.0, 0.0]))
        if np.max(np.abs(remainder)) > 1e-6 * max(1.0, np.max(np.abs(poly_coeffs))):
            return None

        roots = np.roots(q_poly)
        scale = max(1.0, np.max(np.abs(q_poly)))
        positive_roots = []
        for root in roots:
            if abs(root.imag) < 1e-5 * max(1.0, abs(root.real)) and root.real > 1e-8:
                r = float(root.real)
                eps = 1e-7 * max(1.0, abs(r))
                left = np.polyval(q_poly, r - eps)
                right = np.polyval(q_poly, r + eps)
                if left * right < 0 or abs(np.polyval(q_poly, r)) < 1e-4 * scale:
                    positive_roots.append(r)
        if not positive_roots:
            return None
        r_max = max(positive_roots)
        return (r_max * r_max) / (2.0 * np.pi), coeffs

    runner_cfg = cfg.get("runner", {}) or {}
    try:
        polish_seconds = max(1.0, float(runner_cfg.get("numerical_polish_seconds", 90)))
    except (TypeError, ValueError):
        polish_seconds = 90.0
    try:
        polish_maxiter = max(100, int(runner_cfg.get("numerical_polish_maxiter", 900)))
    except (TypeError, ValueError):
        polish_maxiter = 900
    try:
        max_extra_dims = max(0, int(runner_cfg.get("numerical_polish_max_extra_dims", 2)))
    except (TypeError, ValueError):
        max_extra_dims = 2
    try:
        polish_rounds = max(1, int(runner_cfg.get("numerical_polish_rounds", 3)))
    except (TypeError, ValueError):
        polish_rounds = 3

    base = np.asarray(coeffs_raw, dtype=float)
    current = fast_c4(base)
    if current is None:
        return None
    best_c4, best_coeffs = current

    def objective(x):
        out = fast_c4(x)
        return 1e3 if out is None else out[0]

    local_scales = (3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2)
    high_dim_scales = (
        1e-14, 3e-14, 1e-13, 3e-13, 1e-12, 3e-12,
        1e-11, 3e-11, 1e-10, 3e-10, 1e-9, 3e-9,
        1e-8, 3e-8,
    )
    low_dim_factors = (0.92, 0.95, 0.97, 0.985, 0.99, 0.995, 1.005, 1.015, 1.03, 1.06)

    rng = np.random.default_rng(12345)

    def run_polish_round(round_base):
        nonlocal best_c4, best_coeffs
        round_base = np.asarray(round_base, dtype=float)
        base_norm = max(1.0, float(np.linalg.norm(round_base)))

        base_variants = [round_base]
        for drop in range(1, min(2, len(round_base) - 3) + 1):
            base_variants.append(round_base[:-drop])

        start_batches = []
        for extra_dims in range(max_extra_dims + 1):
            target_dim = len(round_base) + extra_dims
            variants_for_dim = list(reversed(base_variants[1:])) + base_variants[:1]

            for variant in variants_for_dim:
                if len(variant) > target_dim:
                    continue
                seed = np.zeros(target_dim, dtype=float)
                seed[: len(variant)] = variant
                starts = [seed]

                low_dim_seeds = []
                if len(variant) >= 2:
                    for factor in low_dim_factors:
                        candidate = seed.copy()
                        candidate[1] *= factor
                        low_dim_seeds.append(candidate)
                    starts.extend(low_dim_seeds)

                if len(variant) < target_dim:
                    first_extra_idx = len(variant)
                    for scale in high_dim_scales:
                        for sign in (-1.0, 1.0):
                            candidate = seed.copy()
                            candidate[first_extra_idx] = sign * scale * base_norm
                            starts.append(candidate)
                    for low_seed in low_dim_seeds:
                        for scale in (1e-12, 3e-12, 1e-11, 3e-11, 1e-10):
                            for sign in (-1.0, 1.0):
                                candidate = low_seed.copy()
                                candidate[first_extra_idx] = sign * scale * base_norm
                                starts.append(candidate)

                random_restarts = 4 if extra_dims == 0 else 8
                for scale in local_scales:
                    for _ in range(random_restarts):
                        noise = rng.normal(size=target_dim) * scale * base_norm
                        if len(variant) < target_dim:
                            noise[len(variant):] *= 1e-6
                        starts.append(seed + noise)
                start_batches.append(starts)

        deadline = time.time() + polish_seconds
        for batch_index, starts in enumerate(start_batches):
            if time.time() > deadline:
                break
            remaining_batches = max(1, len(start_batches) - batch_index)
            batch_deadline = min(
                deadline,
                time.time() + max(1.0, (deadline - time.time()) / remaining_batches),
            )
            for start in starts:
                if time.time() > batch_deadline:
                    break
                try:
                    result = minimize(
                        objective,
                        start,
                        method="Nelder-Mead",
                        options={"maxiter": polish_maxiter, "xatol": 1e-9, "fatol": 1e-9, "disp": False},
                    )
                except Exception:  # noqa: BLE001
                    continue
                out = fast_c4(result.x)
                if out is not None and out[0] < best_c4 - 1e-12:
                    best_c4, best_coeffs = out

    for round_index in range(polish_rounds):
        round_start_c4 = best_c4
        print(
            "[POLISH] uncertainty round "
            f"{round_index + 1}/{polish_rounds} start: c4={round_start_c4:.12g}, "
            f"dim={len(best_coeffs)}"
        )
        run_polish_round(best_coeffs)
        if best_c4 >= round_start_c4 - 1e-12:
            print(
                "[POLISH] uncertainty round "
                f"{round_index + 1}/{polish_rounds} no improvement; stopping."
            )
            break
        print(
            "[POLISH] uncertainty round "
            f"{round_index + 1}/{polish_rounds} improved: "
            f"c4 {round_start_c4:.12g} -> {best_c4:.12g}, dim={len(best_coeffs)}"
        )

    previous_c4 = metrics.get("c4_bound")
    if isinstance(previous_c4, (int, float)) and best_c4 >= float(previous_c4) - 1e-12:
        return None

    # Strictly recompute with the benchmark evaluator before creating code.
    try:
        spec = importlib.util.spec_from_file_location(
            "_oa_uncertainty_eval_for_polish",
            str(benchmark_task.evaluator_module_path),
        )
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        strict_c4, strict_rmax = module.compute_c4_and_rmax(best_coeffs)
        strict_score = float(module.BENCHMARK / strict_c4)
    except Exception as exc:  # noqa: BLE001
        print(f"[POLISH] uncertainty strict validation failed: {exc}")
        return None

    previous_score = metrics.get("combined_score") or metrics.get("runtime_score")
    if isinstance(previous_score, (int, float)) and strict_score <= float(previous_score) + 1e-12:
        return None

    coeff_literal = repr([float(x) for x in best_coeffs])
    code = (
        "import numpy as np\n\n"
        "# EVOLVE-BLOCK-START\n"
        "def run():\n"
        f"    coeffs = np.array({coeff_literal}, dtype=float)\n"
        f"    c4_bound = {strict_c4!r}\n"
        f"    r_max = {strict_rmax!r}\n"
        "    return coeffs, c4_bound, r_max\n"
        "# EVOLVE-BLOCK-END\n"
    )
    print(
        "[POLISH] uncertainty seed generated "
        f"(c4 {previous_c4} -> {strict_c4:.12g}, score={strict_score:.12g})"
    )
    return ArchiveGeneratedArtifact(
        "uncertainty_polish_seed",
        code,
        {
            "parent_id": warm_start.get("artifact_id", "warm_start"),
            "operator": "deterministic_uncertainty_polish",
        },
    )


def save_result_if_best(
    *,
    llm_name: str,
    task_slug: str,
    benchmark_task: Optional[BenchmarkTask],
    cfg: Dict[str, Any],
    cfg_path: Path,
    best_record_metrics: Dict[str, Any],
    best_code: str,
    best_artifact_id: str,
    best_is_passed: bool = True,
    objective_key: str,
    maximize: bool,
    improvement_history: list,
    elapsed_sec: float,
) -> Tuple[bool, Path, Optional[float]]:
    """
    如果本次 best 的 runtime_score 高于档案中已有值, 则把 config/代码/成绩写入
    RESULT_ROOT/<baseline>/<task>/<model>/. 返回 (is_new_best, out_dir, prev_score).
    """
    import math
    baseline_name = _baseline_name(cfg)
    out_dir = _result_dir(
        baseline_name=baseline_name,
        task_slug=task_slug,
        llm_name=llm_name,
    )
    current_score = best_record_metrics.get(objective_key)
    if not best_is_passed:
        return False, out_dir, None
    if not isinstance(current_score, (int, float)) or math.isnan(current_score) or math.isinf(current_score):
        return False, out_dir, None

    prev_score = _read_previous_best_score(out_dir)

    if prev_score is None:
        is_new_best = True
    elif maximize:
        is_new_best = float(current_score) > float(prev_score) + 1e-12
    else:
        is_new_best = float(current_score) < float(prev_score) - 1e-12
    if not is_new_best:
        return False, out_dir, prev_score

    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) 代码/程序
    code_name = _code_filename(benchmark_task)
    (out_dir / code_name).write_text(best_code, encoding="utf-8")

    # 2) config.yaml 副本 (剥掉 _meta, yaml 注入过的运行时字段)
    cfg_copy = {k: v for k, v in cfg.items() if k != "_meta"}
    (out_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg_copy, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    # 3) 原始 yaml 也拷一份作为 "config.source.yaml" (便于看注释)
    try:
        (out_dir / "config.source.yaml").write_text(
            cfg_path.read_text(encoding="utf-8"), encoding="utf-8",
        )
    except Exception:  # noqa: BLE001
        pass

    # 4) 成绩 json
    # 抽出"绝对值"主指标 (sum_radii/combined_score/...)
    abs_candidates = ["sum_radii", "combined_score", "target_ratio",
                      "accuracy", "reward", "value"]
    abs_metric, abs_value = None, None
    for k in abs_candidates:
        if k in best_record_metrics and isinstance(best_record_metrics[k], (int, float)):
            abs_metric, abs_value = k, float(best_record_metrics[k])
            break

    summary = {
        "task_slug": task_slug,
        "task_full_name": _benchmark_display_name(benchmark_task, cfg),
        "baseline": baseline_name,
        "model_slug": llm_name,
        "result_layout": "result/<baseline>/<task>/<model>",
        "is_benchmark": benchmark_task is not None,
        "benchmark_category": benchmark_task.category if benchmark_task else None,
        "benchmark_task_id": (
            ((cfg.get("benchmark", {}) or {}).get("task_id"))
            if benchmark_task else None
        ) or (benchmark_task.task_id if benchmark_task else None),
        "benchmark_language": benchmark_task.language if benchmark_task else None,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_sec": round(elapsed_sec, 2),
        "objective_key": objective_key,
        "maximize": bool(maximize),
        "runtime_score": float(current_score),
        "previous_runtime_score": float(prev_score) if prev_score is not None else None,
        "delta_vs_previous": (
            float(current_score) - float(prev_score) if prev_score is not None else None
        ),
        "abs_metric": abs_metric,
        "abs_value": abs_value,
        "all_metrics": {
            k: (float(v) if isinstance(v, (int, float)) else v)
            for k, v in best_record_metrics.items()
        },
        "best_artifact_id": best_artifact_id,
        "best_is_passed": bool(best_is_passed),
        "improvement_history": improvement_history,
        "code_file": code_name,
        "llm": {
            "provider": cfg.get("llm", {}).get("provider"),
            "model": cfg.get("llm", {}).get("model"),
        },
        "runner": {
            "baseline": _baseline_name(cfg),
            "max_iterations": cfg.get("runner", {}).get("max_iterations"),
            "stagnation_threshold": cfg.get("runner", {}).get("stagnation_threshold"),
        },
    }
    for metric_key in ("combined_score", "c4_bound", "r_max", "coeffs"):
        metric_value = best_record_metrics.get(metric_key)
        if isinstance(metric_value, (int, float)):
            summary[metric_key] = float(metric_value)
        elif metric_key == "coeffs" and isinstance(metric_value, list) and len(metric_value) <= 256:
            summary[metric_key] = metric_value
    (out_dir / "result.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    return True, out_dir, prev_score


# ==========================================
# 2. 初始化（仅接受合并好的 config dict）
# ==========================================
def initialize_system(cfg: Dict[str, Any],
                      task_desc: str,
                      benchmark_task: Optional[BenchmarkTask] = None):
    """根据 cfg 和 task_desc 实例化全部 12 个模块。"""

    runner_cfg = cfg["runner"]
    llm_cfg = cfg["llm"]
    harness_cfg = cfg["harness"]
    archive_cfg = cfg["archive"]
    memory_cfg = cfg["memory"]

    # 1. Problem Formulator
    #    支持 yaml 中 `runner.budget` 覆盖模板预算 (max_llm_calls / max_sandbox_runs /
    #    max_wall_clock_mins / max_api_cost_usd), 数值优化类 benchmark 常常需要更大预算。
    formulator = RuleBasedFormulator()
    budget_overrides = runner_cfg.get("budget") or {}
    formulator_mode = _formulator_mode_for_task(runner_cfg, benchmark_task)
    problem = formulator.formulate(
        task_desc,
        formulator_mode,
        budget_overrides=budget_overrides if isinstance(budget_overrides, dict) else None,
    )

    # 2. State
    state = OptimizationState(
        active_constraints=problem.constraints,
        remaining_budget=RemainingBudget(
            llm_calls=problem.budget.max_llm_calls,
            sandbox_runs=problem.budget.max_sandbox_runs,
            wall_clock_mins=problem.budget.max_wall_clock_mins,
            api_cost_usd=problem.budget.max_api_cost_usd,
        ),
        current_phase=OptimizationPhase.FEASIBILITY,
    )

    # 3. LLM Client (provider 全部由 yaml 决定，但仍允许通过 env var 兜底 api_key)
    provider = llm_cfg["provider"]
    model = llm_cfg.get("model")
    api_key = llm_cfg.get("api_key")
    base_url = llm_cfg.get("base_url")

    if not model and provider != "mock":
        raise ValueError(
            f"[Config Error] llm.model is required for provider '{provider}'. "
            f"Please fill it in the YAML config."
        )

    print(f"[OK] Initializing LLM client (provider={provider}, model={model})")
    # yaml 里 llm.extra 可以透传给具体后端。
    llm_extra = llm_cfg.get("extra", {}) or {}
    llm_client = LLMClientFactory.create(
        provider=provider,
        api_key=api_key,
        model=model,
        base_url=base_url,
        temperature=llm_cfg.get("temperature", 0.7),
        max_tokens=llm_cfg.get("max_tokens", 4096),
        top_p=llm_cfg.get("top_p", 1.0),
        timeout=float(llm_cfg.get("timeout", 120)),
        max_retries=int(llm_cfg.get("max_retries", 2)),
        **llm_extra,
    )

    # 4. Controller / Query / Context / Operator
    baseline_profile = get_baseline_profile(runner_cfg.get("baseline"))
    if baseline_profile is None:
        controller = RuleBasedController(
            stagnation_threshold=int(runner_cfg.get("stagnation_threshold", 3))
        )
        print("[OK] Baseline profile -> opticom (full framework)")
    else:
        controller = BaselineController(
            profile=baseline_profile,
            stagnation_threshold=int(runner_cfg.get("stagnation_threshold", 3)),
        )
        print(f"[OK] Baseline profile -> {baseline_profile.display_name}")
    controller.config["maximize"] = bool(archive_cfg.get("maximize", True))
    controller.config["task_description"] = task_desc
    if benchmark_task is not None:
        controller.config["benchmark_task_id"] = benchmark_task.task_id
        controller.config["benchmark_category"] = benchmark_task.category
        controller.config["benchmark_language"] = benchmark_task.language
    query_engine = QueryEngine()
    # yaml `context` 段可以调节压缩阈值: 对复杂数值脚本 (>180 行) 默认 4000 字符更合适。
    context_cfg = cfg.get("context", {}) or {}
    context_assembler = ContextAssembler(
        max_context_length=int(context_cfg.get("max_context_length", 8000)),
        artifact_max_chars=int(context_cfg.get("artifact_max_chars", 4000)),
        evidence_max_chars=int(context_cfg.get("evidence_max_chars", 1500)),
    )
    operator_library = OperatorLibrary()

    # 5. Synthesizer
    synthesizer = ArtifactSynthesizer(llm_client=llm_client)

    # 6. Harness: benchmark 绑定时走 BenchmarkTaskHarness
    python_executable = harness_cfg.get("python_executable") or sys.executable
    if benchmark_task is not None:
        bench_name = _benchmark_display_name(benchmark_task, cfg)
        print(f"[OK] BenchmarkTaskHarness -> {bench_name} (using {python_executable})")
        harness_adapter = BenchmarkTaskHarness(
            task=benchmark_task,
            timeout_seconds=int(harness_cfg.get("benchmark_timeout", 300)),
            python_executable=python_executable,
        )
    else:
        print(f"[OK] PythonCodeHarness -> Inline Task (using {python_executable})")
        harness_adapter = PythonCodeHarness(
            timeout_seconds=int(harness_cfg.get("python_timeout", 45)),
            python_executable=python_executable,
        )
    harness = ExecutionHarness(adapter=harness_adapter)

    # 7. Evaluation / Archive / Memory / StrategyAdapter
    eval_cfg = cfg.get("evaluation", {}) or {}
    raw_soft_limit = eval_cfg.get("cost_soft_limit_ms")
    if raw_soft_limit is None:
        if benchmark_task is not None:
            soft_limit_ms = float(harness_cfg.get("benchmark_timeout", 300)) * 1000.0
        else:
            soft_limit_ms = 30000.0
    else:
        soft_limit_ms = float(raw_soft_limit)
    evaluator = EvaluationStack(
        llm_client=llm_client,
        enable_robustness=bool(eval_cfg.get("enable_robustness_gate", False)),
        cost_soft_limit_ms=soft_limit_ms,
    )

    objective_key = archive_cfg.get("objective_metric") \
        or f"{problem.objective.value}_score"

    archive = ArchiveManager(
        top_k=int(archive_cfg.get("top_k", 3)),
        objective_metric=objective_key,
        maximize=bool(archive_cfg.get("maximize", True)),
    )
    memory = OptimizationMemory(
        llm_client=llm_client,
        summarize_every_n_events=int(memory_cfg.get("summarize_every_n_events", 3)),
    )
    strategy_adapter = StrategyAdapter(
        llm_client=llm_client,
        controller=controller,
        operator_library=operator_library,
    )

    return (problem, state, controller, query_engine, context_assembler,
            operator_library, synthesizer, harness, evaluator,
            archive, memory, strategy_adapter, objective_key)


# ==========================================
# 3. State Glue
# ==========================================
def update_state_glue(state: OptimizationState, archive: ArchiveManager,
                      memory: OptimizationMemory, records: list, obj_key: str) -> None:
    best_archived = archive.best_artifact
    if best_archived:
        snapshot = archive.snapshot()
        best_score = best_archived.record.metrics.get(obj_key, 0.0)
        has_valid_score = (
            best_score not in (None, -float("inf"), float("inf"))
            and best_score == best_score
        )
        compact_metrics = snapshot.get("best_metrics") or {}
        metric_text = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in compact_metrics.items()
        )
        descriptor = "Current top performer"
        if metric_text:
            descriptor += f"; metrics: {metric_text}"
        preview = ArtifactPreview(
            artifact_id=best_archived.artifact_id,
            score=best_score,
            is_feasible=bool(best_archived.record.is_passed and has_valid_score),
            summary_descriptor=descriptor,
        )
        state.update_from_archive(
            best_artifact=preview,
            frontier_summary=snapshot.get("top_k_exemplars", [])[:1],
        )

    recent_failures = [r.diagnostics for r in records if not r.is_passed]
    recent_signals = [r.revision_signal for r in records if not r.is_passed]
    state.update_from_memory(
        errors=recent_failures,
        revision_signals=recent_signals,
        uncertainty_flags=[],
    )


# ==========================================
# 4. 主循环
# ==========================================
def run_optimization_loop(cfg: Dict[str, Any],
                          task: str,
                          benchmark_task: Optional[BenchmarkTask] = None):

    max_iterations = int(cfg["runner"].get("max_iterations", 10))
    display_name = (_benchmark_display_name(benchmark_task, cfg)
                    if benchmark_task else "Inline Task (no benchmark)")
    print(f"Starting Agentic Optimization Loop\nTask: {display_name}\n" + "=" * 50)

    try:
        (problem, state, controller, query_engine, context_assembler,
         operator_library, synthesizer, harness, evaluator,
         archive, memory, strategy_adapter,
         objective_key) = initialize_system(cfg, task, benchmark_task=benchmark_task)
    except ValueError as e:
        print(e)
        return None

    baseline_profile = get_baseline_profile(cfg.get("runner", {}).get("baseline"))
    iteration = 1

    # 初始 artifact: benchmark 用 initial_program, 否则默认 circle packing naive
    if benchmark_task is not None:
        initial_code = benchmark_task.get_initial_program_code() or "# empty initial program\n"
        init_descriptor = f"benchmark:{_benchmark_display_name(benchmark_task, cfg)} initial_program"
    else:
        initial_code = """import math

def construct_packing_n26():
    radius = 1.0 / 12.0
    circles = []
    count = 0
    for i in range(6):
        for j in range(5):
            if count >= 26:
                break
            x = radius + i * (2 * radius)
            y = radius + j * (2 * radius)
            circles.append((x, y, radius))
            count += 1
    return circles

if __name__ == "__main__":
    circles = construct_packing_n26()
    for x, y, r in circles:
        assert 0 <= x - r and x + r <= 1.0, "Circle out of bounds"
        assert 0 <= y - r and y + r <= 1.0, "Circle out of bounds"
    sum_radii = sum(c[2] for c in circles)
    print(f"Sum of Radii: {sum_radii:.4f}")
"""
        init_descriptor = "Naive 6x5 grid layout"

    state.update_from_archive(
        ArtifactPreview("init", -float("inf"), False, init_descriptor), []
    )
    setattr(state, "current_artifact", initial_code)

    cfg_path_for_results = Path(cfg.get("_meta", {}).get("config_path") or (V1_ROOT / "config.yaml"))
    warm_start = _load_warm_start_result(
        cfg=cfg,
        cfg_path=cfg_path_for_results,
        benchmark_task=benchmark_task,
        objective_key=objective_key,
    )
    if warm_start is not None:
        warm_artifact_id = f"warm_{warm_start['artifact_id']}"
        warm_artifact = ArchiveGeneratedArtifact(
            warm_artifact_id,
            warm_start["code"],
            {"parent_id": "result_archive", "operator": "warm_start_from_result"},
        )
        warm_record = ArchiveEvaluationRecord(
            artifact_id=warm_artifact_id,
            metrics=warm_start["metrics"],
            failure_type="none",
            diagnostics=f"Warm-started from {warm_start['result_dir']}",
            confidence=1.0,
            revision_signal="Continue improving this archived best artifact.",
            is_passed=True,
        )
        archive.update([warm_artifact], [warm_record])
        metric_text = ", ".join(
            f"{k}={v:.6g}" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in archive.snapshot().get("best_metrics", {}).items()
        )
        descriptor = "Warm-started archived best"
        if metric_text:
            descriptor += f"; metrics: {metric_text}"
        state.update_from_archive(
            ArtifactPreview(
                artifact_id=warm_artifact_id,
                score=float(warm_start["score"]),
                is_feasible=True,
                summary_descriptor=descriptor,
            ),
            archive.snapshot().get("top_k_exemplars", [])[:1],
        )
        setattr(state, "current_artifact", warm_start["code"])
        print(
            f"[WARM] Loaded previous best from {warm_start['result_dir']} "
            f"({objective_key}={warm_start['score']:.6g})"
        )
        polish_artifact = _build_uncertainty_polish_artifact(
            cfg=cfg,
            benchmark_task=benchmark_task,
            warm_start=warm_start,
        )
        if polish_artifact is not None:
            print("[POLISH] Evaluating deterministic numerical polish seed...")
            polish_traces = harness.execute([polish_artifact])
            polish_records = evaluator.evaluate([polish_artifact], polish_traces, problem)
            archive.update([polish_artifact], polish_records)
            update_state_glue(
                state=state,
                archive=archive,
                memory=memory,
                records=polish_records,
                obj_key=objective_key,
            )
            polish_record = polish_records[0] if polish_records else None
            if polish_record and polish_record.is_passed:
                score = polish_record.metrics.get(objective_key)
                print(f"[POLISH] Seed passed ({objective_key}={score}).")
            else:
                diagnostics = polish_record.diagnostics if polish_record else "no record"
                print(f"[POLISH] Seed did not pass: {diagnostics[:160]}")
    elif benchmark_task is not None and bool(
        (cfg.get("runner", {}) or {}).get("seed_initial_evaluation", True)
    ):
        print("[SEED] Evaluating benchmark initial artifact as incumbent fallback...")
        seed_artifact = ArchiveGeneratedArtifact(
            "initial_seed",
            initial_code,
            {"parent_id": "root", "operator": "initial_seed"},
        )
        seed_traces = harness.execute([seed_artifact])
        seed_records = evaluator.evaluate([seed_artifact], seed_traces, problem)
        archive.update([seed_artifact], seed_records)
        update_state_glue(
            state=state,
            archive=archive,
            memory=memory,
            records=seed_records,
            obj_key=objective_key,
        )
        seed_record = seed_records[0] if seed_records else None
        if seed_record and seed_record.is_passed:
            score = seed_record.metrics.get(objective_key)
            print(f"[SEED] Initial artifact passed ({objective_key}={score}).")
        else:
            diagnostics = seed_record.diagnostics if seed_record else "no record"
            print(f"[SEED] Initial artifact did not pass: {diagnostics[:160]}")

    # ---- 进步曲线跟踪: 每次 runtime_score 创新高时记录一条 ----
    # 条目形如 {"iter": 3, "runtime_score": 0.67, "abs_metric": "sum_radii", "abs_value": 1.74, "artifact_id": "..."}
    improvement_history: list[dict] = []
    maximize_objective = bool(cfg.get("archive", {}).get("maximize", True))
    last_best_score: float = -float("inf") if maximize_objective else float("inf")

    # 候选的"绝对值"指标名 (按优先级): 第一命中即作为主显示值
    ABS_METRIC_CANDIDATES = [
        "sum_radii",        # math/circle_packing
        "combined_score",   # 多数 ADRS / math benchmark
        "target_ratio",
        "accuracy",
        "reward",
        "value",
    ]

    def _pick_abs_metric(metrics: dict) -> tuple[Optional[str], Optional[float]]:
        for k in ABS_METRIC_CANDIDATES:
            if k in metrics and isinstance(metrics[k], (int, float)):
                return k, float(metrics[k])
        # 兜底: 第一个跟 *_score 无关的数值字段
        for k, v in metrics.items():
            if k.endswith("_score") or k in ("score", "execution_time_ms"):
                continue
            if isinstance(v, (int, float)):
                return k, float(v)
        return None, None

    while not state.remaining_budget.is_exhausted() and iteration <= max_iterations:
        print(f"\n[Iteration {iteration}/{max_iterations}]")

        # 1. Controller
        action = controller.plan(state)
        print(f"  |- Controller decided: {action.operator_choice} "
              f"(Branch width: {action.branch_width})")

        if action.operator_choice == "TRIGGER_STRATEGY_ADAPTER":
            print("  |- [WARNING] Stagnation threshold hit! Invoking Strategy Adapter...")
            report = strategy_adapter.adapt(memory, controller.consecutive_no_gain)
            print(f"  |- Strategy Adjusted: {report}")
            controller.consecutive_no_gain = 0
            iteration += 1
            continue

        # 2. Query Engine
        evidence = query_engine.run(action.query_plan)
        print(f"  |- Retrieved {len(evidence)} evidence pieces.")

        operator_obj = operator_library.get_operator(action.operator_choice) \
            or operator_library.get_operator("local_revision")
        operator_instruction = getattr(operator_obj, "instruction_template", "")

        # 3. Context Assembler
        state_dict = asdict(state)
        state_dict["current_artifact"] = archive.get_best_content() or initial_code
        state_dict["objective"] = problem.task_description
        action_dict = asdict(action)
        action_dict["operator_instruction"] = operator_instruction
        if baseline_profile is not None:
            action_dict["baseline_profile"] = baseline_profile.to_dict()

        context = context_assembler.build(
            state=state_dict,
            query_results=evidence,
            archive_snapshot=archive.snapshot(),
            memory_snapshot=filter_memory_snapshot(baseline_profile, memory.snapshot()),
            action_package=action_dict,
        )

        _snap = archive.snapshot()
        if _snap.get("best_score") not in (None, -float("inf")):
            print(f"  |- [CTX] Injected best_score into prompt: {_snap['best_score']:.4f}")

        # 4. Synthesizer
        print(f"  |- Calling LLM to synthesize {action.branch_width} candidates...")
        artifacts = synthesizer.generate(
            operator=operator_obj, context=context, n=action.branch_width
        )

        # Clean up markdown code fences
        from benchmark_adapter import BenchmarkTaskHarness as _BH
        for art in artifacts:
            art.content = _BH._clean_llm_output(art.content)

        print("  |- Synthesis complete. Candidate previews:")
        for i, art in enumerate(artifacts):
            preview = art.content.strip()[:60].replace("\n", "\\n")
            print(f"     |- [C{i+1}] {preview}...")

        # 5. Execution Harness
        print("  |- Executing in Sandbox...")
        traces = harness.execute(artifacts)

        if benchmark_task is None:
            print("  |- Execution Traces Analysis:")
            for i, trace in enumerate(traces):
                if trace.is_completed:
                    if "Sum of Radii:" in trace.stdout:
                        try:
                            score_str = trace.stdout.split("Sum of Radii:")[1].split()[0].strip()
                            trace.metrics["score"] = float(score_str)
                            print(f"     |- [T{i+1}] [OK] Score={trace.metrics['score']}")
                        except Exception:
                            trace.metrics["score"] = -float("inf")
                            out_preview = trace.stdout.strip()[:50].replace("\n", "\\n")
                            print(f"     |- [T{i+1}] [WARN] parse fail | STDOUT: {out_preview}...")
                    else:
                        trace.metrics["score"] = -float("inf")
                        out_preview = trace.stdout.strip()[:50].replace("\n", "\\n")
                        print(f"     |- [T{i+1}] [WARN] missing 'Sum of Radii:' | STDOUT: {out_preview}...")
                else:
                    trace.metrics["score"] = -float("inf")
                    err_preview = trace.stderr.strip()[:80].replace("\n", "\\n")
                    print(f"     |- [T{i+1}] [FAIL] STDERR: {err_preview}...")
        else:
            print("  |- Execution complete (metrics filled by BenchmarkTaskHarness).")
            for i, trace in enumerate(traces):
                status = "OK" if trace.is_completed and trace.return_code == 0 else "FAIL"
                score = trace.metrics.get("score", trace.metrics)
                print(f"     |- [T{i+1}] [{status}] metrics preview: {score}")

        avg_time = sum(t.execution_time_ms for t in traces) / len(traces) if traces else 0
        print(f"  |- Execution complete. ({avg_time:.1f}ms avg)")

        # 6. Evaluation Stack
        print("  |- Evaluating traces (5-layer gates)...")
        records = evaluator.evaluate(artifacts=artifacts, traces=traces, problem=problem)
        passed_count = sum(1 for r in records if r.is_passed)
        print(f"  |- Evaluation: {passed_count}/{len(records)} passed all gates.")
        for i, rec in enumerate(records):
            if rec.is_passed:
                print(f"     |- [Eval {i+1}] [PASSED] {objective_key}={rec.metrics.get(objective_key)}")
            else:
                fail_reason = rec.failure_type.value.upper()
                diag_preview = rec.diagnostics.strip()[:80].replace("\n", "\\n")
                print(f"     |- [Eval {i+1}] [FAIL@{fail_reason}] {diag_preview}...")

        # 7. Update Archive / Memory / State
        archive.update(artifacts, records)
        memory.write(action, evidence, artifacts, traces, records)
        update_state_glue(state, archive, memory, records, objective_key)

        state.consume_budget(llm_calls=action.branch_width, sandbox_runs=action.branch_width)
        state.record_operator_execution(action.operator_choice, is_success=(passed_count > 0))

        best = archive.best_artifact
        if best:
            best_score = best.record.metrics.get(objective_key, -float("inf"))
            abs_key, abs_val = _pick_abs_metric(best.record.metrics)

            abs_str = (
                f"{abs_key}={abs_val:.6g}" if abs_key is not None else "abs=N/A"
            )
            print(f"  |- [BEST] {objective_key}={best_score:.6g} | {abs_str} "
                  f"| id={best.artifact_id}")

            # 判断是否"进步": 按 archive.maximize 支持最大化或最小化
            is_valid_best = (
                best_score is not None
                and best_score not in (-float("inf"), float("inf"))
                and best_score == best_score
                and bool(best.record.is_passed)
            )
            is_improved = (
                best_score > last_best_score + 1e-12
                if maximize_objective
                else best_score < last_best_score - 1e-12
            ) if is_valid_best else False
            if is_improved:
                improvement_history.append({
                    "iter": iteration,
                    "runtime_score": best_score,
                    "abs_metric": abs_key,
                    "abs_value": abs_val,
                    "artifact_id": best.artifact_id,
                })
                initial_sentinel = -float("inf") if maximize_objective else float("inf")
                delta = (best_score - last_best_score
                         if last_best_score != initial_sentinel else best_score)
                print(f"  |- [IMPROVED @ iter {iteration}] "
                      f"{objective_key}: {last_best_score if last_best_score != initial_sentinel else str(initial_sentinel)} "
                      f"-> {best_score:.6g} (Δ={delta:.6g}) | {abs_str}")
                last_best_score = best_score

        iteration += 1

    print("\n" + "=" * 50)
    if state.remaining_budget.is_exhausted():
        stop_reason = (
            "budget exhausted "
            f"(llm_calls={state.remaining_budget.llm_calls}, "
            f"sandbox_runs={state.remaining_budget.sandbox_runs}, "
            f"api_cost_usd={state.remaining_budget.api_cost_usd:.4g})"
        )
    elif iteration > max_iterations:
        stop_reason = f"reached max_iterations={max_iterations}"
    else:
        stop_reason = "loop stopped"
    print(f"[OK] Optimization Loop Finished. Stop reason: {stop_reason}.")

    # ---- 进步曲线汇总 ----
    print("\n=== Improvement History ===")
    if not improvement_history:
        print("  (no passed improvement recorded)")
    else:
        print(f"  {'iter':>5} | {'runtime_score':>14} | {'abs_metric':<15} | {'abs_value':>12} | artifact_id")
        print("  " + "-" * 78)
        for row in improvement_history:
            abs_metric = row["abs_metric"] or "N/A"
            abs_val = row["abs_value"]
            abs_val_str = f"{abs_val:.6g}" if isinstance(abs_val, (int, float)) else "N/A"
            print(f"  {row['iter']:>5} | {row['runtime_score']:>14.6g} | "
                  f"{abs_metric:<15} | {abs_val_str:>12} | {row['artifact_id']}")
        first, last = improvement_history[0], improvement_history[-1]
        print(f"  Total improvements: {len(improvement_history)} | "
              f"first @ iter {first['iter']} -> final @ iter {last['iter']} "
              f"({last['runtime_score']:.6g})")

    # ---- 打包回传给 main() 的结果 (供 result archive 使用) ----
    best_final_code = archive.get_best_content() or initial_code
    best_obj = archive.best_artifact
    best_bundle = {
        "code": best_final_code,
        "metrics": dict(best_obj.record.metrics) if best_obj else {},
        "artifact_id": best_obj.artifact_id if best_obj else "none",
        "is_passed": bool(best_obj.record.is_passed) if best_obj else False,
        "objective_key": objective_key,
        "improvement_history": improvement_history,
    }
    return best_bundle


# ==========================================
# 5. CLI
# ==========================================
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="OptiCom YAML-driven runner")
    parser.add_argument(
        "--config", "-c", default=None,
        help="Path to YAML config file (see example/*.yaml).",
    )
    parser.add_argument(
        "--benchmark", "-b", default=None,
        help="Override yaml benchmark.task_id (and force benchmark.enabled=true).",
    )
    parser.add_argument(
        "--benchmarks-root", default=None,
        help="Override yaml benchmark.root.",
    )
    parser.add_argument(
        "--max-iter", type=int, default=None,
        help="Override yaml runner.max_iterations.",
    )
    parser.add_argument(
        "--baseline", default=None,
        help=(
            "Run a named baseline profile instead of the full framework. "
            f"Choices include: {', '.join(list_baseline_names())}."
        ),
    )
    parser.add_argument(
        "--no-warm-start",
        action="store_true",
        help="Disable loading result/<baseline>/<task>/<model>/best_program as the initial incumbent.",
    )
    parser.add_argument(
        "--no-seed-initial",
        action="store_true",
        help="Disable evaluating the benchmark initial artifact before the first optimization iteration.",
    )
    parser.add_argument(
        "--list-benchmarks", action="store_true",
        help="List all benchmark task_ids under yaml benchmark.root and exit.",
    )
    parser.add_argument(
        "--list-baselines", action="store_true",
        help="List baseline profiles and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    if args.list_baselines:
        print(profile_catalog_json())
        raise SystemExit(0)

    if not args.config:
        raise SystemExit("--config is required unless --list-baselines is used.")

    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    # ---- CLI 覆盖 yaml ----
    if args.baseline is not None:
        cfg["runner"]["baseline"] = args.baseline
    if args.benchmark is not None:
        cfg["benchmark"]["enabled"] = True
        cfg["benchmark"]["task_id"] = args.benchmark
    if args.benchmarks_root is not None:
        cfg["benchmark"]["root"] = args.benchmarks_root
    if args.max_iter is not None:
        cfg["runner"]["max_iterations"] = args.max_iter
    if args.no_warm_start:
        cfg["runner"]["warm_start_from_result"] = False
    if args.no_seed_initial:
        cfg["runner"]["seed_initial_evaluation"] = False

    # ---- 模式 1: list benchmarks ----
    if args.list_benchmarks:
        root = _resolve_path(
            cfg["benchmark"].get("root") or "./benchmarks",
            cfg["_meta"]["config_dir"],
        )
        print(f"[Listing benchmarks under: {root}]")
        reg = BenchmarkRegistry(root)
        total = 0
        for cat, ts in reg.list_by_category().items():
            print(f"[{cat}] ({len(ts)})")
            for t in ts:
                print(f"  - {t}")
            total += len(ts)
        print(f"[Total] {total} benchmark task(s).")
        raise SystemExit(0)

    # ---- 模式 2: 绑定 benchmark ----
    benchmark_task: Optional[BenchmarkTask] = None
    task_description: str
    if cfg["benchmark"].get("enabled"):
        root = _resolve_path(
            cfg["benchmark"].get("root") or "./benchmarks",
            cfg["_meta"]["config_dir"],
        )
        task_id = cfg["benchmark"].get("task_id")
        if not task_id:
            raise ValueError(
                "benchmark.enabled=true 但未配置 benchmark.task_id。"
                "请在 yaml 中设置 task_id，或者运行 --list-benchmarks 查看候选。"
            )
        reg = BenchmarkRegistry(root)
        benchmark_task = reg.get(task_id)
        task_description = benchmark_task.get_task_description()
        print(f"[OK] Loaded benchmark task: {_benchmark_display_name(benchmark_task, cfg)}")
    else:
        # ---- 模式 3: 使用 yaml 里的 inline task_description ----
        task_description = cfg["runner"].get("task_description")
        if not task_description or not str(task_description).strip():
            raise ValueError(
                "benchmark.enabled=false 时必须在 yaml 的 runner.task_description "
                "里填入任务描述，或把 benchmark.enabled 改成 true。"
            )
        task_description = str(task_description).strip()
        print("[OK] Using inline runner.task_description from YAML.")

    t0 = time.time()
    best_bundle = run_optimization_loop(
        cfg=cfg,
        task=task_description,
        benchmark_task=benchmark_task,
    )
    elapsed = time.time() - t0

    if best_bundle is None:
        print(f"[DONE] Total elapsed: {elapsed:.1f}s | Initialization failed.")
        raise SystemExit(1)

    best_final_code = best_bundle["code"]
    print("\nFINAL OPTIMIZED ARTIFACT")
    print("-" * 40)
    print(best_final_code)
    print("-" * 40)

    # ---- Result Archive: 若刷新任务历史最佳, 则落盘 ----
    task_slug = _task_slug(benchmark_task, cfg_path, cfg)
    llm_name = _llm_name(cfg)
    try:
        is_new_best, out_dir, prev_score = save_result_if_best(
            llm_name=llm_name,
            task_slug=task_slug,
            benchmark_task=benchmark_task,
            cfg=cfg,
            cfg_path=cfg_path,
            best_record_metrics=best_bundle["metrics"],
            best_code=best_final_code,
            best_artifact_id=best_bundle["artifact_id"],
            best_is_passed=bool(best_bundle.get("is_passed", False)),
            objective_key=best_bundle["objective_key"],
            maximize=bool(cfg.get("archive", {}).get("maximize", True)),
            improvement_history=best_bundle["improvement_history"],
            elapsed_sec=elapsed,
        )
    except Exception as e:  # noqa: BLE001
        print(f"[WARN] Failed to write result archive: {e}")
        is_new_best, out_dir, prev_score = False, _result_dir(
            baseline_name=_baseline_name(cfg),
            task_slug=task_slug,
            llm_name=llm_name,
        ), None

    current_score = best_bundle["metrics"].get(best_bundle["objective_key"])
    if is_new_best:
        prev_str = f"{prev_score:.6g}" if prev_score is not None else "N/A (first run)"
        cur_str = f"{current_score:.6g}" if isinstance(current_score, (int, float)) else "N/A"
        print(f"\n[ARCHIVE] NEW BEST saved -> {out_dir}")
        print(f"          task_slug: {task_slug}")
        print(f"          runtime_score: {prev_str} -> {cur_str}")
    else:
        prev_str = f"{prev_score:.6g}" if prev_score is not None else "N/A"
        cur_str = f"{current_score:.6g}" if isinstance(current_score, (int, float)) else "N/A"
        print(f"\n[ARCHIVE] No improvement over previous best "
              f"(task_slug={task_slug}, prev={prev_str}, this_run={cur_str}); "
              f"archive not updated.")

    print(f"[DONE] Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
