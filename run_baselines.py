"""
Batch runner for evaluating baseline profiles under the OptiCom framework.

Example:
    python3 run_baselines.py \
        --config example/benchmark_arc.yaml \
        --benchmark circle_packing \
        --baselines textgrad,opro,reflexion,gepa \
        --max-iter 5

For a dependency-only smoke test without model calls:
    python3 run_baselines.py -c example/benchmark_arc.yaml -b circle_packing \
        --baselines textgrad --max-iter 1 --llm-provider mock --llm-model mock
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_FILE = Path(__file__).resolve()
V1_ROOT = THIS_FILE.parent
SRC_DIR = V1_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from baseline_profiles import (  # noqa: E402
    get_baseline_profile,
    list_baseline_names,
    normalize_baseline_name,
    profile_catalog_json,
)
from benchmark_adapter import BenchmarkRegistry, BenchmarkTask  # noqa: E402
from run import (  # noqa: E402
    _code_filename,
    _llm_name,
    _resolve_path,
    _safe_slug,
    _task_slug,
    load_config,
    run_optimization_loop,
    save_result_if_best,
)


ABS_METRIC_CANDIDATES = [
    "sum_radii",
    "combined_score",
    "target_ratio",
    "accuracy",
    "reward",
    "value",
]


def _parse_csv(value: str) -> List[str]:
    if value.lower() in {"all", "*"}:
        return list_baseline_names()
    return [part.strip() for part in value.split(",") if part.strip()]


def _pick_abs_metric(metrics: Dict[str, Any]) -> tuple[Optional[str], Optional[float]]:
    for key in ABS_METRIC_CANDIDATES:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    for key, value in metrics.items():
        if key.endswith("_score") or key in {"score", "execution_time_ms"}:
            continue
        if isinstance(value, (int, float)):
            return key, float(value)
    return None, None


def _resolve_benchmark_task(cfg: Dict[str, Any]) -> tuple[Optional[BenchmarkTask], str]:
    if not cfg["benchmark"].get("enabled"):
        task_description = str(cfg["runner"].get("task_description") or "").strip()
        if not task_description:
            raise ValueError("Inline runs require runner.task_description.")
        return None, task_description

    root = _resolve_path(
        cfg["benchmark"].get("root") or "./benchmarks",
        cfg["_meta"]["config_dir"],
    )
    registry = BenchmarkRegistry(root)
    task = registry.get(cfg["benchmark"]["task_id"])
    return task, task.get_task_description()


def _configure_overrides(
    cfg: Dict[str, Any],
    args: argparse.Namespace,
    baseline: str,
) -> Dict[str, Any]:
    out = copy.deepcopy(cfg)
    out["runner"]["baseline"] = baseline
    if args.max_iter is not None:
        out["runner"]["max_iterations"] = int(args.max_iter)
    if args.benchmark:
        out["benchmark"]["enabled"] = True
        out["benchmark"]["task_id"] = args.benchmark
    if args.benchmarks_root:
        out["benchmark"]["root"] = args.benchmarks_root

    if args.llm_provider:
        out["llm"]["provider"] = args.llm_provider
    if args.llm_model:
        out["llm"]["model"] = args.llm_model
    if args.llm_api_key is not None:
        out["llm"]["api_key"] = args.llm_api_key or None
    if args.llm_base_url is not None:
        out["llm"]["base_url"] = args.llm_base_url or None
    return out


def _write_artifacts(
    out_dir: Path,
    baseline: str,
    benchmark_task: Optional[BenchmarkTask],
    bundle: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    baseline_dir = out_dir / _safe_slug(baseline)
    baseline_dir.mkdir(parents=True, exist_ok=True)

    code_name = _code_filename(benchmark_task)
    (baseline_dir / code_name).write_text(bundle.get("code", ""), encoding="utf-8")

    metrics = dict(bundle.get("metrics", {}) or {})
    objective_key = bundle.get("objective_key")
    runtime_score = metrics.get(objective_key)
    abs_metric, abs_value = _pick_abs_metric(metrics)

    summary = {
        "baseline": baseline,
        "objective_key": objective_key,
        "runtime_score": float(runtime_score) if isinstance(runtime_score, (int, float)) else None,
        "abs_metric": abs_metric,
        "abs_value": abs_value,
        "artifact_id": bundle.get("artifact_id"),
        "best_passed": bool(bundle.get("is_passed", False)),
        "elapsed_sec": round(elapsed, 2),
        "improvement_history": bundle.get("improvement_history", []),
        "all_metrics": metrics,
        "code_file": str(baseline_dir / code_name),
        "log_file": str(baseline_dir / "run.log"),
    }
    (baseline_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return summary


def _archive_best_result(
    cfg: Dict[str, Any],
    benchmark_task: Optional[BenchmarkTask],
    bundle: Dict[str, Any],
    elapsed: float,
) -> Dict[str, Any]:
    cfg_path = Path(cfg["_meta"]["config_path"]).resolve()
    task_slug = _task_slug(benchmark_task, cfg_path, cfg)
    llm_name = _llm_name(cfg)
    is_new_best, archive_dir, previous_score = save_result_if_best(
        llm_name=llm_name,
        task_slug=task_slug,
        benchmark_task=benchmark_task,
        cfg=cfg,
        cfg_path=cfg_path,
        best_record_metrics=dict(bundle.get("metrics", {}) or {}),
        best_code=bundle.get("code", ""),
        best_artifact_id=bundle.get("artifact_id", "none"),
        best_is_passed=bool(bundle.get("is_passed", False)),
        objective_key=bundle.get("objective_key", "runtime_score"),
        maximize=bool(cfg.get("archive", {}).get("maximize", True)),
        improvement_history=bundle.get("improvement_history", []),
        elapsed_sec=elapsed,
    )
    return {
        "archive_dir": str(archive_dir),
        "archive_new_best": bool(is_new_best),
        "archive_previous_score": previous_score,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run baseline profiles as framework configurations.")
    parser.add_argument("--config", "-c", default=None, help="Base YAML config.")
    parser.add_argument("--benchmark", "-b", default=None, help="Override benchmark.task_id.")
    parser.add_argument("--benchmarks-root", default=None, help="Override benchmark.root.")
    parser.add_argument(
        "--baselines",
        default="all",
        help="Comma-separated baselines or 'all'. Example: textgrad,opro,reflexion",
    )
    parser.add_argument("--max-iter", type=int, default=None, help="Override runner.max_iterations.")
    parser.add_argument("--out-dir", default=None, help="Directory for logs and summaries.")
    parser.add_argument("--llm-provider", default=None, help="Override llm.provider, e.g. mock.")
    parser.add_argument("--llm-model", default=None, help="Override llm.model, e.g. mock.")
    parser.add_argument("--llm-api-key", default=None, help="Override llm.api_key; empty string means None.")
    parser.add_argument("--llm-base-url", default=None, help="Override llm.base_url.")
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop the suite if one baseline fails.",
    )
    parser.add_argument(
        "--list-baselines",
        action="store_true",
        help="Print available baseline names and exit.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.list_baselines:
        print(profile_catalog_json())
        return
    if not args.config:
        raise SystemExit("--config is required unless --list-baselines is used.")

    base_cfg = load_config(Path(args.config).resolve())
    baselines = [normalize_baseline_name(name) for name in _parse_csv(args.baselines)]
    for baseline in baselines:
        # Validate early so typos fail before model calls start.
        get_baseline_profile(baseline)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out_dir or (V1_ROOT / "baseline_runs" / timestamp)).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    suite_summary = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "config": str(Path(args.config).resolve()),
        "benchmark": args.benchmark or base_cfg.get("benchmark", {}).get("task_id"),
        "baselines": baselines,
        "runs": [],
    }

    for baseline in baselines:
        cfg = _configure_overrides(base_cfg, args, baseline)
        benchmark_task, task_description = _resolve_benchmark_task(cfg)
        baseline_dir = out_dir / _safe_slug(baseline)
        baseline_dir.mkdir(parents=True, exist_ok=True)
        log_path = baseline_dir / "run.log"

        print(f"[RUN] baseline={baseline} -> {log_path}")
        start = time.time()
        try:
            with log_path.open("w", encoding="utf-8") as log_fp:
                with contextlib.redirect_stdout(log_fp), contextlib.redirect_stderr(log_fp):
                    bundle = run_optimization_loop(
                        cfg=cfg,
                        task=task_description,
                        benchmark_task=benchmark_task,
                    )
            elapsed = time.time() - start
            if bundle is None:
                raise RuntimeError("Optimization loop returned None.")
            summary = _write_artifacts(out_dir, baseline, benchmark_task, bundle, elapsed)
            archive_summary = _archive_best_result(cfg, benchmark_task, bundle, elapsed)
            summary.update(archive_summary)
            status = "ok" if summary["best_passed"] else "no_pass"
            suite_summary["runs"].append({**summary, "status": status})
            archive_tag = "new_best" if summary["archive_new_best"] else "not_archived"
            print(
                f"[{status.upper()}] baseline={baseline} score={summary['runtime_score']} "
                f"{summary['abs_metric']}={summary['abs_value']} elapsed={summary['elapsed_sec']}s "
                f"archive={archive_tag}"
            )
        except Exception as exc:  # noqa: BLE001
            elapsed = time.time() - start
            error_summary = {
                "baseline": baseline,
                "status": "error",
                "error": str(exc),
                "elapsed_sec": round(elapsed, 2),
                "log_file": str(log_path),
            }
            suite_summary["runs"].append(error_summary)
            print(f"[ERROR] baseline={baseline}: {exc}")
            if args.stop_on_error:
                break

        (out_dir / "suite_summary.json").write_text(
            json.dumps(suite_summary, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    print(f"[DONE] Baseline suite summary -> {out_dir / 'suite_summary.json'}")


if __name__ == "__main__":
    main()
