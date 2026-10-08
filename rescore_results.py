"""
rescore_results.py
==================

Re-evaluate archived best programs with the current benchmark evaluator.

This is useful after an evaluator bug fix: historical result.json files may
still contain scores produced by the old evaluator. By default this script is
dry-run only and prints a fresh ranking. Use --write to update result.json.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


V1_ROOT = Path(__file__).resolve().parent
SRC_DIR = V1_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from artifact_synthesizer import GeneratedArtifact  # noqa: E402
from benchmark_adapter import BenchmarkRegistry, BenchmarkTaskHarness  # noqa: E402


@dataclass
class ResultEntry:
    json_path: Path
    program_path: Path
    baseline: str
    task_slug: str
    task_id: str
    model: str
    data: Dict[str, Any]


@dataclass
class RescoreRow:
    entry: ResultEntry
    old_score: Optional[float]
    new_score: float
    metrics: Dict[str, float]
    passed: bool
    error: str = ""


def _iter_result_json_files(result_root: Path) -> Iterable[Path]:
    yield from sorted(result_root.glob("*/*/*/result.json"))


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _infer_task_id(task_slug: str, data: Dict[str, Any]) -> str:
    task_id = data.get("benchmark_task_id")
    if isinstance(task_id, str) and task_id.strip():
        return task_id.strip()
    if "__" in task_slug:
        return task_slug.split("__", 1)[1]
    return task_slug


def _find_program_path(json_path: Path, data: Dict[str, Any]) -> Optional[Path]:
    code_file = data.get("code_file")
    if isinstance(code_file, str) and code_file.strip():
        candidate = json_path.parent / code_file
        if candidate.exists():
            return candidate

    candidates = sorted(json_path.parent.glob("best_program.*"))
    return candidates[0] if candidates else None


def collect_entries(
    result_root: Path,
    *,
    baseline: Optional[str] = None,
    task: Optional[str] = None,
    model: Optional[str] = None,
) -> List[ResultEntry]:
    entries: List[ResultEntry] = []
    for json_path in _iter_result_json_files(result_root):
        try:
            data = json.loads(json_path.read_text(encoding="utf-8"))
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] failed to read {json_path}: {exc}")
            continue

        row_baseline = str(data.get("baseline") or json_path.parents[2].name)
        row_task_slug = str(data.get("task_slug") or json_path.parents[1].name)
        row_model = str(data.get("model_slug") or json_path.parent.name)
        row_task_id = _infer_task_id(row_task_slug, data)

        if baseline and row_baseline != baseline:
            continue
        if model and row_model != model:
            continue
        if task and task not in {row_task_slug, row_task_id}:
            continue

        program_path = _find_program_path(json_path, data)
        if program_path is None:
            print(f"[WARN] no best_program.* found next to {json_path}")
            continue

        entries.append(
            ResultEntry(
                json_path=json_path,
                program_path=program_path,
                baseline=row_baseline,
                task_slug=row_task_slug,
                task_id=row_task_id,
                model=row_model,
                data=data,
            )
        )
    return entries


def rescore_entry(
    entry: ResultEntry,
    registry: BenchmarkRegistry,
    *,
    timeout_seconds: int,
    python_executable: str,
) -> RescoreRow:
    old_score = _as_float(entry.data.get("runtime_score"))
    try:
        task = registry.get(entry.task_id)
        harness = BenchmarkTaskHarness(
            task=task,
            timeout_seconds=timeout_seconds,
            python_executable=python_executable,
            isolate_evaluator=True,
        )
        artifact = GeneratedArtifact(
            artifact_id=f"rescore:{entry.baseline}:{entry.task_slug}:{entry.model}",
            content=entry.program_path.read_text(encoding="utf-8", errors="ignore"),
        )
        trace = harness.run([artifact])[0]
        metrics = {
            key: float(value)
            for key, value in trace.metrics.items()
            if isinstance(value, (int, float)) and math.isfinite(float(value))
        }
        new_score = float(metrics.get("score", 0.0))
        error = trace.stderr.strip()
        return RescoreRow(
            entry=entry,
            old_score=old_score,
            new_score=new_score,
            metrics=metrics,
            passed=(trace.return_code == 0),
            error=error,
        )
    except Exception as exc:  # noqa: BLE001
        return RescoreRow(
            entry=entry,
            old_score=old_score,
            new_score=0.0,
            metrics={"score": 0.0},
            passed=False,
            error=str(exc),
        )


def write_rescore(row: RescoreRow, *, backup: bool = True) -> None:
    json_path = row.entry.json_path
    data = dict(row.entry.data)
    if backup:
        backup_path = json_path.with_suffix(json_path.suffix + ".bak")
        if not backup_path.exists():
            backup_path.write_text(
                json.dumps(row.entry.data, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    data["runtime_score"] = row.new_score
    data["all_metrics"] = row.metrics
    data["best_is_passed"] = bool(row.passed)
    data["rescored_at"] = datetime.now().isoformat(timespec="seconds")
    data["rescored_from_runtime_score"] = row.old_score
    data["rescored_program_path"] = str(row.entry.program_path)
    data.pop("rescored_with_evaluator", None)
    data["objective_key"] = "score"
    if "combined_score" in row.metrics:
        data["abs_metric"] = "combined_score"
        data["abs_value"] = row.metrics["combined_score"]
    elif "score" in row.metrics:
        data["abs_metric"] = "score"
        data["abs_value"] = row.metrics["score"]
    if row.error:
        data["rescore_error"] = row.error[-2000:]
    else:
        data.pop("rescore_error", None)

    json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def print_rows(rows: List[RescoreRow]) -> None:
    if not rows:
        print("No rows to show.")
        return

    rows = sorted(rows, key=lambda row: row.new_score, reverse=True)
    print(
        f"{'Rank':>4} | {'Baseline':<18} | {'Task':<24} | {'Model':<24} | "
        f"{'Old':>11} | {'New':>11} | {'Success':>9}"
    )
    print("-" * 112)
    for idx, row in enumerate(rows, 1):
        old = "-" if row.old_score is None else f"{row.old_score:.6f}"
        success = row.metrics.get("success_rate")
        success_str = "-" if success is None else f"{success:.3f}"
        print(
            f"{idx:>4} | {row.entry.baseline:<18} | {row.entry.task_slug:<24} | "
            f"{row.entry.model:<24} | {old:>11} | {row.new_score:>11.6f} | "
            f"{success_str:>9}"
        )
    failures = [row for row in rows if row.error]
    if failures:
        print("\nWarnings:")
        for row in failures[:10]:
            print(f"  - {row.entry.baseline}/{row.entry.task_slug}: {row.error.splitlines()[0]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", default=str(V1_ROOT / "result"))
    parser.add_argument("--benchmark-root", default=str(V1_ROOT / "benchmarks"))
    parser.add_argument("--baseline", default=None)
    parser.add_argument("--task", default=None, help="Task id or task slug, e.g. prism or ADRS__prism")
    parser.add_argument("--model", default=None)
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--write", action="store_true", help="Update result.json files in place")
    parser.add_argument("--no-backup", action="store_true", help="Do not create result.json.bak on --write")
    args = parser.parse_args()

    result_root = Path(args.result_root).resolve()
    benchmark_root = Path(args.benchmark_root).resolve()
    if not result_root.exists():
        raise SystemExit(f"result root not found: {result_root}")

    entries = collect_entries(
        result_root,
        baseline=args.baseline,
        task=args.task,
        model=args.model,
    )
    if not entries:
        print("No matching archived results found.")
        return

    registry = BenchmarkRegistry(benchmark_root)
    print(
        f"Rescoring {len(entries)} archived result(s) with current evaluators "
        f"from {benchmark_root}"
    )
    if not args.write:
        print("Mode: dry-run (use --write to update result.json)\n")

    rows = [
        rescore_entry(
            entry,
            registry,
            timeout_seconds=args.timeout,
            python_executable=args.python,
        )
        for entry in entries
    ]

    if args.write:
        for row in rows:
            write_rescore(row, backup=not args.no_backup)
        print(f"Updated {len(rows)} result.json file(s).\n")

    print_rows(rows)


if __name__ == "__main__":
    main()
