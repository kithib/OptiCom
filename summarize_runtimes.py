"""
Summarize OptiCom runtime records.

Scans:
  - result/<baseline>/<task>/<model>/result.json        (new best-result layout)
  - result/<model>/<task>/result.json                   (legacy best-result layout)
  - baseline_runs/<timestamp>/suite_summary.json        (batch baseline suites)

Default output is sorted from shortest to longest elapsed time.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Optional


V1_ROOT = Path(__file__).resolve().parent


@dataclass
class RuntimeRow:
    elapsed_sec: float
    task_slug: str
    task_id: str
    baseline: str
    model: str
    source: str
    status: str
    runtime_score: Optional[float]
    path: str
    timestamp: str = ""


FIELDNAMES = list(RuntimeRow.__dataclass_fields__.keys())


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _model_from_data(data: dict[str, Any], fallback: str) -> str:
    if data.get("model_slug"):
        return str(data["model_slug"])
    llm = data.get("llm") or {}
    provider = llm.get("provider")
    model = llm.get("model")
    if provider or model:
        return f"{provider or 'unknown'}_{model or 'unknown'}"
    return fallback


def _task_id_from_slug(slug: str) -> str:
    return slug.split("__", 1)[1] if "__" in slug else slug


def _iter_result_json_files(result_root: Path) -> Iterable[Path]:
    if not result_root.exists():
        return []
    # Include both old and new layouts without duplicates.
    files = set(result_root.glob("*/*/result.json"))
    files.update(result_root.glob("*/*/*/result.json"))
    return sorted(files)


def _row_from_result_file(path: Path, result_root: Path) -> Optional[RuntimeRow]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None

    elapsed = _safe_float(data.get("elapsed_sec"))
    if elapsed is None:
        return None

    rel = path.relative_to(result_root)
    parts = rel.parts
    is_new_layout = len(parts) >= 4

    task_slug = str(data.get("task_slug") or (parts[1] if is_new_layout else parts[-2]))
    task_id = str(data.get("benchmark_task_id") or _task_id_from_slug(task_slug))

    if is_new_layout:
        baseline = str(data.get("baseline") or parts[0])
        model = _model_from_data(data, fallback=parts[2])
    else:
        # Legacy layout was result/<model>/<task>/result.json.
        baseline = str(data.get("baseline") or (data.get("runner") or {}).get("baseline") or "opticom")
        model = _model_from_data(data, fallback=parts[0])

    return RuntimeRow(
        elapsed_sec=elapsed,
        task_slug=task_slug,
        task_id=task_id,
        baseline=baseline,
        model=model,
        source="best_result",
        status="best",
        runtime_score=_safe_float(data.get("runtime_score")),
        path=str(path),
        timestamp=str(data.get("timestamp") or ""),
    )


def _row_from_suite_run(
    run: dict[str, Any],
    suite: dict[str, Any],
    suite_file: Path,
) -> Optional[RuntimeRow]:
    elapsed = _safe_float(run.get("elapsed_sec"))
    if elapsed is None:
        return None

    archive_dir = str(run.get("archive_dir") or "")
    archive_path = Path(archive_dir) if archive_dir else None
    task_slug = ""
    model = ""
    if archive_path and len(archive_path.parts) >= 3:
        task_slug = archive_path.parent.name
        model = archive_path.name

    benchmark = str(suite.get("benchmark") or "")
    if not task_slug:
        task_slug = benchmark or "unknown_task"
    task_id = benchmark or _task_id_from_slug(task_slug)

    return RuntimeRow(
        elapsed_sec=elapsed,
        task_slug=task_slug,
        task_id=task_id,
        baseline=str(run.get("baseline") or "unknown_baseline"),
        model=model or "unknown_model",
        source="baseline_suite",
        status=str(run.get("status") or "unknown"),
        runtime_score=_safe_float(run.get("runtime_score")),
        path=str(suite_file),
        timestamp=str(suite.get("created_at") or suite_file.parent.name),
    )


def collect_runtime_rows(
    result_root: Path,
    baseline_runs_root: Path,
    source: str = "all",
) -> list[RuntimeRow]:
    rows: list[RuntimeRow] = []

    if source in {"all", "result"}:
        for path in _iter_result_json_files(result_root):
            row = _row_from_result_file(path, result_root)
            if row:
                rows.append(row)

    if source in {"all", "baseline-runs"} and baseline_runs_root.exists():
        for suite_file in sorted(baseline_runs_root.glob("*/suite_summary.json")):
            try:
                suite = json.loads(suite_file.read_text(encoding="utf-8"))
            except Exception:
                continue
            for run in suite.get("runs", []) or []:
                row = _row_from_suite_run(run, suite, suite_file)
                if row:
                    rows.append(row)

    return rows


def _format_duration(seconds: float) -> str:
    total = int(round(seconds))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    if m:
        return f"{m}m{s:02d}s"
    return f"{s}s"


def _filter_rows(
    rows: list[RuntimeRow],
    task: Optional[str],
    baseline: Optional[str],
    model: Optional[str],
) -> list[RuntimeRow]:
    out = rows
    if task:
        out = [r for r in out if task in r.task_slug or task in r.task_id]
    if baseline:
        out = [r for r in out if r.baseline == baseline]
    if model:
        out = [r for r in out if model in r.model]
    return out


def _print_table(rows: list[RuntimeRow]) -> None:
    print(
        f"{'Sec':>10} | {'Duration':>10} | {'Task':<30} | {'Baseline':<16} | "
        f"{'Model':<28} | {'Source':<14} | {'Status':<8} | {'Score':>10}"
    )
    print("-" * 145)
    for row in rows:
        score = "N/A" if row.runtime_score is None else f"{row.runtime_score:.6g}"
        print(
            f"{row.elapsed_sec:>10.2f} | {_format_duration(row.elapsed_sec):>10} | "
            f"{row.task_id:<30} | {row.baseline:<16} | {row.model:<28} | "
            f"{row.source:<14} | {row.status:<8} | {score:>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize run times by task, sorted shortest to longest.")
    parser.add_argument("--result-root", default=str(V1_ROOT / "result"))
    parser.add_argument("--baseline-runs-root", default=str(V1_ROOT / "baseline_runs"))
    parser.add_argument("--source", choices=["all", "result", "baseline-runs"], default="all")
    parser.add_argument("--task", default=None, help="Filter by task id or task slug substring.")
    parser.add_argument("--baseline", default=None, help="Filter by exact baseline name.")
    parser.add_argument("--model", default=None, help="Filter by model slug substring.")
    parser.add_argument("--top", type=int, default=None, help="Show only the first N rows after sorting.")
    parser.add_argument("--format", choices=["table", "json", "csv"], default="table")
    args = parser.parse_args()

    rows = collect_runtime_rows(
        result_root=Path(args.result_root),
        baseline_runs_root=Path(args.baseline_runs_root),
        source=args.source,
    )
    rows = _filter_rows(rows, args.task, args.baseline, args.model)
    rows.sort(key=lambda r: (r.elapsed_sec, r.task_id, r.baseline, r.model, r.source))
    if args.top is not None:
        rows = rows[: max(0, args.top)]

    if args.format == "json":
        print(json.dumps([asdict(r) for r in rows], indent=2, ensure_ascii=False))
    elif args.format == "csv":
        writer = csv.DictWriter(sys.stdout, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    else:
        if not rows:
            print("No runtime records found.")
            return
        _print_table(rows)


if __name__ == "__main__":
    main()
