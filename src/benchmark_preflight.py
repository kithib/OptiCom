"""
Task-specific benchmark asset preflight.

This module checks data files that evaluator/requirements.txt cannot install.
It is intentionally small and explicit: when a benchmark needs external assets,
we list the exact files and, when possible, the downloader.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import yaml  # type: ignore

try:
    from benchmark_adapter import BenchmarkRegistry, BenchmarkTask
except ImportError:  # pragma: no cover - allows running from repo root
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from benchmark_adapter import BenchmarkRegistry, BenchmarkTask


@dataclass
class AssetRule:
    task_id: str
    required: list[Path]
    download_script: Path | None = None
    download_url: str | None = None
    copy_sources: list[Path] = field(default_factory=list)
    download_target: Path | None = None
    download_files: list[tuple[str, Path]] = field(default_factory=list)


def _resolve_task(config_path: Path, task_id: str) -> BenchmarkTask:
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    root = cfg.get("benchmark", {}).get("root") or "./benchmarks"
    root_path = Path(root)
    if not root_path.is_absolute():
        root_path = (config_path.parent / root_path).resolve()
    return BenchmarkRegistry(str(root_path)).get(task_id)


def _asset_rule_for(task: BenchmarkTask) -> AssetRule | None:
    task_name = task.task_id.lower()
    evaluator_dir = task.evaluator_module_path.parent

    if task_name == "cloudcast":
        base_url = "https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/cloudcast"
        file_paths = [
            "profiles/cost.csv",
            "profiles/throughput.csv",
            "examples/config/intra_aws.json",
            "examples/config/intra_azure.json",
            "examples/config/intra_gcp.json",
            "examples/config/inter_agz.json",
            "examples/config/inter_gaz2.json",
        ]
        return AssetRule(
            task_id=task.task_id,
            required=[evaluator_dir / rel for rel in file_paths],
            download_files=[(f"{base_url}/{rel}", evaluator_dir / rel) for rel in file_paths],
            download_script=evaluator_dir / "download_dataset.sh",
        )

    if task_name == "llm_sql":
        base_url = "https://huggingface.co/datasets/f20180301/adrs-data/resolve/main/llm_sql"
        file_paths = [
            "datasets/movies.csv",
            "datasets/beer.csv",
            "datasets/BIRD.csv",
            "datasets/PDMX.csv",
            "datasets/products.csv",
        ]
        return AssetRule(
            task_id=task.task_id,
            required=[evaluator_dir / rel for rel in file_paths],
            download_files=[(f"{base_url}/{rel}", evaluator_dir / rel) for rel in file_paths],
            download_script=evaluator_dir / "download_dataset.sh",
        )

    if task_name == "eplb":
        target = evaluator_dir / "expert-load.json"
        return AssetRule(
            task_id=task.task_id,
            required=[target],
            copy_sources=[task.root_dir / "expert-load.json"],
            download_url=(
                "https://huggingface.co/datasets/abmfy/eplb-openevolve/"
                "resolve/main/expert-load.json"
            ),
            download_target=target,
        )

    return None


def _missing(paths: Sequence[Path]) -> list[Path]:
    return [path for path in paths if not path.exists()]


def _run_download_script(script: Path) -> None:
    if not script.exists():
        raise FileNotFoundError(f"download script not found: {script}")
    subprocess.run(["bash", str(script)], cwd=str(script.parent), check=True)


def _download_url(url: str, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    try:
        urllib.request.urlretrieve(url, tmp)
        tmp.replace(target)
    finally:
        if tmp.exists():
            tmp.unlink()


def ensure_task_assets(task: BenchmarkTask, *, auto_download: bool = False) -> tuple[bool, list[Path], list[str]]:
    """Return (ok, missing_paths, actions)."""
    rule = _asset_rule_for(task)
    if rule is None:
        return True, [], [f"No task-specific asset checks for {task.task_id}."]

    actions: list[str] = []
    missing = _missing(rule.required)
    if not missing:
        return True, [], [f"Assets ready for {task.task_id}."]

    if auto_download:
        for source in rule.copy_sources:
            if source.exists() and rule.download_target is not None:
                rule.download_target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, rule.download_target)
                actions.append(f"Copied {source} -> {rule.download_target}")
                break

        missing = _missing(rule.required)
        if missing and rule.download_files:
            missing_set = {path.resolve() for path in missing}
            for url, target in rule.download_files:
                if target.resolve() not in missing_set:
                    continue
                actions.append(f"Downloading {url} -> {target}")
                _download_url(url, target)

        missing = _missing(rule.required)
        if missing and rule.download_script is not None:
            actions.append(f"Running asset downloader: {rule.download_script}")
            _run_download_script(rule.download_script)

        missing = _missing(rule.required)
        if missing and rule.download_url is not None and rule.download_target is not None:
            actions.append(f"Downloading {rule.download_url} -> {rule.download_target}")
            _download_url(rule.download_url, rule.download_target)

    missing = _missing(rule.required)
    return not missing, missing, actions


def main() -> int:
    parser = argparse.ArgumentParser(description="Check benchmark task data assets.")
    parser.add_argument("-c", "--config", required=True, help="YAML config path.")
    parser.add_argument("-b", "--benchmark", required=True, help="Benchmark task id.")
    parser.add_argument(
        "--auto-download",
        action="store_true",
        help="Download/copy known missing assets when possible.",
    )
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    task = _resolve_task(config_path, args.benchmark)

    try:
        ok, missing, actions = ensure_task_assets(task, auto_download=args.auto_download)
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] Benchmark asset preflight failed for {args.benchmark}: {exc}")
        return 1

    for action in actions:
        print(f"[INFO] {action}")

    if ok:
        print(f"[OK] Benchmark assets ready for {args.benchmark}.")
        return 0

    print(f"[ERROR] Missing required benchmark assets for {args.benchmark}:")
    for path in missing:
        print(f"  - {path}")
    if not args.auto_download:
        print("[INFO] Enable auto-download, or download the files listed above.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
