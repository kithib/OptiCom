import json


def test_rescore_results_dry_run_and_write(tmp_path):
    from rescore_results import (
        BenchmarkRegistry,
        collect_entries,
        rescore_entry,
        write_rescore,
    )

    benchmark_root = tmp_path / "benchmarks"
    task_dir = benchmark_root / "mockcat" / "mocktask"
    task_dir.mkdir(parents=True)
    (task_dir / "initial_program.py").write_text("# initial\n", encoding="utf-8")
    (task_dir / "evaluator.py").write_text(
        "\n".join(
            [
                "from pathlib import Path",
                "def evaluate(program_path):",
                "    text = Path(program_path).read_text()",
                "    score = 0.75 if 'good' in text else 0.25",
                "    return {'combined_score': score, 'success_rate': 1.0}",
            ]
        ),
        encoding="utf-8",
    )

    archive_dir = tmp_path / "result" / "textgrad" / "mockcat__mocktask" / "mock_model"
    archive_dir.mkdir(parents=True)
    (archive_dir / "best_program.py").write_text("# good program\n", encoding="utf-8")
    result_json = archive_dir / "result.json"
    result_json.write_text(
        json.dumps(
            {
                "baseline": "textgrad",
                "task_slug": "mockcat__mocktask",
                "benchmark_task_id": "mocktask",
                "model_slug": "mock_model",
                "runtime_score": 99.0,
                "all_metrics": {"score": 99.0},
                "code_file": "best_program.py",
            }
        ),
        encoding="utf-8",
    )

    entries = collect_entries(tmp_path / "result", task="mockcat__mocktask")
    assert len(entries) == 1

    registry = BenchmarkRegistry(benchmark_root)
    row = rescore_entry(
        entries[0],
        registry,
        timeout_seconds=30,
        python_executable="python3",
    )

    assert row.old_score == 99.0
    assert row.new_score == 0.75
    assert row.metrics["combined_score"] == 0.75
    assert json.loads(result_json.read_text(encoding="utf-8"))["runtime_score"] == 99.0

    write_rescore(row)
    updated = json.loads(result_json.read_text(encoding="utf-8"))
    assert updated["runtime_score"] == 0.75
    assert updated["all_metrics"]["combined_score"] == 0.75
    assert updated["rescored_from_runtime_score"] == 99.0
    assert updated["rescored_program_path"].endswith("best_program.py")
    assert (archive_dir / "result.json.bak").exists()
