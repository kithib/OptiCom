import sys
from pathlib import Path


V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))


def test_prism_score_averages_failures_as_zero(tmp_path):
    from benchmarks.ADRS.prism.evaluator.evaluator import evaluate

    program = tmp_path / "partial_success.py"
    program.write_text(
        "GPU_MEM_SIZE = 80\n"
        "_CALLS = 0\n"
        "def compute_model_placement(gpu_num, models):\n"
        "    global _CALLS\n"
        "    _CALLS += 1\n"
        "    if _CALLS > 2:\n"
        "        raise RuntimeError('intentional failure')\n"
        "    placement = {g: [] for g in range(gpu_num)}\n"
        "    remaining = [GPU_MEM_SIZE for _ in range(gpu_num)]\n"
        "    for model in sorted(models, key=lambda m: m.model_size, reverse=True):\n"
        "        for gpu_id in range(gpu_num):\n"
        "            if remaining[gpu_id] >= model.model_size:\n"
        "                placement[gpu_id].append(model)\n"
        "                remaining[gpu_id] -= model.model_size\n"
        "                break\n"
        "    return placement\n",
        encoding="utf-8",
    )

    result = evaluate(str(program))

    assert result["success_rate"] == 0.04
    assert 0.0 < result["combined_score"] < 0.1
    assert 0.0 <= result["combined_score"] <= 1.0


def test_prism_score_is_bounded_for_initial_program():
    from benchmarks.ADRS.prism.evaluator.evaluator import evaluate

    result = evaluate(str(V1_ROOT / "benchmarks" / "ADRS" / "prism" / "initial_program.py"))

    assert result["success_rate"] == 1.0
    assert 0.0 <= result["combined_score"] <= 1.0
    assert result["metric_version"] == "prism_bounded_case_average_v2"
