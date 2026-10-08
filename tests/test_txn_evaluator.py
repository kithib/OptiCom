import sys
from pathlib import Path


V1_ROOT = Path(__file__).resolve().parents[1]
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))


def test_txn_evaluator_rejects_short_schedules(tmp_path):
    from benchmarks.ADRS.txn_scheduling.evaluator.evaluator import evaluate

    program = tmp_path / "short_schedule.py"
    program.write_text(
        "def get_random_costs():\n"
        "    return 3, [[0], [0], [0]]\n",
        encoding="utf-8",
    )

    result = evaluate(str(program))

    assert result["validity"] == 0.0
    assert result["combined_score"] == 0.0


def test_txn_validate_schedule_requires_complete_permutation():
    from benchmarks.ADRS.txn_scheduling.evaluator.evaluator import validate_schedule

    assert validate_schedule([0, 1, 2], expected_num_txns=3)
    assert not validate_schedule([0], expected_num_txns=3)
    assert not validate_schedule([0, 1, 1], expected_num_txns=3)
    assert not validate_schedule([0, 1, 3], expected_num_txns=3)
