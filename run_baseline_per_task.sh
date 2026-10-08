#!/bin/bash
# ==============================================================================
# run_baseline_per_task.sh
#
# 用途: 对单个 benchmark 任务先跑 OptiCom, 再跑指定 baselines。
#       运行前会自动安装该任务 evaluator/requirements.txt 中声明的依赖。
#
# 用法:
#   ./run_baseline_per_task.sh -b matmul
#   ./run_baseline_per_task.sh --benchmark uncertainty_ineq --baselines all --max-iter 30
#   ./run_baseline_per_task.sh --dry-run -b hexagon_packing_12
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_PWD="$PWD"
cd "$SCRIPT_DIR" || exit 1

CONFIG_FILE="example/benchmark_arc.yaml"
BENCH=""
MAX_ITER_VALUE="30"
BASELINES="all"
PIVOT_MODE="by_baseline_model"
PIVOT_METRIC="runtime_score"
DRY_RUN=0
AUTO_INSTALL=1
INSTALLED_REQ_FILES="|"

usage() {
    echo "Usage: $0 -b <benchmark_task> [-c <config_yaml>] [--baselines <names|all>] [--max-iter <n>] [--dry-run] [--no-auto-install]"
    echo "  -b/--benchmark selects one benchmark task, e.g. matmul, uncertainty_ineq"
    echo "  --no-auto-install disables automatic pip install for evaluator requirements."
}

resolve_config_path() {
    local input="$1"
    if [[ "$input" = /* ]]; then
        echo "$input"
    elif [ -f "$ORIGINAL_PWD/$input" ]; then
        (cd "$(dirname "$ORIGINAL_PWD/$input")" && printf "%s/%s\n" "$PWD" "$(basename "$input")")
    else
        (cd "$(dirname "$SCRIPT_DIR/$input")" 2>/dev/null && printf "%s/%s\n" "$PWD" "$(basename "$input")")
    fi
}

while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -c|--config)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            CONFIG_FILE="$2"
            shift 2
            ;;
        -b|--benchmark|--bench)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            BENCH="$2"
            shift 2
            ;;
        --baselines)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            BASELINES="$2"
            shift 2
            ;;
        -i|--max-iter)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            MAX_ITER_VALUE="$2"
            shift 2
            ;;
        --pivot)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            PIVOT_MODE="$2"
            shift 2
            ;;
        --pivot-metric)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            PIVOT_METRIC="$2"
            shift 2
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-auto-install)
            AUTO_INSTALL=0
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
done

if [ -z "$BENCH" ]; then
    echo "[ERROR] Missing benchmark task. Use -b <task_id>."
    usage
    exit 1
fi

CONFIG_FILE="$(resolve_config_path "$CONFIG_FILE")"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

HOST_PYTHON="${PYTHON:-python3}"
TARGET_PYTHON=$("$HOST_PYTHON" - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
    print(cfg.get("harness", {}).get("python_executable") or sys.executable)
except Exception:
    print(sys.executable)
PY
)

ensure_pip() {
    if "$TARGET_PYTHON" -m pip --version >/dev/null 2>&1; then
        return 0
    fi
    echo "[INFO] pip not found for $TARGET_PYTHON; trying ensurepip..."
    "$TARGET_PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 || {
        echo "[ERROR] pip is unavailable for $TARGET_PYTHON and ensurepip failed."
        return 1
    }
}

install_requirements_file() {
    local req_file="$1"
    [ -f "$req_file" ] || return 0

    case "$INSTALLED_REQ_FILES" in
        *"|$req_file|"*)
            echo "[INFO] Requirements already checked: $req_file"
            return 0
            ;;
    esac

    if [ "$AUTO_INSTALL" -eq 0 ]; then
        echo "[INFO] Auto-install disabled; skipping: $req_file"
        INSTALLED_REQ_FILES="${INSTALLED_REQ_FILES}${req_file}|"
        return 0
    fi

    ensure_pip || return 1
    echo "[INFO] Installing evaluator requirements: $req_file"
    "$TARGET_PYTHON" -m pip install -r "$req_file" || return 1
    INSTALLED_REQ_FILES="${INSTALLED_REQ_FILES}${req_file}|"
}

ensure_core_dependencies() {
    if "$TARGET_PYTHON" - <<'PY' >/dev/null 2>&1
import numpy, openai, scipy, yaml
PY
    then
        return 0
    fi

    if [ "$AUTO_INSTALL" -eq 0 ]; then
        echo "[ERROR] Missing core python dependencies in $TARGET_PYTHON and auto-install is disabled."
        return 1
    fi

    ensure_pip || return 1
    echo "[INFO] Installing core python dependencies for benchmark runner..."
    "$TARGET_PYTHON" -m pip install numpy scipy PyYAML openai || return 1

    "$TARGET_PYTHON" - <<'PY' >/dev/null
import numpy, openai, scipy, yaml
PY
}

task_requirements_files() {
    local task_id="$1"
    "$TARGET_PYTHON" - "$CONFIG_FILE" "$task_id" <<'PY'
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from benchmark_adapter import BenchmarkRegistry

cfg_path = Path(sys.argv[1]).resolve()
task_id = sys.argv[2]
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
root = cfg.get("benchmark", {}).get("root") or "./benchmarks"
root_path = Path(root)
if not root_path.is_absolute():
    root_path = (cfg_path.parent / root_path).resolve()

task = BenchmarkRegistry(str(root_path)).get(task_id)
candidates = [
    task.root_dir / "requirements.txt",
    task.root_dir / "evaluator" / "requirements.txt",
    task.evaluator_module_path.parent / "requirements.txt",
]
seen = set()
for path in candidates:
    path = path.resolve()
    if path.exists() and path not in seen:
        print(path)
        seen.add(path)
PY
}

install_task_requirements() {
    local task_id="$1"
    local req_file
    local found=0

    while IFS= read -r req_file; do
        [ -n "$req_file" ] || continue
        found=1
        install_requirements_file "$req_file" || return 1
    done < <(task_requirements_files "$task_id")

    if [ "$found" -eq 0 ]; then
        echo "[INFO] No evaluator requirements found for task: $task_id"
    fi
}

check_task_assets() {
    local task_id="$1"
    local args=(src/benchmark_preflight.py -c "$CONFIG_FILE" -b "$task_id")
    if [ "$AUTO_INSTALL" -eq 1 ]; then
        args+=(--auto-download)
    fi
    "$TARGET_PYTHON" "${args[@]}"
}

show_task_assets() {
    local task_id="$1"
    "$TARGET_PYTHON" src/benchmark_preflight.py -c "$CONFIG_FILE" -b "$task_id"
}

echo "Running preflight checks using python: $TARGET_PYTHON"
ensure_core_dependencies || exit 1

TASK_SLUG=$("$TARGET_PYTHON" - "$CONFIG_FILE" "$BENCH" <<'PY'
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from benchmark_adapter import BenchmarkRegistry

cfg_path = Path(sys.argv[1]).resolve()
task_id = sys.argv[2]
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
root = cfg.get("benchmark", {}).get("root") or "./benchmarks"
root_path = Path(root)
if not root_path.is_absolute():
    root_path = (cfg_path.parent / root_path).resolve()
task = BenchmarkRegistry(str(root_path)).get(task_id)
print(f"{task.category}__{task_id}".replace("/", "__"))
PY
)

echo "============================================================"
echo " Baseline benchmark runner"
echo "============================================================"
echo "  CONFIG       = $CONFIG_FILE"
echo "  BENCH        = $BENCH"
echo "  MAX_ITER     = $MAX_ITER_VALUE"
echo "  BASELINES    = $BASELINES"
echo "  TASK_SLUG    = $TASK_SLUG"
echo "  PIVOT_MODE   = $PIVOT_MODE"
echo "  PIVOT_METRIC = $PIVOT_METRIC"
echo "  WORKDIR      = $SCRIPT_DIR"
echo "  PYTHON       = $TARGET_PYTHON"
echo "------------------------------------------------------------"

if [ "$DRY_RUN" -eq 1 ]; then
    echo "Dry run only. Commands:"
    echo "  $TARGET_PYTHON run.py -c $CONFIG_FILE --benchmark $BENCH --max-iter $MAX_ITER_VALUE"
    echo "  $TARGET_PYTHON run_baselines.py -c $CONFIG_FILE -b $BENCH --baselines $BASELINES --max-iter $MAX_ITER_VALUE"
    echo "Evaluator requirements:"
    task_requirements_files "$BENCH" | sed 's/^/  - /'
    echo "Benchmark assets:"
    echo "  (dry-run checks only; actual run with auto-install enabled will download known missing assets)"
    show_task_assets "$BENCH" || true
    exit 0
fi

install_task_requirements "$BENCH" || {
    echo "[ERROR] Failed to install evaluator requirements for task: $BENCH"
    exit 1
}
check_task_assets "$BENCH" || {
    echo "[ERROR] Failed to prepare benchmark assets for task: $BENCH"
    exit 1
}
echo "Preflight checks passed."

echo "[INFO] START: run.py - custom run on '$BENCH'"
"$TARGET_PYTHON" run.py -c "$CONFIG_FILE" --benchmark "$BENCH" --max-iter "$MAX_ITER_VALUE"
OA_EXIT=$?
if [ "$OA_EXIT" -ne 0 ]; then
    echo "[ERROR] run.py failed for '$BENCH' with exit code $OA_EXIT"
    exit "$OA_EXIT"
fi

echo "[INFO] START: run_baselines.py - baselines=$BASELINES"
"$TARGET_PYTHON" run_baselines.py -c "$CONFIG_FILE" -b "$BENCH" --baselines "$BASELINES" --max-iter "$MAX_ITER_VALUE"
BASELINE_EXIT=$?
if [ "$BASELINE_EXIT" -ne 0 ]; then
    echo "[ERROR] run_baselines.py failed for '$BENCH' with exit code $BASELINE_EXIT"
    exit "$BASELINE_EXIT"
fi

if [ -f "summarize_results.py" ]; then
    echo ""
    echo "=== TASK RESULTS SUMMARY ==="
    if "$TARGET_PYTHON" summarize_results.py --help 2>&1 | grep -q -- "--pivot"; then
        "$TARGET_PYTHON" summarize_results.py -c "$CONFIG_FILE" --task "$TASK_SLUG" --pivot "$PIVOT_MODE" --pivot-metric "$PIVOT_METRIC"
    else
        "$TARGET_PYTHON" summarize_results.py -c "$CONFIG_FILE" --task "$TASK_SLUG"
    fi
fi
