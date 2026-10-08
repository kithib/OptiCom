#!/bin/bash
# ==============================================================================
# run_all_benchmarks.sh
#
# 用途: 遍历当前配置下所有 benchmark 任务并逐一执行，完成后打印成绩汇总表。
#
# 用法:
#   ./run_all_benchmarks.sh
#   ./run_all_benchmarks.sh -c example/benchmark_arc.yaml
#   ./run_all_benchmarks.sh -c example/benchmark_arc.yaml -i 3
#   ./run_all_benchmarks.sh -b gepa -i 5
#   ./run_all_benchmarks.sh --dry-run -b gepa
#   ./run_all_benchmarks.sh -r 2
#   ./run_all_benchmarks.sh -h
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_PWD="$PWD"
cd "$SCRIPT_DIR" || exit 1

CONFIG_FILE="example/benchmark_arc.yaml"
MAX_ITER_VALUE=""
MAX_RETRIES=2
BASELINE="opticom"
DRY_RUN=0
AUTO_INSTALL=1
INSTALLED_REQ_FILES="|"

usage() {
    echo "Usage: $0 [-c <config_yaml>] [-i <max_iterations>] [-r <max_retries>] [-b <baseline>] [--dry-run] [--no-auto-install]"
    echo "  -b/--baseline defaults to opticom. Example: gepa, textgrad, lats"
    echo "  --no-auto-install disables automatic pip install for benchmark evaluator requirements."
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

# 1. 解析参数
while [[ "$#" -gt 0 ]]; do
    case "$1" in
        -c|--config)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            CONFIG_FILE="$2"
            shift 2
            ;;
        -i|--max-iter)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            MAX_ITER_VALUE="$2"
            shift 2
            ;;
        -r|--retries)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            MAX_RETRIES="$2"
            shift 2
            ;;
        -b|--baseline)
            [ -n "${2:-}" ] || { echo "[ERROR] Missing value for $1"; exit 1; }
            BASELINE="$2"
            shift 2
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        -n|--dry-run)
            DRY_RUN=1
            shift
            ;;
        --no-auto-install)
            AUTO_INSTALL=0
            shift
            ;;
        *)
            echo "Unknown parameter passed: $1"
            usage
            exit 1
            ;;
    esac
done

CONFIG_FILE="$(resolve_config_path "$CONFIG_FILE")"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "[ERROR] Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "==========================================================="
echo "Starting Full Benchmark Evaluation"
echo "Config: $CONFIG_FILE"
echo "Baseline: $BASELINE"
echo "==========================================================="

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

# 1.1 解析 config 中的 python_executable。用 here-doc 避免多行 python -c 的引号错配。
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

# 1.5 运行前自检 (Preflight Check)
echo "Running preflight checks using python: $TARGET_PYTHON"
ensure_core_dependencies || {
    echo "[ERROR] Failed to prepare core python dependencies in $TARGET_PYTHON"
    exit 1
}
[ -z "$OPENAI_API_KEY" ] && echo "[WARN] OPENAI_API_KEY empty, image_gen/sky_festival may fail."
command -v g++ >/dev/null || echo "[WARN] g++ missing, ale_bench/frontier-cs may fail."
if [ -d "benchmarks/arc_benchmark" ] && [ ! -d "benchmarks/arc_benchmark/data" ]; then
    echo "[WARN] arc_benchmark/data not found, ARC evaluation may fail. (Run convert_arc_agi2_data.py)"
fi
echo "Preflight checks passed."

# 2. 提取所有 benchmark task_id
TASK_IDS=$("$TARGET_PYTHON" - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from benchmark_adapter import BenchmarkRegistry

cfg_path = Path(sys.argv[1]).resolve()
cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
root = cfg.get("benchmark", {}).get("root") or "./benchmarks"
root_path = Path(root)
if not root_path.is_absolute():
    root_path = (cfg_path.parent / root_path).resolve()

for task_id in BenchmarkRegistry(str(root_path)).list_tasks():
    print(task_id)
PY
)

if [ -z "$TASK_IDS" ]; then
    echo "[ERROR] No benchmarks found or failed to parse list."
    exit 1
fi

TOTAL_TASKS=$(echo "$TASK_IDS" | wc -w | tr -d ' ')
echo "Found $TOTAL_TASKS tasks to evaluate."

# 3. 提取当前 LLM 名字
LLM_NAME=$("$TARGET_PYTHON" - "$CONFIG_FILE" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, ".")
from run import _llm_name, load_config

print(_llm_name(load_config(Path(sys.argv[1]).resolve())))
PY
)

if [ "$DRY_RUN" -eq 1 ]; then
    echo ""
    echo "Dry run only. No benchmark will be executed."
    echo "Model: $LLM_NAME"
    echo "Best-result root: result/${BASELINE}/<task>/${LLM_NAME}/"
    echo "Tasks:"
    echo "$TASK_IDS" | sed 's/^/  - /'
    exit 0
fi

# 4. 循环执行
COUNT=1
LOG_DIR="result/${BASELINE}/_logs/${LLM_NAME}"
mkdir -p "$LOG_DIR"
FAILED_LIST_FILE="$LOG_DIR/failed_benchmarks.txt"
echo "Failed benchmarks list for $BASELINE / $LLM_NAME (created at $(date)):" > "$FAILED_LIST_FILE"

while read -r TASK_ID; do
    [ -z "$TASK_ID" ] && continue
    echo ""
    echo "-----------------------------------------------------------"
    echo "[$COUNT/$TOTAL_TASKS] Running task: $TASK_ID"
    echo "-----------------------------------------------------------"

    LOG_FILE="$LOG_DIR/${TASK_ID}.log"
    ATTEMPT=1
    SUCCESS=0

    echo "[INFO] Preparing evaluator environment for task: $TASK_ID"
    if ! install_task_requirements "$TASK_ID"; then
        echo "[ERROR] Failed to install evaluator requirements for task: $TASK_ID"
        echo "- $TASK_ID (failed to install evaluator requirements)" >> "$FAILED_LIST_FILE"
        COUNT=$((COUNT + 1))
        continue
    fi
    if ! check_task_assets "$TASK_ID"; then
        echo "[ERROR] Failed to prepare benchmark assets for task: $TASK_ID"
        echo "- $TASK_ID (failed to prepare benchmark assets)" >> "$FAILED_LIST_FILE"
        COUNT=$((COUNT + 1))
        continue
    fi

    while [ "$ATTEMPT" -le $((MAX_RETRIES + 1)) ]; do
        if [ "$ATTEMPT" -gt 1 ]; then
            echo "[WARN] Task $TASK_ID failed. Retrying... (Attempt $ATTEMPT of $((MAX_RETRIES + 1)))"
            sleep 2
        fi

        RUN_ARGS=(run.py -c "$CONFIG_FILE" --benchmark "$TASK_ID")
        if [ -n "$MAX_ITER_VALUE" ]; then
            RUN_ARGS+=(--max-iter "$MAX_ITER_VALUE")
        fi
        if [ "$BASELINE" != "opticom" ] && [ "$BASELINE" != "default" ]; then
            RUN_ARGS+=(--baseline "$BASELINE")
        fi

        "$TARGET_PYTHON" "${RUN_ARGS[@]}" 2>&1 | tee "$LOG_FILE"
        EXIT_CODE=${PIPESTATUS[0]}

        if [ "$EXIT_CODE" -eq 0 ]; then
            SUCCESS=1
            break
        else
            ATTEMPT=$((ATTEMPT + 1))
        fi
    done

    if [ "$SUCCESS" -eq 0 ]; then
        echo "[ERROR] Task $TASK_ID failed permanently after $((MAX_RETRIES + 1)) attempts."
        echo "Check the detailed error log at: $LOG_FILE"
        echo "- $TASK_ID (see $LOG_FILE)" >> "$FAILED_LIST_FILE"
    fi

    COUNT=$((COUNT + 1))
done <<< "$TASK_IDS"

echo ""
echo "==========================================================="
echo "All $TOTAL_TASKS benchmarks evaluated."
echo "Best results are saved under: result/${BASELINE}/<task>/${LLM_NAME}/"
echo "Logs are saved under: $LOG_DIR"
if [ -f "$FAILED_LIST_FILE" ] && [ "$(wc -l < "$FAILED_LIST_FILE")" -gt 1 ]; then
    echo "[!] Some tasks failed. See $FAILED_LIST_FILE for details."
fi
echo "==========================================================="

# 5. 打印全局成绩汇总
if [ -f "summarize_results.py" ]; then
    echo ""
    echo "=== GLOBAL RESULTS SUMMARY ==="
    "$TARGET_PYTHON" summarize_results.py -c "$CONFIG_FILE" --baseline "$BASELINE"
fi
