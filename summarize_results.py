"""
summarize_results.py
====================
遍历 result/<baseline>/<task>/<model>/result.json，读取所有已完成任务的历史最佳，
并以表格形式打印成绩单。可用 --config 按模型筛选。
"""
import argparse
import json
import sys
from pathlib import Path


V1_ROOT = Path(__file__).resolve().parent
if str(V1_ROOT) not in sys.path:
    sys.path.insert(0, str(V1_ROOT))


def _get_llm_name(cfg_path: str) -> str:
    from run import _llm_name, load_config
    return _llm_name(load_config(Path(cfg_path)))


def _iter_result_files(result_root: Path):
    yield from sorted(result_root.glob("*/*/*/result.json"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None, help="Optional YAML config used to filter one model")
    parser.add_argument("--baseline", default=None, help="Optional baseline filter, e.g. gepa")
    parser.add_argument("--task", default=None, help="Optional task slug filter, e.g. math__circle_packing")
    parser.add_argument("--result-root", default=str(V1_ROOT / "result"), help="Result root directory")
    parser.add_argument(
        "--pivot",
        choices=["none", "by_baseline", "by_model", "by_baseline_model"],
        default="none",
        help=(
            "横向对比模式：\n"
            "  none               -> 扁平表（默认）\n"
            "  by_baseline        -> 行=task, 列=baseline\n"
            "  by_model           -> 行=task, 列=model\n"
            "  by_baseline_model  -> 行=task, 列=baseline/model"
        ),
    )
    parser.add_argument(
        "--pivot-metric",
        choices=["runtime_score", "abs_value"],
        default="runtime_score",
        help="横向对比的数值指标字段",
    )
    args = parser.parse_args()

    llm_name = _get_llm_name(args.config) if args.config else None
    filters = []
    if llm_name:
        filters.append(f"model={llm_name}")
    if args.baseline:
        filters.append(f"baseline={args.baseline}")
    if args.task:
        filters.append(f"task={args.task}")
    print("Gathering results" + (f" ({', '.join(filters)})" if filters else "") + "\n")

    result_root = Path(args.result_root)
    if not result_root.exists() or not result_root.is_dir():
        print(f"No result directory found at {result_root}.")
        return

    # 1. 收集所有 result.json 数据
    rows = []
    for json_file in _iter_result_files(result_root):
        try:
            with json_file.open("r", encoding="utf-8") as f:
                data = json.load(f)
            model = data.get("model_slug") or json_file.parent.name
            baseline = data.get("baseline") or json_file.parents[2].name
            slug = data.get("task_slug") or json_file.parents[1].name

            if llm_name and model != llm_name:
                continue
            if args.baseline and baseline != args.baseline:
                continue
            if args.task and slug != args.task:
                continue

            cat = data.get("benchmark_category") or "N/A"
            task = data.get("benchmark_task_id") or slug
            
            score = data.get("runtime_score", -float('inf'))
            score_str = f"{score:.4f}" if isinstance(score, float) and score != -float('inf') else str(score)
            
            abs_m = data.get("abs_metric") or "N/A"
            abs_v = data.get("abs_value")
            if abs_v is None:
                abs_v = "N/A"
            abs_v_str = f"{abs_v:.4f}" if isinstance(abs_v, float) else str(abs_v)
            
            # 是否在这一轮打破了记录 (有 delta 且 > 0)
            delta = data.get("delta_vs_previous")
            improved = "Yes" if isinstance(delta, float) and delta > 0 else "-"
            
            elapsed = data.get("elapsed_sec", "N/A")
            
            rows.append({
                "baseline": baseline, "model": model, "slug": slug, "cat": cat, "task": task,
                "score": score_str, "abs_metric": abs_m, "abs_value": abs_v_str,
                "improved": improved, "elapsed": elapsed,
                "_score_raw": score if isinstance(score, (int, float)) else None,
                "_abs_value_raw": abs_v if isinstance(abs_v, (int, float)) else None,
            })
        except Exception as e:
            print(f"[WARN] Failed to read {json_file}: {e}")

    if not rows:
        print("No valid result.json files found.")
        return

    # 横向对比模式（pivot table）
    if args.pivot != "none":
        _print_pivot_table(rows, pivot_mode=args.pivot, metric=args.pivot_metric)
        return

    # 2. 格式化输出表头
    print(f"{'Baseline':<15} | {'Category':<15} | {'Task ID':<30} | {'Model':<24} | {'Runtime Score':<15} | {'Abs Metric':<15} | {'Abs Value':<12} | {'New Best?':<10} | {'Time(s)'}")
    print("-" * 170)

    # 3. 按 Category 分组排序打印
    rows.sort(key=lambda r: (r['baseline'], r['cat'], r['task'], r['model']))
    for r in rows:
        print(f"{r['baseline']:<15} | {r['cat']:<15} | {r['task']:<30} | {r['model']:<24} | {r['score']:<15} | {r['abs_metric']:<15} | {r['abs_value']:<12} | {r['improved']:<10} | {r['elapsed']}")


def _print_pivot_table(rows, *, pivot_mode: str, metric: str):
    """
    竖排展示：以 task 为一个 block，block 内按 (维度 -> 分数) 一行两个并排，
    便于横向对同一个任务在多个 baseline / model 下进行对比。
    """

    def _col_key(r):
        if pivot_mode == "by_baseline":
            return r["baseline"]
        if pivot_mode == "by_model":
            return r["model"]
        return f"{r['baseline']}/{r['model']}"

    raw_field = "_score_raw" if metric == "runtime_score" else "_abs_value_raw"

    matrix: dict[tuple, dict[str, float]] = {}
    cols: set[str] = set()

    for r in rows:
        val = r.get(raw_field)
        if val is None:
            continue
        row_key = (r["cat"], r["task"], r["slug"])
        col = _col_key(r)
        cols.add(col)
        cell = matrix.setdefault(row_key, {})
        if col not in cell or val > cell[col]:
            cell[col] = val

    if not matrix:
        print(f"No numeric values for pivot metric '{metric}'.")
        return

    col_list = sorted(cols)
    row_keys = sorted(matrix.keys())

    name_w = max(20, max(len(c) for c in col_list) + 2)
    val_w = 12
    pair_w = name_w + 3 + val_w  # "<name> : <value>"

    print(f"\n[Pivot] metric={metric}, mode={pivot_mode} (竖排 / 一行两个 / 按分数降序)\n")

    for key in row_keys:
        cat, task, slug = key
        cells = matrix[key]
        best_val = max(cells.values()) if cells else None
        best_cols = {c for c, v in cells.items() if v == best_val}

        # 按分数降序排序：缺失值的列排在最后
        sorted_cols = sorted(
            col_list,
            key=lambda c: (0, -cells[c]) if c in cells else (1, 0),
        )

        title = f"== [{cat}] {task}  (slug={slug}) =="
        print(title)

        # 一行排两组 (col -> value)，按已经降序好的顺序
        for i in range(0, len(sorted_cols), 2):
            line_parts = []
            for c in sorted_cols[i : i + 2]:
                if c in cells:
                    marker = " *" if c in best_cols and len(cells) > 1 else "  "
                    val_str = f"{cells[c]:.4f}{marker}"
                else:
                    val_str = "-"
                line_parts.append(f"{c:<{name_w}} : {val_str:<{val_w}}")
            # 让两个 pair 之间留一点间隔
            print("  " + "    ".join(line_parts))

        if len(cells) > 1:
            print(f"  >> Best: {', '.join(sorted(best_cols))}  ({best_val:.4f})")
        print()

    print("(* 表示该任务下取得最佳分数的列；多个列同分时会都被标记)")

if __name__ == "__main__":
    main()
# python summarize_results.py --task ADRS__prism --pivot by_baseline_model --pivot-metric runtime_score