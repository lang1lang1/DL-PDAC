from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List

import numpy as np
from scipy import stats


METRICS = [
    {
        "key": "final_eval_reward",
        "label": "Eval reward",
        "display_scale": 1.0,
        "display_unit": "",
    },
    {
        "key": "final_eval_episode_violation_rate",
        "label": "Eval episode violation (%)",
        "display_scale": 100.0,
        "display_unit": "%",
    },
    {
        "key": "final_eval_step_violation_rate",
        "label": "Eval step violation (%)",
        "display_scale": 100.0,
        "display_unit": "%",
    },
    {
        "key": "reward_auc",
        "label": "Reward AUC",
        "display_scale": 1.0,
        "display_unit": "",
    },
]


def parse_args():
    parser = argparse.ArgumentParser(description="Export Table C4 for n=20 sparse vs independent.")
    parser.add_argument(
        "--raw-csv",
        type=str,
        required=True,
        help="Path to controlled_v2_raw_runs.csv",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save Table C4 outputs.",
    )
    return parser.parse_args()


def load_rows(path: str) -> List[Dict]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def filter_rows(rows: List[Dict], n_agents: int, variant_label: str) -> List[Dict]:
    kept = [
        row
        for row in rows
        if int(row["n_agents"]) == n_agents and row["variant_label"] == variant_label
    ]
    kept.sort(key=lambda row: int(row["seed"]))
    return kept


def hedges_g(x: np.ndarray, y: np.ndarray) -> float:
    nx = x.size
    ny = y.size
    if nx < 2 or ny < 2:
        return float("nan")
    sx = x.std(ddof=1)
    sy = y.std(ddof=1)
    pooled = math.sqrt(((nx - 1) * sx * sx + (ny - 1) * sy * sy) / max(nx + ny - 2, 1))
    if pooled == 0:
        return 0.0
    d = (x.mean() - y.mean()) / pooled
    correction = 1.0 - 3.0 / max(4.0 * (nx + ny) - 9.0, 1.0)
    return float(d * correction)


def holm_adjust(rows: List[Dict], key: str = "p_value_raw") -> None:
    sortable = [(idx, row[key]) for idx, row in enumerate(rows)]
    sortable.sort(key=lambda item: item[1])
    m = len(sortable)
    prev_adj = 0.0
    for rank, (idx, p_value) in enumerate(sortable, start=1):
        adjusted = min(1.0, max(prev_adj, (m - rank + 1) * p_value))
        rows[idx]["p_value_holm"] = adjusted
        prev_adj = adjusted


def build_seed_table(independent_rows: List[Dict], sparse_rows: List[Dict]) -> List[Dict]:
    seeds = sorted({int(row["seed"]) for row in independent_rows} | {int(row["seed"]) for row in sparse_rows})
    independent_by_seed = {int(row["seed"]): row for row in independent_rows}
    sparse_by_seed = {int(row["seed"]): row for row in sparse_rows}

    table = []
    for seed in seeds:
        ind = independent_by_seed[seed]
        spa = sparse_by_seed[seed]
        item = {"seed": seed}
        for metric in METRICS:
            key = metric["key"]
            scale = metric["display_scale"]
            item[f"independent_{key}"] = float(ind[key]) * scale
            item[f"sparse_{key}"] = float(spa[key]) * scale
        table.append(item)
    return table


def build_stats_table(independent_rows: List[Dict], sparse_rows: List[Dict]) -> List[Dict]:
    stats_rows: List[Dict] = []
    for metric in METRICS:
        key = metric["key"]
        scale = metric["display_scale"]
        x = np.asarray([float(row[key]) for row in sparse_rows], dtype=float)
        y = np.asarray([float(row[key]) for row in independent_rows], dtype=float)
        p_value = float(stats.ttest_ind(x, y, equal_var=False).pvalue)
        stats_rows.append(
            {
                "metric": metric["label"],
                "mean_sparse": float(np.mean(x) * scale),
                "mean_independent": float(np.mean(y) * scale),
                "mean_difference_sparse_minus_independent": float((np.mean(x) - np.mean(y)) * scale),
                "hedges_g": hedges_g(x, y),
                "p_value_raw": p_value,
                "p_value_holm": float("nan"),
                "n_sparse": int(x.size),
                "n_independent": int(y.size),
            }
        )
    holm_adjust(stats_rows)
    return stats_rows


def save_csv(rows: List[Dict], path: str) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: List[Dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def format_metric_value(metric_label: str, value: float) -> str:
    if "(%)" in metric_label:
        return f"{value:.3f}"
    if metric_label == "Eval reward":
        return f"{value:.3f}"
    return f"{value:.3f}"


def save_markdown(seed_rows: List[Dict], stats_rows: List[Dict], path: str) -> None:
    lines: List[str] = []
    lines.append("# Table C4")
    lines.append("")
    lines.append("n=20, sparse k=2 vs independent, using the 10 real seeds already present in the controlled raw runs.")
    lines.append("")
    lines.append("## Statistics")
    lines.append("")
    lines.append("| Metric | Sparse mean | Independent mean | Mean diff (Sparse - Independent) | Hedges' g | Raw two-sided p | Holm-adjusted p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in stats_rows:
        lines.append(
            "| {metric} | {sparse} | {ind} | {diff} | {g:.3f} | {praw:.6f} | {pholm:.6f} |".format(
                metric=row["metric"],
                sparse=format_metric_value(row["metric"], row["mean_sparse"]),
                ind=format_metric_value(row["metric"], row["mean_independent"]),
                diff=format_metric_value(row["metric"], row["mean_difference_sparse_minus_independent"]),
                g=row["hedges_g"],
                praw=row["p_value_raw"],
                pholm=row["p_value_holm"],
            )
        )
    lines.append("")
    lines.append("## Seed Values")
    lines.append("")
    header = ["Seed"]
    for metric in METRICS:
        header.append(f"Independent {metric['label']}")
        header.append(f"Sparse {metric['label']}")
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "|".join(["---:" if idx == 0 else "---:" for idx, _ in enumerate(header)]) + "|")
    for row in seed_rows:
        cells = [str(row["seed"])]
        for metric in METRICS:
            key = metric["key"]
            cells.append(format_metric_value(metric["label"], row[f"independent_{key}"]))
            cells.append(format_metric_value(metric["label"], row[f"sparse_{key}"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    rows = load_rows(args.raw_csv)
    independent_rows = filter_rows(rows, n_agents=20, variant_label="controlled_independent")
    sparse_rows = filter_rows(rows, n_agents=20, variant_label="controlled_sparse_k2")

    if len(independent_rows) != 10 or len(sparse_rows) != 10:
        raise ValueError(
            f"Expected 10 seeds for each group, got independent={len(independent_rows)}, sparse={len(sparse_rows)}"
        )

    seed_rows = build_seed_table(independent_rows, sparse_rows)
    stats_rows = build_stats_table(independent_rows, sparse_rows)

    base = os.path.join(args.output_dir, "Table C4")
    save_csv(stats_rows, base + ".csv")
    save_json(stats_rows, base + ".json")
    save_markdown(seed_rows, stats_rows, base + ".md")
    save_csv(seed_rows, base + " seeds.csv")
    save_json(seed_rows, base + " seeds.json")

    print(f"Saved stats table: {base}.csv")
    print(f"Saved stats json: {base}.json")
    print(f"Saved stats markdown: {base}.md")
    print(f"Saved seed table: {base} seeds.csv")
    print(f"Saved seed json: {base} seeds.json")


if __name__ == "__main__":
    main()
