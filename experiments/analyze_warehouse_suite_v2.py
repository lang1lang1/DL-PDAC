"""
Aggregate warehouse benchmark v2 summaries.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze warehouse suite v2")
    parser.add_argument("--root_dir", type=str, required=True)
    parser.add_argument("--output_prefix", type=str, default="warehouse_v2")
    return parser.parse_args()


def collect_summaries(root_dir: str) -> List[Dict]:
    rows = []
    for dirpath, _, filenames in os.walk(root_dir):
        if "summary.json" not in filenames:
            continue
        path = os.path.join(dirpath, "summary.json")
        with open(path, "r", encoding="utf-8") as f:
            item = json.load(f)
        item["run_dir"] = dirpath
        rows.append(item)
    return rows


def mean_sem(values: List[float]):
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        return float(arr[0]), 0.0
    return float(arr.mean()), float(arr.std(ddof=1) / np.sqrt(arr.size))


def aggregate(rows: List[Dict]) -> List[Dict]:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["algorithm"], row["n_agents"])].append(row)

    output = []
    metrics = [
        "eval_reward",
        "throughput_items_per_episode",
        "throughput_items_per_step",
        "success_rate",
        "deadlock_rate",
        "robot_collision_step_rate",
        "boundary_contact_step_rate",
        "mean_episode_length",
        "mean_final_items_picked",
    ]
    comm_metrics = [
        "total_scalars_per_episode",
        "total_scalars_per_step",
        "latency_ms_per_step",
        "latency_ms_per_update",
    ]

    for (algorithm, n_agents), items in sorted(grouped.items()):
        out = {
            "algorithm": algorithm,
            "n_agents": n_agents,
            "num_runs": len(items),
        }
        for metric in metrics:
            m, s = mean_sem([x[metric] for x in items])
            out[f"mean_{metric}"] = m
            out[f"sem_{metric}"] = s
        for metric in comm_metrics:
            m, s = mean_sem([x["comm_stats"].get(metric, 0.0) for x in items])
            out[f"mean_{metric}"] = m
            out[f"sem_{metric}"] = s
        output.append(out)
    return output


def save_csv(rows: List[Dict], path: str):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_json(rows: List[Dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)


def main():
    args = parse_args()
    raw_rows = collect_summaries(args.root_dir)
    agg_rows = aggregate(raw_rows)

    raw_csv = os.path.join(args.root_dir, f"{args.output_prefix}_raw.csv")
    agg_csv = os.path.join(args.root_dir, f"{args.output_prefix}_summary.csv")
    agg_json = os.path.join(args.root_dir, f"{args.output_prefix}_summary.json")

    save_csv(raw_rows, raw_csv)
    save_csv(agg_rows, agg_csv)
    save_json(agg_rows, agg_json)

    print(f"Saved raw rows: {raw_csv}")
    print(f"Saved summary CSV: {agg_csv}")
    print(f"Saved summary JSON: {agg_json}")


if __name__ == "__main__":
    main()
