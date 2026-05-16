from __future__ import annotations

import csv
import math
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"

def read_csv(name: str) -> list[dict[str, str]]:
    path = PROCESSED / name
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def assert_close(actual: str, expected: float, label: str, tol: float = 1e-9) -> None:
    value = float(actual)
    if not math.isclose(value, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {value}")


def check_key_values() -> None:
    table1 = {row["scale_n"]: row for row in read_csv("table1_navigation_main.csv")}
    assert_close(table1["20"]["reward_mean"], 1018.7, "Table 1 n=20 reward mean")
    assert_close(table1["20"]["violation_percent_mean"], 4.75, "Table 1 n=20 violation mean")

    comm = {row["communication"]: row for row in read_csv("table_comm_tradeoff_n20.csv")}
    assert_close(comm["k=4"]["episode_violation_percent_mean"], 0.00, "n=20 k=4 episode violation")
    assert_close(comm["full"]["communication_scalars_per_step"], 117730, "n=20 full communication scalars")

    controlled = {
        (row["scale_n"], row["variant"]): row
        for row in read_csv("table_controlled_filter_value.csv")
    }
    sparse20 = controlled[("20", "Sparse distributed learning (k=2) + filter")]
    assert_close(sparse20["eval_episode_violation_percent_mean"], 0.50, "Controlled n=20 sparse episode violation")
    assert_close(sparse20["communication_scalars_per_step"], 22926, "Controlled n=20 sparse communication")

    warehouse = {row["method"]: row for row in read_csv("table_warehouse_stress_test.csv")}
    assert_close(warehouse["DL-PDAC sparse (k=2)"]["items_per_episode_mean"], 5.24, "Warehouse sparse items")
    assert_close(
        warehouse["DL-PDAC sparse (k=2)"]["robot_collision_step_rate_percent_mean"],
        28.71,
        "Warehouse sparse robot-collision rate",
    )

    residual = {row["metric"]: row for row in read_csv("residual_audit_summary.csv")}
    assert_close(residual["mean_residual_delta_plus_epsilon"]["value"], 0.4633, "Mean residual")
    assert_close(residual["full_eval_epsilon_zero_rate"]["value"], 88.91, "Eval epsilon=0 rate")


def main() -> int:
    check_key_values()
    print("Release check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
