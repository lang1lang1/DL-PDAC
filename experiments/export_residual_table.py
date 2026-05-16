import argparse
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description="Export paper-ready residual summary tables")
    parser.add_argument("summary_json", type=str, help="Path to residual audit summary JSON")
    parser.add_argument("--label", type=str, default="", help="Optional display label for the setting")
    parser.add_argument("--stem", type=str, default="residual_metrics_table", help="Output filename stem")
    return parser.parse_args()


def fmt(x: float) -> str:
    return f"{x:.4f}"


def main():
    args = parse_args()
    summary_path = Path(args.summary_json).resolve()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    run_dir = summary_path.parent
    label = args.label or f"{data['variant_label']}, n={data['n_agents']}, seed={data['seed']}"

    row = {
        "setting": label,
        "mean_delta": float(data["empirical_delta"]["mean"]),
        "p95_delta": float(data["empirical_delta"]["p95"]),
        "mean_epsilon": float(data["full_eval_epsilon"]["mean"]),
        "p95_epsilon": float(data["full_eval_epsilon"]["p95"]),
        "mean_residual_slack": float(data["residual_slack"]["mean"]),
        "p95_residual": float(data["residual"]["p95"]),
        "delta_audit_count": int(data["audited_state_count"]),
        "epsilon_eval_count": int(data["full_eval_epsilon"]["count"]),
    }

    md_path = run_dir / f"{args.stem}.md"
    tex_path = run_dir / f"{args.stem}.tex"
    csv_path = run_dir / f"{args.stem}.csv"
    json_path = run_dir / f"{args.stem}.json"

    md = "\n".join(
        [
            "| Setting | Mean $\\delta$ | P95 $\\delta$ | Mean $\\epsilon$ | P95 $\\epsilon$ | Mean $(\\delta+\\epsilon)/\\alpha_f$ | P95 $(\\delta+\\epsilon)$ |",
            "|---|---:|---:|---:|---:|---:|---:|",
            f"| {row['setting']} | {fmt(row['mean_delta'])} | {fmt(row['p95_delta'])} | {fmt(row['mean_epsilon'])} | {fmt(row['p95_epsilon'])} | {fmt(row['mean_residual_slack'])} | {fmt(row['p95_residual'])} |",
            "",
            f"Audit counts: sampled $\\delta$ states = {row['delta_audit_count']}, full-eval $\\epsilon$ agent-steps = {row['epsilon_eval_count']}.",
        ]
    )

    tex = "\n".join(
        [
            r"\begin{table}[t]",
            r"\centering",
            r"\caption{Empirical scale of the implementation-side residual terms for the discrete runtime filter. The proxy-mismatch term $\delta$ is estimated by offline one-step audit on sampled evaluation states, while $\epsilon$ is computed from the full logged final-evaluation trajectories.}",
            r"\label{tab:residual_scale_example}",
            r"\begin{tabular}{lcccccc}",
            r"\toprule",
            r"Setting & Mean $\delta$ & P95 $\delta$ & Mean $\epsilon$ & P95 $\epsilon$ & Mean $(\delta+\epsilon)/\alpha_f$ & P95 $(\delta+\epsilon)$ \\",
            r"\midrule",
            f"{row['setting']} & {fmt(row['mean_delta'])} & {fmt(row['p95_delta'])} & {fmt(row['mean_epsilon'])} & {fmt(row['p95_epsilon'])} & {fmt(row['mean_residual_slack'])} & {fmt(row['p95_residual'])} \\\\",
            r"\bottomrule",
            r"\end{tabular}",
            rf"\vspace{{2pt}}",
            rf"\footnotesize Offline audit states: {row['delta_audit_count']}. Full-eval $\epsilon$ agent-steps: {row['epsilon_eval_count']}.",
            r"\end{table}",
        ]
    )

    csv = "\n".join(
        [
            "setting,mean_delta,p95_delta,mean_epsilon,p95_epsilon,mean_residual_slack,p95_residual,delta_audit_count,epsilon_eval_count",
            f"{row['setting']},{fmt(row['mean_delta'])},{fmt(row['p95_delta'])},{fmt(row['mean_epsilon'])},{fmt(row['p95_epsilon'])},{fmt(row['mean_residual_slack'])},{fmt(row['p95_residual'])},{row['delta_audit_count']},{row['epsilon_eval_count']}",
        ]
    )

    md_path.write_text(md, encoding="utf-8")
    tex_path.write_text(tex, encoding="utf-8")
    csv_path.write_text(csv, encoding="utf-8")
    json_path.write_text(json.dumps(row, indent=2), encoding="utf-8")

    print(f"Saved: {md_path}")
    print(f"Saved: {tex_path}")
    print(f"Saved: {csv_path}")
    print(f"Saved: {json_path}")


if __name__ == "__main__":
    main()
