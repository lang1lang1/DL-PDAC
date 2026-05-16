"""
Offline residual audit for the discrete navigation runtime filter.

This script replays a saved revision-navigation run, samples agent-time states
from deterministic evaluation trajectories, and estimates an empirical
implementation residual:

  empirical_delta_i(t) = max_a [ g_tilde_i(s_t, a) - ghat_i(s_t, a) ]_+

where g_tilde_i is a simulator-based one-step Lyapunov-violation audit under
fixed other-agent executed actions, and ghat_i is the discrete geometric proxy
used by the runtime filter.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dl_pac_v2 import DLPACSystemV2
from envs import MultiAgentParticleEnv


@dataclass
class AuditRecord:
    agent_idx: int
    obs_i: np.ndarray
    policy_action_i: np.ndarray
    executed_actions: List[int]
    epsilon: float
    snapshot: Dict[str, object]


def parse_args():
    parser = argparse.ArgumentParser(description="Audit discrete navigation residuals")
    parser.add_argument("run_dir", type=str, help="Directory containing run_metadata.json and final_checkpoint.pt")
    parser.add_argument("--n_states", type=int, default=2000, help="Number of agent-time states to audit")
    parser.add_argument("--eval_episodes", type=int, default=100, help="Deterministic evaluation horizon used for sampling")
    parser.add_argument("--seed_base", type=int, default=900000, help="Base seed for deterministic audit rollouts")
    parser.add_argument("--alpha_f", type=float, default=3.0, help="Residual-slack normalization coefficient")
    parser.add_argument("--beta_f", type=float, default=0.0, help="Relaxation constant used in g_tilde")
    parser.add_argument("--device", type=str, default="", help="Override device (defaults to run metadata device)")
    parser.add_argument("--output_name", type=str, default="residual_audit", help="Stem for output files")
    return parser.parse_args()


def load_run(run_dir: str, device_override: str = ""):
    metadata_path = os.path.join(run_dir, "run_metadata.json")
    checkpoint_path = os.path.join(run_dir, "final_checkpoint.pt")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Missing final checkpoint: {checkpoint_path}")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)
    args = metadata["args"]
    adjacency = np.asarray(metadata["adjacency"], dtype=np.float32)
    device = device_override or args.get("device", "cpu")

    boundary_size = MultiAgentParticleEnv.compute_boundary_size(args["n_agents"])
    vision_range = max(3.0, boundary_size * 0.75)
    env = MultiAgentParticleEnv(
        n_agents=args["n_agents"],
        n_landmarks=2,
        max_steps=args["max_steps"],
        boundary_size=boundary_size,
        vision_range=vision_range,
    )
    system = DLPACSystemV2(
        n_agents=args["n_agents"],
        obs_dims=[env.obs_dim] * args["n_agents"],
        action_dims=[env.action_dim] * args["n_agents"],
        adjacency_matrix=adjacency,
        k_hops=args["k_hops"],
        device=device,
        lyapunov_coef=args["lyapunov_coef"],
        dual_lr=args["dual_lr"],
        actor_consensus=args["actor_consensus"],
        dual_mode=args["dual_mode"],
    )
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="You are using `torch.load` with `weights_only=False`",
            category=FutureWarning,
        )
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    system.load_checkpoint_payload(checkpoint)
    return metadata, env, system


def reservoir_sample(records: List[AuditRecord], record: AuditRecord, seen: int, rng: np.random.Generator, limit: int):
    if len(records) < limit:
        records.append(record)
        return
    idx = int(rng.integers(seen))
    if idx < limit:
        records[idx] = record


def collect_state_sample(
    env: MultiAgentParticleEnv,
    system: DLPACSystemV2,
    n_states: int,
    eval_episodes: int,
    seed_base: int,
) -> tuple[List[AuditRecord], Dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed_base)
    records: List[AuditRecord] = []
    seen = 0
    full_eval_storage = {
        "epsilon": [],
        "raw_proxy_violation": [],
        "selected_proxy_violation": [],
        "filter_applied": [],
        "safe_action_exists": [],
    }

    for eval_ep in range(eval_episodes):
        obs, _ = env.reset(seed=seed_base + eval_ep)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]

        for _step in range(env.max_steps):
            policy_actions = system.act(obs_list, deterministic=True)
            executed_actions: List[int] = []
            filter_metrics: List[Dict[str, object]] = []
            for agent_idx, agent in enumerate(sorted(obs.keys())):
                metrics = env.evaluate_filter_metrics(agent_idx, policy_actions[agent_idx])
                filter_metrics.append(metrics)
                executed_actions.append(int(metrics["selected_action"]))
                full_eval_storage["epsilon"].append(float(metrics["epsilon"]))
                full_eval_storage["raw_proxy_violation"].append(float(metrics["raw_proxy_violation"]))
                full_eval_storage["selected_proxy_violation"].append(float(metrics["selected_proxy_violation"]))
                full_eval_storage["filter_applied"].append(float(metrics["filter_applied"]))
                full_eval_storage["safe_action_exists"].append(float(metrics["safe_action_exists"]))

            snapshot = env.get_state_snapshot()
            for agent_idx in range(env.n_agents):
                seen += 1
                reservoir_sample(
                    records,
                    AuditRecord(
                        agent_idx=agent_idx,
                        obs_i=np.asarray(obs_list[agent_idx], dtype=np.float32).copy(),
                        policy_action_i=np.asarray(policy_actions[agent_idx], dtype=np.float32).copy(),
                        executed_actions=executed_actions.copy(),
                        epsilon=float(filter_metrics[agent_idx]["epsilon"]),
                        snapshot={
                            "agents": snapshot["agents"].copy(),
                            "timestep": int(snapshot["timestep"]),
                            "agent_positions": np.asarray(snapshot["agent_positions"], dtype=np.float32).copy(),
                            "agent_velocities": np.asarray(snapshot["agent_velocities"], dtype=np.float32).copy(),
                            "landmarks": np.asarray(snapshot["landmarks"], dtype=np.float32).copy(),
                            "seed": int(snapshot["seed"]),
                        },
                    ),
                    seen,
                    rng,
                    n_states,
                )

            action_dict = {
                agent: executed_actions[i]
                for i, agent in enumerate(sorted(obs.keys()))
            }
            next_obs, _rewards, terminations, truncations, _infos = env.step(
                action_dict,
                use_safety_filter=False,
            )
            obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            if any(terminations.values()) or any(truncations.values()):
                break

    return records, {
        key: np.asarray(values, dtype=np.float32)
        for key, values in full_eval_storage.items()
    }


def lyapunov_value(agent, obs_i: np.ndarray) -> float:
    with torch.no_grad():
        obs_t = torch.as_tensor(obs_i, dtype=torch.float32, device=agent.device).unsqueeze(0)
        return float(agent.lyapunov(obs_t).squeeze().item())


def audit_records(
    env: MultiAgentParticleEnv,
    system: DLPACSystemV2,
    records: List[AuditRecord],
    alpha_f: float,
    beta_f: float,
):
    empirical_delta = []
    epsilon = []
    residual = []
    slack = []

    agent_names = sorted(env.possible_agents)

    for record in records:
        agent_idx = record.agent_idx
        agent = system.agents[agent_idx]
        current_v = lyapunov_value(agent, record.obs_i)
        max_gap = 0.0

        for candidate_action in range(env.action_dim):
            env.set_state_snapshot(record.snapshot)
            proxy = env.compute_proxy_violation(agent_idx, candidate_action)
            action_dict = {
                agent_names[i]: record.executed_actions[i]
                for i in range(env.n_agents)
            }
            action_dict[agent_names[agent_idx]] = candidate_action
            next_obs, _rewards, _terminations, _truncations, infos = env.step(
                action_dict,
                use_safety_filter=False,
            )
            next_obs_i = np.asarray(next_obs[agent_names[agent_idx]], dtype=np.float32)
            next_v = lyapunov_value(agent, next_obs_i)
            constraint_i = float(infos[agent_names[agent_idx]].get("constraint", 0.0))
            g_tilde = next_v - current_v + alpha_f * constraint_i - beta_f
            max_gap = max(max_gap, max(0.0, g_tilde - proxy))

        empirical_delta.append(max_gap)
        epsilon.append(float(record.epsilon))
        residual.append(max_gap + float(record.epsilon))
        slack.append((max_gap + float(record.epsilon)) / alpha_f)

    return {
        "empirical_delta": np.asarray(empirical_delta, dtype=np.float32),
        "epsilon": np.asarray(epsilon, dtype=np.float32),
        "residual": np.asarray(residual, dtype=np.float32),
        "slack": np.asarray(slack, dtype=np.float32),
    }


def summarize_array(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"mean": 0.0, "median": 0.0, "p95": 0.0, "max": 0.0}
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


def main():
    args = parse_args()
    run_dir = os.path.abspath(args.run_dir)
    metadata, env, system = load_run(run_dir, device_override=args.device)
    records, full_eval_arrays = collect_state_sample(
        env=env,
        system=system,
        n_states=args.n_states,
        eval_episodes=args.eval_episodes,
        seed_base=args.seed_base,
    )
    arrays = audit_records(
        env=env,
        system=system,
        records=records,
        alpha_f=args.alpha_f,
        beta_f=args.beta_f,
    )

    summary = {
        "run_dir": run_dir,
        "variant_label": metadata.get("variant_label", ""),
        "n_agents": int(metadata["args"]["n_agents"]),
        "seed": int(metadata["args"]["seed"]),
        "k_hops": int(metadata["args"]["k_hops"]),
        "audited_state_count": int(len(records)),
        "alpha_f": float(args.alpha_f),
        "beta_f": float(args.beta_f),
        "empirical_delta": summarize_array(arrays["empirical_delta"]),
        "sampled_epsilon": summarize_array(arrays["epsilon"]),
        "residual": summarize_array(arrays["residual"]),
        "residual_slack": summarize_array(arrays["slack"]),
        "paper_report": {
            "p95_residual": float(np.percentile(arrays["residual"], 95)) if arrays["residual"].size else 0.0,
            "mean_residual_slack": float(np.mean(arrays["slack"])) if arrays["slack"].size else 0.0,
        },
        "full_eval_epsilon": {
            **summarize_array(full_eval_arrays["epsilon"]),
            "count": int(full_eval_arrays["epsilon"].size),
            "zero_rate": float(np.mean(full_eval_arrays["epsilon"] <= 1e-12)) if full_eval_arrays["epsilon"].size else 0.0,
        },
    }

    filter_log_path = os.path.join(run_dir, "filter_residual_logs.npz")
    if os.path.exists(filter_log_path):
        full_logs = np.load(filter_log_path)
        train_epsilon = np.asarray(full_logs["train_epsilon"], dtype=np.float32)
        summary["full_train_epsilon"] = summarize_array(train_epsilon)
        summary["full_train_epsilon"]["count"] = int(train_epsilon.size)
        summary["full_train_epsilon"]["zero_rate"] = float(np.mean(train_epsilon <= 1e-12)) if train_epsilon.size else 0.0

    summary_path = os.path.join(run_dir, f"{args.output_name}_summary.json")
    arrays_path = os.path.join(run_dir, f"{args.output_name}_samples.npz")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    np.savez_compressed(
        arrays_path,
        **arrays,
        full_eval_epsilon=full_eval_arrays["epsilon"],
        full_eval_raw_proxy_violation=full_eval_arrays["raw_proxy_violation"],
        full_eval_selected_proxy_violation=full_eval_arrays["selected_proxy_violation"],
        full_eval_filter_applied=full_eval_arrays["filter_applied"],
        full_eval_safe_action_exists=full_eval_arrays["safe_action_exists"],
    )

    print(json.dumps(summary, indent=2))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved arrays to:   {arrays_path}")


if __name__ == "__main__":
    main()
