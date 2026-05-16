"""
Revision-grade navigation training entrypoint.

This version fixes the evaluation accounting used for paper-facing controlled
comparisons and trade-off studies. In particular, it reports both episode-level
and step-level violation rates over a dedicated evaluation horizon.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dl_pac_v2 import DLPACSystemV2
from envs import ConstraintTracker, MultiAgentParticleEnv

from revision_metrics import (
    compute_normalized_auc,
    create_adjacency_matrix,
    enrich_comm_stats,
    evaluate_navigation,
    summarize_filter_metrics,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Revision-grade navigation training")
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--k_hops", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--update_interval", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./logs_v2/revision_nav_v2")
    parser.add_argument("--lyapunov_coef", type=float, default=0.5)
    parser.add_argument("--dual_lr", type=float, default=0.01)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--dual_mode", choices=["local", "centralized", "sparse"], default="local")
    parser.add_argument("--actor_consensus", action="store_true")
    parser.add_argument("--no_actor_consensus", action="store_false", dest="actor_consensus")
    parser.set_defaults(actor_consensus=False)
    parser.add_argument("--safety_filter", action="store_true")
    parser.add_argument("--no_safety_filter", action="store_false", dest="safety_filter")
    parser.set_defaults(safety_filter=True)
    parser.add_argument("--topology", choices=["random", "chain", "full"], default="chain")
    parser.add_argument("--connectivity", type=float, default=0.5)
    parser.add_argument("--variant_label", type=str, default="")
    return parser.parse_args()


def infer_variant_label(args) -> str:
    if args.variant_label:
        return args.variant_label
    consensus_tag = "cons" if args.actor_consensus else "ind"
    filter_tag = "sf" if args.safety_filter else "nofilter"
    return f"{args.dual_mode}_{consensus_tag}_{filter_tag}_{args.topology}"


def run_experiment(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    boundary_size = MultiAgentParticleEnv.compute_boundary_size(args.n_agents)
    vision_range = max(3.0, boundary_size * 0.75)
    env = MultiAgentParticleEnv(
        n_agents=args.n_agents,
        n_landmarks=2,
        max_steps=args.max_steps,
        boundary_size=boundary_size,
        vision_range=vision_range,
    )

    adjacency = create_adjacency_matrix(
        n_agents=args.n_agents,
        topology=args.topology,
        connectivity=args.connectivity,
        seed=args.seed,
    )

    system = DLPACSystemV2(
        n_agents=args.n_agents,
        obs_dims=[env.obs_dim] * args.n_agents,
        action_dims=[env.action_dim] * args.n_agents,
        adjacency_matrix=adjacency,
        k_hops=args.k_hops,
        device=args.device,
        lyapunov_coef=args.lyapunov_coef,
        dual_lr=args.dual_lr,
        actor_consensus=args.actor_consensus,
        dual_mode=args.dual_mode,
    )

    variant_label = infer_variant_label(args)
    exp_dir = os.path.join(
        args.save_dir,
        f"{variant_label}_n{args.n_agents}_k{args.k_hops}_s{args.seed}",
    )
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "variant_label": variant_label,
                "args": vars(args),
                "adjacency": adjacency.tolist(),
            },
            f,
            indent=2,
        )

    constraint_tracker = ConstraintTracker()
    episode_rewards: List[float] = []
    episode_lengths: List[int] = []
    episode_violation_steps: List[int] = []
    dual_history: List[float] = []
    eval_history: List[Dict[str, float]] = []
    train_filter_storage = {
        "epsilon": [],
        "raw_proxy_violation": [],
        "selected_proxy_violation": [],
        "filter_applied": [],
        "safe_action_exists": [],
    }
    step_count = 0

    start_time = time.perf_counter()

    print("=== Revision Navigation V2 ===")
    print(f"Variant: {variant_label}")
    print(f"Agents: {args.n_agents}, Train episodes: {args.n_episodes}, Eval episodes: {args.eval_episodes}")
    print(f"Dual mode: {args.dual_mode}, Actor consensus: {args.actor_consensus}, Safety filter: {args.safety_filter}")
    print(f"Topology: {args.topology}, k={args.k_hops}, Device: {args.device}")
    print(f"\n{'Ep':>6} | {'TrainR':>8} | {'TrainViol':>9} | {'Dual':>6} | {'EvalR':>8} | {'EvalEpV':>8} | {'EvalStV':>8}")
    print("-" * 86)

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        constraint_tracker.reset_episode()
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        episode_reward = 0.0
        violation_steps = 0

        for step in range(args.max_steps):
            actions = system.act(obs_list, deterministic=False)
            action_dict = {
                agent: int(np.argmax(actions[i]))
                for i, agent in enumerate(sorted(obs.keys()))
            }

            next_obs, rewards, terminations, truncations, infos = env.step(
                action_dict,
                use_safety_filter=args.safety_filter,
                policy_actions={agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))},
            )
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get("constraint", 0.0) for agent in sorted(infos.keys())]
            constraint_dict = {agent: infos[agent].get("constraint", 0.0) for agent in infos.keys()}

            constraint_tracker.add(constraint_dict)
            if any(c > 0 for c in constraint_list):
                violation_steps += 1
            if args.safety_filter:
                for agent in sorted(infos.keys()):
                    metrics = infos[agent].get("filter_metrics")
                    if not metrics:
                        continue
                    train_filter_storage["epsilon"].append(float(metrics["epsilon"]))
                    train_filter_storage["raw_proxy_violation"].append(float(metrics["raw_proxy_violation"]))
                    train_filter_storage["selected_proxy_violation"].append(float(metrics["selected_proxy_violation"]))
                    train_filter_storage["filter_applied"].append(float(metrics["filter_applied"]))
                    train_filter_storage["safe_action_exists"].append(float(metrics["safe_action_exists"]))

            system.step(
                obs_list=obs_list,
                action_list=actions,
                reward_list=reward_list,
                constraint_list=constraint_list,
                next_obs_list=next_obs_list,
                done_list=[terminations[agent] or truncations[agent] for agent in sorted(terminations.keys())],
                agent_positions=env.agent_positions,
            )

            if (step + 1) % args.update_interval == 0:
                system.update_all()

            obs_list = next_obs_list
            episode_reward += float(sum(reward_list))
            step_count += 1

        episode_rewards.append(episode_reward)
        episode_lengths.append(step + 1)
        episode_violation_steps.append(violation_steps)

        if args.dual_mode == "centralized" and system.global_dual is not None:
            dual_history.append(float(system.global_dual))
        else:
            dual_history.append(float(np.mean([agent.dual.item() for agent in system.agents])))

        eval_stats = {}
        if (episode + 1) % args.eval_interval == 0:
            eval_stats = evaluate_navigation(
                system=system,
                env=env,
                n_episodes=args.eval_episodes,
                use_safety_filter=args.safety_filter,
                seed_base=100000 + args.seed * 1000 + episode * args.eval_episodes,
                collect_filter_metrics=args.safety_filter,
            )
            eval_stats["episode"] = episode + 1
            eval_history.append(eval_stats)

        if (episode + 1) % args.log_interval == 0:
            train_viol = constraint_tracker.get_stats()["violation_rate"]
            print(
                f"{episode + 1:>6} | {np.mean(episode_rewards[-args.log_interval:]):>8.2f} | "
                f"{train_viol:>9.2%} | {dual_history[-1]:>6.2f} | "
                f"{eval_stats.get('eval_reward', float('nan')):>8.2f} | "
                f"{eval_stats.get('eval_episode_violation_rate', float('nan')):>8.2%} | "
                f"{eval_stats.get('eval_step_violation_rate', float('nan')):>8.2%}"
            )

    wall_clock_sec = time.perf_counter() - start_time
    final_train_stats = constraint_tracker.get_stats()
    final_eval = evaluate_navigation(
        system=system,
        env=env,
        n_episodes=args.eval_episodes,
        use_safety_filter=args.safety_filter,
        seed_base=900000 + args.seed * 1000,
        collect_filter_metrics=args.safety_filter,
        return_filter_arrays=args.safety_filter,
    )
    final_eval_filter_arrays = {}
    if args.safety_filter:
        for key in [
            "filter_epsilon_values",
            "filter_raw_proxy_violation_values",
            "filter_selected_proxy_violation_values",
            "filter_applied_values",
            "filter_safe_action_exists_values",
        ]:
            final_eval_filter_arrays[key] = np.asarray(final_eval.pop(key), dtype=np.float32)
    train_filter_summary = summarize_filter_metrics(train_filter_storage if args.safety_filter else None)
    comm_stats = enrich_comm_stats(
        comm_stats=system.get_communication_stats(),
        total_steps=step_count,
        total_episodes=args.n_episodes,
        wall_clock_sec=wall_clock_sec,
    )

    reward_auc = compute_normalized_auc(eval_history, "eval_reward", args.n_episodes)
    ep_violation_auc = compute_normalized_auc(eval_history, "eval_episode_violation_rate", args.n_episodes)
    step_violation_auc = compute_normalized_auc(eval_history, "eval_step_violation_rate", args.n_episodes)
    episodes_to_low_violation = next(
        (item["episode"] for item in eval_history if item["eval_episode_violation_rate"] <= 0.05),
        None,
    )

    print("\n" + "=" * 86)
    print("Training Complete")
    print("=" * 86)
    print(f"Final train reward (last 100): {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final train violation rate: {final_train_stats['violation_rate']:.2%}")
    print(f"Final eval reward: {final_eval['eval_reward']:.2f}")
    print(f"Final eval episode violation: {final_eval['eval_episode_violation_rate']:.2%}")
    print(f"Final eval step violation: {final_eval['eval_step_violation_rate']:.2%}")
    if args.safety_filter:
        print(
            "Train filter residuals: "
            f"eps_mean={train_filter_summary['epsilon_mean']:.4f}, "
            f"eps_p95={train_filter_summary['epsilon_p95']:.4f}, "
            f"feasible={train_filter_summary['proxy_feasible_rate']:.2%}"
        )
        print(
            "Final eval filter residuals: "
            f"eps_mean={final_eval['epsilon_mean']:.4f}, "
            f"eps_p95={final_eval['epsilon_p95']:.4f}, "
            f"feasible={final_eval['proxy_feasible_rate']:.2%}"
        )
    print(f"Wall clock: {wall_clock_sec:.1f}s")
    print(
        "Communication totals: "
        f"{comm_stats['total_scalars']:.0f} scalars, "
        f"{comm_stats['total_scalars_per_episode']:.1f} / episode, "
        f"{comm_stats['total_scalars_per_step']:.1f} / step"
    )

    results = {
        "variant_label": variant_label,
        "n_agents": args.n_agents,
        "episode_rewards": episode_rewards,
        "episode_lengths": episode_lengths,
        "episode_violation_steps": episode_violation_steps,
        "constraint_violations": constraint_tracker.violations,
        "dual_history": dual_history,
        "eval_history": eval_history,
        "final_eval": final_eval,
        "final_train_stats": final_train_stats,
        "wall_clock_sec": wall_clock_sec,
        "total_steps": step_count,
        "comm_stats": comm_stats,
        "reward_auc": reward_auc,
        "episode_violation_auc": ep_violation_auc,
        "step_violation_auc": step_violation_auc,
        "episodes_to_low_violation": episodes_to_low_violation,
        "train_filter_summary": train_filter_summary,
        "args": vars(args),
    }
    torch.save(results, os.path.join(exp_dir, "results.pt"))
    torch.save(system.get_checkpoint_payload(), os.path.join(exp_dir, "final_checkpoint.pt"))

    if args.safety_filter:
        np.savez_compressed(
            os.path.join(exp_dir, "filter_residual_logs.npz"),
            train_epsilon=np.asarray(train_filter_storage["epsilon"], dtype=np.float32),
            train_raw_proxy_violation=np.asarray(train_filter_storage["raw_proxy_violation"], dtype=np.float32),
            train_selected_proxy_violation=np.asarray(train_filter_storage["selected_proxy_violation"], dtype=np.float32),
            train_filter_applied=np.asarray(train_filter_storage["filter_applied"], dtype=np.float32),
            train_safe_action_exists=np.asarray(train_filter_storage["safe_action_exists"], dtype=np.float32),
            final_eval_epsilon=final_eval_filter_arrays["filter_epsilon_values"],
            final_eval_raw_proxy_violation=final_eval_filter_arrays["filter_raw_proxy_violation_values"],
            final_eval_selected_proxy_violation=final_eval_filter_arrays["filter_selected_proxy_violation_values"],
            final_eval_filter_applied=final_eval_filter_arrays["filter_applied_values"],
            final_eval_safe_action_exists=final_eval_filter_arrays["filter_safe_action_exists_values"],
        )

    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant_label": variant_label,
                "n_agents": args.n_agents,
                "seed": args.seed,
                "k_hops": args.k_hops,
                "actor_consensus": args.actor_consensus,
                "dual_mode": args.dual_mode,
                "safety_filter": args.safety_filter,
                "final_train_reward_last100": float(np.mean(episode_rewards[-100:])),
                "final_train_violation_rate": float(final_train_stats["violation_rate"]),
                "final_eval_reward": float(final_eval["eval_reward"]),
                "final_eval_episode_violation_rate": float(final_eval["eval_episode_violation_rate"]),
                "final_eval_step_violation_rate": float(final_eval["eval_step_violation_rate"]),
                "final_eval_mean_constraint_sum": float(final_eval["eval_mean_constraint_sum"]),
                "reward_auc": float(reward_auc),
                "episode_violation_auc": float(ep_violation_auc),
                "step_violation_auc": float(step_violation_auc),
                "episodes_to_low_violation": episodes_to_low_violation,
                "train_filter_summary": train_filter_summary,
                "final_eval_filter_summary": {
                    k: final_eval[k]
                    for k in [
                        "filter_metric_count",
                        "epsilon_mean",
                        "epsilon_p95",
                        "raw_proxy_violation_mean",
                        "selected_proxy_violation_mean",
                        "filter_applied_rate",
                        "proxy_feasible_rate",
                    ]
                },
                "wall_clock_sec": float(wall_clock_sec),
                "total_steps": int(step_count),
                "comm_stats": comm_stats,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    print(f"Saved results to: {exp_dir}")
    return results


if __name__ == "__main__":
    run_experiment(parse_args())
