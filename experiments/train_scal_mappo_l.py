"""Train a Scal-MAPPO-L style baseline on the paper navigation task."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from envs import ConstraintTracker, MultiAgentParticleEnv
from scal_mappo_l import ScalMAPPOLSystem

from revision_metrics import (
    compute_normalized_auc,
    create_adjacency_matrix,
    enrich_comm_stats,
    evaluate_navigation,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Scal-MAPPO-L baseline for navigation")
    parser.add_argument("--n_agents", type=int, default=20)
    parser.add_argument("--n_episodes", type=int, default=100)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--k_hops", type=int, default=2)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--eval_interval", type=int, default=20)
    parser.add_argument("--eval_episodes", type=int, default=50)
    parser.add_argument("--rollout_steps", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--dual_lr", type=float, default=0.01)
    parser.add_argument("--cost_limit", type=float, default=0.0)
    parser.add_argument("--cost_mode", choices=["magnitude", "binary"], default="binary")
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--minibatch_size", type=int, default=256)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--topology", choices=["chain", "random", "full"], default="chain")
    parser.add_argument("--connectivity", type=float, default=0.5)
    parser.add_argument("--safety_filter", action="store_true")
    parser.add_argument("--no_safety_filter", action="store_false", dest="safety_filter")
    parser.set_defaults(safety_filter=False)
    parser.add_argument("--save_dir", type=str, default="./logs_v2/scal_mappo_l")
    parser.add_argument("--variant_label", type=str, default="")
    return parser.parse_args()


def infer_variant_label(args) -> str:
    if args.variant_label:
        return args.variant_label
    filter_tag = "sf" if args.safety_filter else "nofilter"
    return f"scal_mappo_l_k{args.k_hops}_{args.topology}_{filter_tag}"


def eval_selection_key(eval_stats: Dict[str, float]):
    """Safety-first checkpoint ordering: lower is better except reward."""
    return (
        float(eval_stats["eval_episode_violation_rate"]),
        float(eval_stats["eval_step_violation_rate"]),
        float(eval_stats["eval_mean_constraint_sum"]),
        -float(eval_stats["eval_reward"]),
    )


def is_better_eval(candidate: Dict[str, float], incumbent: Optional[Dict[str, float]]) -> bool:
    if incumbent is None:
        return True
    return eval_selection_key(candidate) < eval_selection_key(incumbent)


def run_experiment(args) -> Dict:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.device == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    boundary_size = MultiAgentParticleEnv.compute_boundary_size(args.n_agents)
    vision_range = max(3.0, boundary_size * 0.75)
    env = MultiAgentParticleEnv(
        n_agents=args.n_agents,
        n_landmarks=2,
        max_steps=args.max_steps,
        boundary_size=boundary_size,
        vision_range=vision_range,
        seed=args.seed,
    )
    adjacency = create_adjacency_matrix(
        n_agents=args.n_agents,
        topology=args.topology,
        connectivity=args.connectivity,
        seed=args.seed,
    )
    system = ScalMAPPOLSystem(
        n_agents=args.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        adjacency_matrix=adjacency,
        k_hops=args.k_hops,
        device=args.device,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        dual_lr=args.dual_lr,
        cost_limit=args.cost_limit,
        ppo_epochs=args.ppo_epochs,
        minibatch_size=args.minibatch_size,
        entropy_coef=args.entropy_coef,
    )

    variant_label = infer_variant_label(args)
    exp_dir = os.path.join(args.save_dir, f"{variant_label}_n{args.n_agents}_s{args.seed}")
    os.makedirs(exp_dir, exist_ok=True)
    with open(os.path.join(exp_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "variant_label": variant_label,
                "args": vars(args),
                "adjacency": adjacency.tolist(),
                "neighborhoods": system.neighborhoods,
                "baseline_note": (
                    "Scal-MAPPO-L-inspired PyTorch adaptation for this repo's discrete "
                    "navigation environment; not the original Safe-MAMuJoCo implementation."
                ),
            },
            f,
            indent=2,
        )

    constraint_tracker = ConstraintTracker()
    episode_rewards: List[float] = []
    episode_violation_steps: List[int] = []
    episode_lengths: List[int] = []
    dual_history: List[float] = []
    update_history: List[Dict[str, float]] = []
    eval_history: List[Dict[str, float]] = []
    best_eval = None
    best_checkpoint_path = os.path.join(exp_dir, "best_checkpoint.pt")
    step_count = 0
    start_time = time.perf_counter()

    print("=== Scal-MAPPO-L Baseline ===")
    print(f"Variant: {variant_label}")
    print(f"Agents: {args.n_agents}, k={args.k_hops}, topology={args.topology}, device={args.device}")
    print(f"Safety filter during train/eval: {args.safety_filter}")
    print(f"\n{'Ep':>6} | {'TrainR':>8} | {'TrainViol':>9} | {'Dual':>6} | {'Cost':>7} | {'EvalR':>8} | {'EvalEpV':>8} | {'EvalStV':>8}")
    print("-" * 100)

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        constraint_tracker.reset_episode()
        episode_reward = 0.0
        violation_steps = 0
        last_update = {}

        for step in range(args.max_steps):
            action_pack = system.sample_actions(obs_list)
            action_dict = {
                agent: int(action_pack["actions"][i])
                for i, agent in enumerate(sorted(obs.keys()))
            }

            next_obs, rewards, terminations, truncations, infos = env.step(
                action_dict,
                use_safety_filter=args.safety_filter,
                policy_actions={
                    agent: action_pack["probs"][i]
                    for i, agent in enumerate(sorted(obs.keys()))
                },
            )
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get("constraint", 0.0) for agent in sorted(infos.keys())]
            train_cost_list = (
                [float(c > 0.0) for c in constraint_list]
                if args.cost_mode == "binary"
                else constraint_list
            )
            done_list = [
                terminations[agent] or truncations[agent]
                for agent in sorted(terminations.keys())
            ]

            constraint_tracker.add({agent: infos[agent].get("constraint", 0.0) for agent in infos.keys()})
            if any(c > 0 for c in constraint_list):
                violation_steps += 1

            system.store(
                masked_obs=action_pack["masked_obs"],
                actions=action_pack["actions"],
                log_probs=action_pack["log_probs"],
                reward_values=action_pack["reward_values"],
                cost_values=action_pack["cost_values"],
                rewards=reward_list,
                costs=train_cost_list,
                dones=done_list,
            )
            if (step_count + 1) % args.rollout_steps == 0:
                last_update = system.update(min_rollout_steps=args.rollout_steps)
                if last_update:
                    update_history.append({"step": step_count + 1, **last_update})

            obs_list = next_obs_list
            episode_reward += float(sum(reward_list))
            step_count += 1
            if any(done_list):
                break

        # Flush any full rollout collected by all agents near the episode boundary.
        boundary_update = system.update(min_rollout_steps=args.rollout_steps)
        if boundary_update:
            last_update = boundary_update
            update_history.append({"step": step_count, **boundary_update})

        episode_rewards.append(episode_reward)
        episode_violation_steps.append(violation_steps)
        episode_lengths.append(step + 1)
        dual_history.append(float(system.duals.mean().item()))

        eval_stats = {}
        if (episode + 1) % args.eval_interval == 0:
            eval_stats = evaluate_navigation(
                system=system,
                env=env,
                n_episodes=args.eval_episodes,
                use_safety_filter=args.safety_filter,
                seed_base=700000 + args.seed * 1000 + episode * args.eval_episodes,
            )
            eval_stats["episode"] = episode + 1
            eval_history.append(eval_stats)
            if is_better_eval(eval_stats, best_eval):
                best_eval = dict(eval_stats)
                best_eval["selected_from"] = "periodic_eval"
                torch.save(
                    {
                        "episode": episode + 1,
                        "eval": best_eval,
                        "selection_rule": (
                            "min episode violation, then min step violation, "
                            "then min constraint sum, then max reward"
                        ),
                        "checkpoint": system.get_checkpoint_payload(),
                    },
                    best_checkpoint_path,
                )

        if (episode + 1) % args.log_interval == 0:
            train_viol = constraint_tracker.get_stats()["violation_rate"]
            print(
                f"{episode + 1:>6} | {np.mean(episode_rewards[-args.log_interval:]):>8.2f} | "
                f"{train_viol:>9.2%} | {dual_history[-1]:>6.2f} | "
                f"{last_update.get('mean_cost', float('nan')):>7.4f} | "
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
    )
    final_checkpoint_payload = system.get_checkpoint_payload()
    final_mean_dual = float(system.duals.mean().item())
    final_eval_for_best = dict(final_eval)
    final_eval_for_best["episode"] = args.n_episodes
    if is_better_eval(final_eval_for_best, best_eval):
        best_eval = dict(final_eval_for_best)
        best_eval["selected_from"] = "final_eval"
        torch.save(
            {
                "episode": args.n_episodes,
                "eval": best_eval,
                "selection_rule": (
                    "min episode violation, then min step violation, "
                    "then min constraint sum, then max reward"
                ),
                "checkpoint": final_checkpoint_payload,
            },
            best_checkpoint_path,
        )

    best_payload = torch.load(best_checkpoint_path, map_location=args.device, weights_only=False)
    system.load_checkpoint_payload(best_payload["checkpoint"])
    best_test_eval = evaluate_navigation(
        system=system,
        env=env,
        n_episodes=args.eval_episodes,
        use_safety_filter=args.safety_filter,
        seed_base=900000 + args.seed * 1000,
    )
    reward_auc = compute_normalized_auc(eval_history, "eval_reward", args.n_episodes)
    ep_violation_auc = compute_normalized_auc(eval_history, "eval_episode_violation_rate", args.n_episodes)
    step_violation_auc = compute_normalized_auc(eval_history, "eval_step_violation_rate", args.n_episodes)

    comm_stats = enrich_comm_stats(
        comm_stats={
            "updates": system.get_communication_stats()["updates"],
            "actor_messages": system.get_communication_stats()["neighbor_messages"],
            "actor_scalars": system.get_communication_stats()["neighbor_scalars"],
            "dual_messages": 0,
            "dual_scalars": 0,
            "mean_neighborhood_size": system.get_communication_stats()["mean_neighborhood_size"],
        },
        total_steps=step_count,
        total_episodes=args.n_episodes,
        wall_clock_sec=wall_clock_sec,
    )

    results = {
        "variant_label": variant_label,
        "baseline": "scal_mappo_l_adapted",
        "n_agents": args.n_agents,
        "episode_rewards": episode_rewards,
        "episode_violation_steps": episode_violation_steps,
        "episode_lengths": episode_lengths,
        "dual_history": dual_history,
        "update_history": update_history,
        "eval_history": eval_history,
        "final_eval": final_eval,
        "best_eval": best_eval,
        "best_test_eval": best_test_eval,
        "final_train_stats": final_train_stats,
        "wall_clock_sec": wall_clock_sec,
        "total_steps": step_count,
        "comm_stats": comm_stats,
        "reward_auc": reward_auc,
        "episode_violation_auc": ep_violation_auc,
        "step_violation_auc": step_violation_auc,
        "args": vars(args),
    }

    torch.save(results, os.path.join(exp_dir, "results.pt"))
    torch.save(final_checkpoint_payload, os.path.join(exp_dir, "final_checkpoint.pt"))
    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "variant_label": variant_label,
                "baseline": "scal_mappo_l_adapted",
                "n_agents": args.n_agents,
                "seed": args.seed,
                "k_hops": args.k_hops,
                "safety_filter": args.safety_filter,
                "final_train_reward_last100": float(np.mean(episode_rewards[-100:])),
                "final_train_violation_rate": float(final_train_stats["violation_rate"]),
                "final_eval_reward": float(final_eval["eval_reward"]),
                "final_eval_episode_violation_rate": float(final_eval["eval_episode_violation_rate"]),
                "final_eval_step_violation_rate": float(final_eval["eval_step_violation_rate"]),
                "final_eval_mean_constraint_sum": float(final_eval["eval_mean_constraint_sum"]),
                "best_eval_reward": float(best_eval["eval_reward"]),
                "best_eval_episode_violation_rate": float(best_eval["eval_episode_violation_rate"]),
                "best_eval_step_violation_rate": float(best_eval["eval_step_violation_rate"]),
                "best_eval_mean_constraint_sum": float(best_eval["eval_mean_constraint_sum"]),
                "best_eval_episode": int(best_eval["episode"]),
                "best_eval_selected_from": best_eval["selected_from"],
                "best_test_eval_reward": float(best_test_eval["eval_reward"]),
                "best_test_eval_episode_violation_rate": float(best_test_eval["eval_episode_violation_rate"]),
                "best_test_eval_step_violation_rate": float(best_test_eval["eval_step_violation_rate"]),
                "best_test_eval_mean_constraint_sum": float(best_test_eval["eval_mean_constraint_sum"]),
                "reward_auc": float(reward_auc),
                "episode_violation_auc": float(ep_violation_auc),
                "step_violation_auc": float(step_violation_auc),
                "mean_dual_final": final_mean_dual,
                "wall_clock_sec": float(wall_clock_sec),
                "total_steps": int(step_count),
                "comm_stats": comm_stats,
                "args": vars(args),
            },
            f,
            indent=2,
        )

    print("\n" + "=" * 100)
    print("Scal-MAPPO-L baseline complete")
    print("=" * 100)
    print(f"Final eval reward: {final_eval['eval_reward']:.2f}")
    print(f"Final eval episode violation: {final_eval['eval_episode_violation_rate']:.2%}")
    print(f"Final eval step violation: {final_eval['eval_step_violation_rate']:.2%}")
    print(
        f"Best eval @ ep {best_eval['episode']}: reward {best_eval['eval_reward']:.2f}, "
        f"episode violation {best_eval['eval_episode_violation_rate']:.2%}, "
        f"step violation {best_eval['eval_step_violation_rate']:.2%}"
    )
    print(
        f"Best test eval: reward {best_test_eval['eval_reward']:.2f}, "
        f"episode violation {best_test_eval['eval_episode_violation_rate']:.2%}, "
        f"step violation {best_test_eval['eval_step_violation_rate']:.2%}"
    )
    print(f"Mean final dual: {final_mean_dual:.3f}")
    print(f"Wall clock: {wall_clock_sec:.1f}s")
    print(f"Saved to: {exp_dir}")
    return results


def main():
    args = parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
