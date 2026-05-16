"""
Revision-grade warehouse benchmark script.

Supported algorithms:
- dlpac_independent
- dlpac_sparse_k2
- dlpac_full
- teamcomm
- mapo
- greedy
- random

This script adds comparative evidence and richer warehouse metrics:
- throughput
- success rate
- deadlock rate
- robot-collision / boundary-contact breakdown
- communication and latency summaries
- optional failure trace export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from baselines import MAPOSystem
from dl_pac_v2 import DLPACSystemV2
from envs import MultiRobotWarehouseEnv
from teamcomm_controller import build_teamcomm_controller, train_teamcomm

from revision_metrics import create_adjacency_matrix, enrich_comm_stats


def parse_args():
    parser = argparse.ArgumentParser(description="Revision-grade warehouse benchmark")
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=[
            "dlpac_independent",
            "dlpac_sparse_k2",
            "dlpac_full",
            "teamcomm",
            "mapo",
            "greedy",
            "random",
        ],
    )
    parser.add_argument("--n_agents", type=int, default=10)
    parser.add_argument("--n_shelves", type=int, default=8)
    parser.add_argument("--n_episodes", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=100)
    parser.add_argument("--eval_episodes", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_interval", type=int, default=50)
    parser.add_argument("--eval_interval", type=int, default=50)
    parser.add_argument("--update_interval", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./logs_v2/warehouse_benchmark_v2")
    parser.add_argument("--record_failure_trace", action="store_true")
    parser.add_argument("--deadlock_window", type=int, default=12)
    parser.add_argument("--deadlock_motion_eps", type=float, default=0.01)
    parser.add_argument("--use_structured_layout", action="store_true")
    parser.add_argument("--use_shelf_obstacles", action="store_true")
    parser.add_argument("--spawn_mode", type=str, default="uniform", choices=["uniform", "dock"])
    parser.add_argument("--dock_depth", type=float, default=2.0)
    parser.add_argument("--shelf_size", type=float, default=1.5)
    parser.add_argument("--aisle_width", type=float, default=2.5)
    parser.add_argument("--pickup_radius", type=float, default=0.0)
    parser.add_argument("--outer_margin", type=float, default=1.2)
    return parser.parse_args()


def safe_probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
    probs = F.softmax(logits, dim=-1)
    probs = torch.nan_to_num(probs, nan=1.0, posinf=1.0, neginf=0.0)
    probs = probs.clamp_min(1e-8)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def zero_comm_stats(total_steps: int, total_episodes: int, wall_clock_sec: float) -> Dict[str, float]:
    return enrich_comm_stats(
        {
            "updates": 0,
            "actor_messages": 0,
            "actor_scalars": 0,
            "dual_messages": 0,
            "dual_scalars": 0,
            "neighborhood_entries": 0,
            "mean_neighborhood_size": 0.0,
        },
        total_steps=total_steps,
        total_episodes=total_episodes,
        wall_clock_sec=wall_clock_sec,
    )


def build_dlpac_controller(args, env):
    if args.algorithm == "dlpac_independent":
        topology = "chain"
        k_hops = 1
        actor_consensus = False
    elif args.algorithm == "dlpac_sparse_k2":
        topology = "chain"
        k_hops = 2
        actor_consensus = True
    elif args.algorithm == "dlpac_full":
        topology = "full"
        k_hops = args.n_agents
        actor_consensus = True
    else:
        raise ValueError(f"Unsupported DL-PAC warehouse algorithm: {args.algorithm}")

    adjacency = create_adjacency_matrix(
        n_agents=args.n_agents,
        topology=topology,
        connectivity=0.5,
        seed=args.seed,
    )
    controller = DLPACSystemV2(
        n_agents=args.n_agents,
        obs_dims=[env.obs_dim] * args.n_agents,
        action_dims=[env.action_dim] * args.n_agents,
        adjacency_matrix=adjacency,
        k_hops=k_hops,
        device=args.device,
        actor_consensus=actor_consensus,
        dual_mode="local",
    )
    return controller, adjacency, {"topology": topology, "k_hops": k_hops, "actor_consensus": actor_consensus}


def build_mapo_controller(args, env):
    controller = MAPOSystem(
        n_agents=args.n_agents,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        device=args.device,
    )
    return controller


def greedy_actions(env: MultiRobotWarehouseEnv) -> List[int]:
    actions = []
    for robot_idx in range(env.n_agents):
        pos = env.robot_positions[robot_idx]
        target = None
        best_dist = float("inf")
        for shelf_idx, shelf in enumerate(env.shelves):
            if env.shelf_picked[shelf_idx]:
                continue
            dist = np.linalg.norm(pos - shelf)
            if dist < best_dist:
                best_dist = dist
                target = shelf

        if target is None:
            actions.append(4)
            continue

        delta = target - pos
        if np.linalg.norm(delta) < (env.shelf_size / 2 + env.agent_size):
            actions.append(4)
        elif abs(delta[0]) > abs(delta[1]):
            actions.append(3 if delta[0] > 0 else 2)
        else:
            actions.append(0 if delta[1] > 0 else 1)
    return actions


def select_eval_actions(algorithm: str, controller, obs_list: List[np.ndarray], env, rng, deterministic: bool):
    if algorithm == "random":
        return [int(rng.integers(env.action_dim)) for _ in range(env.n_agents)]
    if algorithm == "greedy":
        return greedy_actions(env)
    if algorithm == "mapo":
        obs_t = torch.FloatTensor(np.array(obs_list)).to(controller.device)
        with torch.no_grad():
            logits, _ = controller.policy(obs_t)
            probs = safe_probs_from_logits(logits)
            if deterministic:
                actions = torch.argmax(probs, dim=-1).cpu().numpy()
            else:
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample().cpu().numpy()
        return [int(a) for a in actions]
    if algorithm == "teamcomm":
        return controller.act(obs_list, deterministic=deterministic)

    action_vectors = controller.act(obs_list, deterministic=deterministic)
    return [int(np.argmax(vec)) for vec in action_vectors]


def plot_failure_trace(trace: Dict, path: str):
    plt.figure(figsize=(7.0, 7.0))
    shelves = np.asarray(trace["shelves"], dtype=float)
    if shelves.size:
        plt.scatter(shelves[:, 0], shelves[:, 1], marker="s", s=80, c="#8c6d31", label="Shelves")

    positions = np.asarray(trace["positions"], dtype=float)  # (T, n_agents, 2)
    n_agents = positions.shape[1]
    for agent_idx in range(n_agents):
        traj = positions[:, agent_idx, :]
        plt.plot(traj[:, 0], traj[:, 1], linewidth=1.6, alpha=0.85)
        plt.scatter(traj[0, 0], traj[0, 1], s=24, marker="o", color="black")
        plt.scatter(traj[-1, 0], traj[-1, 1], s=28, marker="x", color="red")

    plt.title(trace["title"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.axis("equal")
    plt.grid(alpha=0.25)
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def evaluate_warehouse(
    algorithm: str,
    controller,
    env: MultiRobotWarehouseEnv,
    n_episodes: int,
    seed_base: int,
    deadlock_window: int,
    deadlock_motion_eps: float,
    record_failure_trace: bool,
) -> Tuple[Dict[str, float], Optional[Dict]]:
    rng = np.random.default_rng(seed_base)

    rewards = []
    lengths = []
    items_picked = []
    success_flags = []
    deadlock_flags = []
    robot_collision_steps = []
    boundary_contact_steps = []
    total_eval_steps = 0
    failure_trace = None

    for eval_ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + eval_ep)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        prev_positions = env.robot_positions.copy()
        half_size = env.warehouse_size / 2 - env.agent_size

        ep_reward = 0.0
        ep_length = 0
        ep_collision_steps = 0
        ep_boundary_steps = 0
        stagnant_steps = 0
        deadlocked = False
        last_items_picked = 0
        trace_positions = [env.robot_positions.copy()]
        trace_items = [0]
        trace_collision = [0]

        for step in range(env.max_steps):
            actions = select_eval_actions(
                algorithm=algorithm,
                controller=controller,
                obs_list=obs_list,
                env=env,
                rng=rng,
                deterministic=True,
            )
            action_dict = {agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))}

            next_obs, rewards_dict, terminations, truncations, infos = env.step(action_dict)
            reward_list = [rewards_dict[agent] for agent in sorted(rewards_dict.keys())]
            step_collision = any(infos[agent].get("constraint", 0.0) > 0 for agent in sorted(infos.keys()))
            step_boundary = bool(np.any(np.abs(env.robot_positions) >= half_size - 1e-6))
            picked_now = max(infos[agent].get("items_picked", 0) for agent in sorted(infos.keys()))
            mean_motion = float(np.mean(np.linalg.norm(env.robot_positions - prev_positions, axis=1)))

            if picked_now > last_items_picked:
                stagnant_steps = 0
            elif mean_motion < deadlock_motion_eps:
                stagnant_steps += 1
            else:
                stagnant_steps = 0
            if stagnant_steps >= deadlock_window:
                deadlocked = True

            ep_reward += float(sum(reward_list))
            ep_length = step + 1
            ep_collision_steps += int(step_collision)
            ep_boundary_steps += int(step_boundary)
            total_eval_steps += 1
            prev_positions = env.robot_positions.copy()
            last_items_picked = picked_now
            obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            if record_failure_trace:
                trace_positions.append(env.robot_positions.copy())
                trace_items.append(int(picked_now))
                trace_collision.append(int(step_collision))

            if any(terminations.values()) or any(truncations.values()):
                break

        success = bool(all(env.shelf_picked))
        rewards.append(ep_reward)
        lengths.append(ep_length)
        items_picked.append(float(sum(env.shelf_picked)))
        success_flags.append(float(success))
        deadlock_flags.append(float(deadlocked and not success))
        robot_collision_steps.append(ep_collision_steps)
        boundary_contact_steps.append(ep_boundary_steps)

        if failure_trace is None and record_failure_trace and (not success or deadlocked or ep_collision_steps > 0):
            failure_trace = {
                "title": f"{algorithm} failure trace (eval episode {eval_ep})",
                "positions": np.asarray(trace_positions).tolist(),
                "items_picked": trace_items,
                "collision_steps": trace_collision,
                "shelves": [np.asarray(shelf).tolist() for shelf in env.shelves],
            }

    metrics = {
        "eval_reward": float(np.mean(rewards)),
        "throughput_items_per_episode": float(np.mean(items_picked)),
        "throughput_items_per_step": float(np.sum(items_picked) / max(total_eval_steps, 1)),
        "success_rate": float(np.mean(success_flags)),
        "deadlock_rate": float(np.mean(deadlock_flags)),
        "robot_collision_step_rate": float(np.sum(robot_collision_steps) / max(total_eval_steps, 1)),
        "boundary_contact_step_rate": float(np.sum(boundary_contact_steps) / max(total_eval_steps, 1)),
        "mean_episode_length": float(np.mean(lengths)),
        "mean_final_items_picked": float(np.mean(items_picked)),
        "total_eval_steps": int(total_eval_steps),
    }
    return metrics, failure_trace


def train_dlpac(args, controller, env):
    episode_rewards = []
    total_steps = 0
    eval_history = []

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        episode_reward = 0.0

        for step in range(args.max_steps):
            actions = controller.act(obs_list, deterministic=False)
            action_dict = {
                agent: int(np.argmax(actions[i]))
                for i, agent in enumerate(sorted(obs.keys()))
            }

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get("constraint", 0.0) for agent in sorted(infos.keys())]

            controller.step(
                obs_list=obs_list,
                action_list=actions,
                reward_list=reward_list,
                constraint_list=constraint_list,
                next_obs_list=next_obs_list,
                done_list=[terminations[agent] or truncations[agent] for agent in sorted(terminations.keys())],
                agent_positions=env.robot_positions,
            )

            if (step + 1) % args.update_interval == 0:
                controller.update_all()

            episode_reward += float(sum(reward_list))
            obs_list = next_obs_list
            total_steps += 1

            if any(terminations.values()) or any(truncations.values()):
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % args.eval_interval == 0:
            eval_stats, _ = evaluate_warehouse(
                algorithm=args.algorithm,
                controller=controller,
                env=env,
                n_episodes=min(20, args.eval_episodes),
                seed_base=500000 + args.seed * 1000 + episode * 20,
                deadlock_window=args.deadlock_window,
                deadlock_motion_eps=args.deadlock_motion_eps,
                record_failure_trace=False,
            )
            eval_stats["episode"] = episode + 1
            eval_history.append(eval_stats)

        if (episode + 1) % args.log_interval == 0:
            print(
                f"{episode + 1:>6} | reward={np.mean(episode_rewards[-args.log_interval:]):>8.2f}"
            )

    return {
        "episode_rewards": episode_rewards,
        "eval_history": eval_history,
        "total_steps": total_steps,
        "comm_stats": controller.get_communication_stats(),
    }


def train_mapo(args, controller, env):
    episode_rewards = []
    total_steps = 0
    eval_history = []

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        episode_reward = 0.0

        for step in range(args.max_steps):
            obs_t = torch.FloatTensor(np.array(obs_list)).to(controller.device)
            with torch.no_grad():
                logits, values = controller.policy(obs_t)
                probs = safe_probs_from_logits(logits)
                dist = torch.distributions.Categorical(probs)
                actions = dist.sample()

            action_dict = {
                agent: int(actions[i].item())
                for i, agent in enumerate(sorted(obs.keys()))
            }
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get("constraint", 0.0) for agent in sorted(infos.keys())]

            for i in range(args.n_agents):
                controller.store(
                    agent_id=i,
                    obs=obs_list[i],
                    action=int(actions[i].item()),
                    reward=float(reward_list[i]),
                    constraint=float(constraint_list[i]),
                    old_log_prob=float(torch.log(probs[i, actions[i]] + 1e-8).item()),
                    value=float(values[i].item()),
                )

            if (step + 1) % args.update_interval == 0:
                controller.update()

            obs_list = next_obs_list
            episode_reward += float(sum(reward_list))
            total_steps += 1

            if any(terminations.values()) or any(truncations.values()):
                break

        episode_rewards.append(episode_reward)

        if (episode + 1) % args.eval_interval == 0:
            eval_stats, _ = evaluate_warehouse(
                algorithm="mapo",
                controller=controller,
                env=env,
                n_episodes=min(20, args.eval_episodes),
                seed_base=700000 + args.seed * 1000 + episode * 20,
                deadlock_window=args.deadlock_window,
                deadlock_motion_eps=args.deadlock_motion_eps,
                record_failure_trace=False,
            )
            eval_stats["episode"] = episode + 1
            eval_history.append(eval_stats)

        if (episode + 1) % args.log_interval == 0:
            print(f"{episode + 1:>6} | reward={np.mean(episode_rewards[-args.log_interval:]):>8.2f}")

    return {
        "episode_rewards": episode_rewards,
        "eval_history": eval_history,
        "total_steps": total_steps,
        "comm_stats": zero_comm_stats(total_steps, args.n_episodes, 0.0),
    }


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    env = MultiRobotWarehouseEnv(
        n_agents=args.n_agents,
        n_shelves=args.n_shelves,
        max_steps=args.max_steps,
        collision_threshold=0.8,
        shelf_size=args.shelf_size,
        aisle_width=args.aisle_width,
        use_structured_layout=args.use_structured_layout,
        use_shelf_obstacles=args.use_shelf_obstacles,
        spawn_mode=args.spawn_mode,
        dock_depth=args.dock_depth,
        pickup_radius=args.pickup_radius,
        outer_margin=args.outer_margin,
    )

    controller = None
    adjacency = None
    config_info = {}
    if args.algorithm.startswith("dlpac_"):
        controller, adjacency, config_info = build_dlpac_controller(args, env)
    elif args.algorithm == "teamcomm":
        controller, adjacency, config_info = build_teamcomm_controller(args, env)
    elif args.algorithm == "mapo":
        controller = build_mapo_controller(args, env)

    exp_dir = os.path.join(args.save_dir, f"{args.algorithm}_n{args.n_agents}_s{args.seed}")
    os.makedirs(exp_dir, exist_ok=True)

    with open(os.path.join(exp_dir, "run_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "algorithm": args.algorithm,
                "args": vars(args),
                "config_info": config_info,
                "adjacency": adjacency.tolist() if adjacency is not None else None,
            },
            f,
            indent=2,
        )

    start_time = time.perf_counter()
    if args.algorithm.startswith("dlpac_"):
        train_result = train_dlpac(args, controller, env)
    elif args.algorithm == "teamcomm":
        train_result = train_teamcomm(args, controller, env, evaluate_warehouse)
    elif args.algorithm == "mapo":
        train_result = train_mapo(args, controller, env)
    else:
        train_result = {
            "episode_rewards": [],
            "eval_history": [],
            "total_steps": 0,
            "comm_stats": zero_comm_stats(1, 1, 0.0),
        }

    wall_clock_sec = time.perf_counter() - start_time
    if args.algorithm.startswith("dlpac_"):
        comm_stats = enrich_comm_stats(
            controller.get_communication_stats(),
            total_steps=train_result["total_steps"],
            total_episodes=max(args.n_episodes, 1),
            wall_clock_sec=wall_clock_sec,
        )
    elif args.algorithm == "teamcomm":
        comm_stats = enrich_comm_stats(
            train_result["comm_stats"],
            total_steps=max(train_result["total_steps"], 1),
            total_episodes=max(args.n_episodes, 1),
            wall_clock_sec=wall_clock_sec,
        )
    else:
        comm_stats = zero_comm_stats(
            total_steps=max(train_result["total_steps"], 1),
            total_episodes=max(args.n_episodes, 1),
            wall_clock_sec=wall_clock_sec,
        )

    eval_metrics, failure_trace = evaluate_warehouse(
        algorithm=args.algorithm,
        controller=controller,
        env=env,
        n_episodes=args.eval_episodes,
        seed_base=900000 + args.seed * 1000,
        deadlock_window=args.deadlock_window,
        deadlock_motion_eps=args.deadlock_motion_eps,
        record_failure_trace=args.record_failure_trace,
    )

    if failure_trace is not None:
        trace_json = os.path.join(exp_dir, "failure_trace.json")
        trace_png = os.path.join(exp_dir, "failure_trace.png")
        with open(trace_json, "w", encoding="utf-8") as f:
            json.dump(failure_trace, f, indent=2)
        plot_failure_trace(failure_trace, trace_png)

    if args.algorithm == "teamcomm":
        controller.save_checkpoint(os.path.join(exp_dir, "teamcomm_checkpoint.pt"))

    summary = {
        "algorithm": args.algorithm,
        "n_agents": args.n_agents,
        "seed": args.seed,
        "wall_clock_sec": float(wall_clock_sec),
        "total_steps": int(train_result["total_steps"]),
        "eval_reward": float(eval_metrics["eval_reward"]),
        "throughput_items_per_episode": float(eval_metrics["throughput_items_per_episode"]),
        "throughput_items_per_step": float(eval_metrics["throughput_items_per_step"]),
        "success_rate": float(eval_metrics["success_rate"]),
        "deadlock_rate": float(eval_metrics["deadlock_rate"]),
        "robot_collision_step_rate": float(eval_metrics["robot_collision_step_rate"]),
        "boundary_contact_step_rate": float(eval_metrics["boundary_contact_step_rate"]),
        "mean_episode_length": float(eval_metrics["mean_episode_length"]),
        "mean_final_items_picked": float(eval_metrics["mean_final_items_picked"]),
        "comm_stats": comm_stats,
        "args": vars(args),
    }

    torch.save(
        {
            "train_result": train_result,
            "eval_metrics": eval_metrics,
            "comm_stats": comm_stats,
            "wall_clock_sec": wall_clock_sec,
            "args": vars(args),
        },
        os.path.join(exp_dir, "results.pt"),
    )
    with open(os.path.join(exp_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("=== Warehouse Benchmark V2 Complete ===")
    print(f"Algorithm: {args.algorithm}")
    print(f"Eval reward: {eval_metrics['eval_reward']:.2f}")
    print(f"Items/episode: {eval_metrics['throughput_items_per_episode']:.2f}")
    print(f"Success rate: {eval_metrics['success_rate']:.2%}")
    print(f"Deadlock rate: {eval_metrics['deadlock_rate']:.2%}")
    print(f"Robot-collision step rate: {eval_metrics['robot_collision_step_rate']:.2%}")
    print(f"Boundary-contact step rate: {eval_metrics['boundary_contact_step_rate']:.2%}")
    print(f"Saved to: {exp_dir}")


if __name__ == "__main__":
    main()
