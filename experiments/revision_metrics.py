"""
Utilities for revision-grade experiment evaluation and aggregation.

These helpers intentionally separate:
- episode-level violation rate: fraction of episodes with at least one violating step
- step-level violation rate: fraction of environment steps with at least one violation
- cumulative communication totals and derived per-episode / per-step metrics
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np


def create_adjacency_matrix(
    n_agents: int,
    topology: str,
    connectivity: float,
    seed: int,
) -> np.ndarray:
    if topology == "full":
        adj = np.ones((n_agents, n_agents), dtype=np.float32)
        np.fill_diagonal(adj, 1.0)
        return adj

    if topology == "chain":
        adj = np.eye(n_agents, dtype=np.float32)
        for i in range(n_agents - 1):
            adj[i, i + 1] = 1.0
            adj[i + 1, i] = 1.0
        return adj

    rng = np.random.default_rng(seed)
    adj = np.zeros((n_agents, n_agents), dtype=np.float32)
    for i in range(n_agents - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if adj[i, j] == 0.0 and rng.random() < connectivity:
                adj[i, j] = 1.0
                adj[j, i] = 1.0
    np.fill_diagonal(adj, 1.0)
    return adj


def compute_normalized_auc(eval_history: List[Dict], key: str, end_episode: int) -> float:
    if not eval_history or end_episode <= 0:
        return float("nan")

    xs = np.array([item["episode"] for item in eval_history], dtype=float)
    ys = np.array([item[key] for item in eval_history], dtype=float)
    if xs.size == 1:
        return float(ys[0])
    return float(np.trapz(ys, xs) / max(end_episode, 1))


def enrich_comm_stats(
    comm_stats: Dict[str, float],
    total_steps: int,
    total_episodes: int,
    wall_clock_sec: float,
) -> Dict[str, float]:
    updates = max(int(comm_stats.get("updates", 0)), 1)
    total_steps = max(int(total_steps), 1)
    total_episodes = max(int(total_episodes), 1)

    total_scalars = float(comm_stats.get("actor_scalars", 0.0) + comm_stats.get("dual_scalars", 0.0))
    total_messages = float(comm_stats.get("actor_messages", 0.0) + comm_stats.get("dual_messages", 0.0))

    enriched = dict(comm_stats)
    enriched.update(
        {
            "total_scalars": total_scalars,
            "total_messages": total_messages,
            "total_scalars_per_episode": total_scalars / total_episodes,
            "total_messages_per_episode": total_messages / total_episodes,
            "total_scalars_per_step": total_scalars / total_steps,
            "total_messages_per_step": total_messages / total_steps,
            "latency_sec_per_step": float(wall_clock_sec) / total_steps,
            "latency_ms_per_step": 1000.0 * float(wall_clock_sec) / total_steps,
            "latency_sec_per_episode": float(wall_clock_sec) / total_episodes,
            "latency_ms_per_update": 1000.0 * float(wall_clock_sec) / updates,
        }
    )
    return enriched


def _append_filter_metrics(storage: Dict[str, List[float]], infos: Dict[str, dict]):
    for agent in sorted(infos.keys()):
        metrics = infos[agent].get("filter_metrics")
        if not metrics:
            continue
        storage["epsilon"].append(float(metrics["epsilon"]))
        storage["raw_proxy_violation"].append(float(metrics["raw_proxy_violation"]))
        storage["selected_proxy_violation"].append(float(metrics["selected_proxy_violation"]))
        storage["filter_applied"].append(float(metrics["filter_applied"]))
        storage["safe_action_exists"].append(float(metrics["safe_action_exists"]))


def summarize_filter_metrics(storage: Optional[Dict[str, List[float]]]) -> Dict[str, float]:
    if not storage or not storage.get("epsilon"):
        return {
            "filter_metric_count": 0,
            "epsilon_mean": 0.0,
            "epsilon_p95": 0.0,
            "raw_proxy_violation_mean": 0.0,
            "selected_proxy_violation_mean": 0.0,
            "filter_applied_rate": 0.0,
            "proxy_feasible_rate": 0.0,
        }

    epsilon = np.asarray(storage["epsilon"], dtype=float)
    raw_proxy = np.asarray(storage["raw_proxy_violation"], dtype=float)
    selected_proxy = np.asarray(storage["selected_proxy_violation"], dtype=float)
    filter_applied = np.asarray(storage["filter_applied"], dtype=float)
    feasible = np.asarray(storage["safe_action_exists"], dtype=float)
    return {
        "filter_metric_count": int(epsilon.size),
        "epsilon_mean": float(np.mean(epsilon)),
        "epsilon_p95": float(np.percentile(epsilon, 95)),
        "raw_proxy_violation_mean": float(np.mean(raw_proxy)),
        "selected_proxy_violation_mean": float(np.mean(selected_proxy)),
        "filter_applied_rate": float(np.mean(filter_applied)),
        "proxy_feasible_rate": float(np.mean(feasible)),
    }


def evaluate_navigation(
    system,
    env,
    n_episodes: int,
    use_safety_filter: bool,
    seed_base: int = 100000,
    collect_filter_metrics: bool = False,
    return_filter_arrays: bool = False,
) -> Dict[str, float]:
    episode_rewards = []
    episode_lengths = []
    episode_has_violation = []
    episode_violation_steps = []
    episode_constraint_sum = []
    total_violation_steps = 0
    total_steps = 0
    filter_storage = {
        "epsilon": [],
        "raw_proxy_violation": [],
        "selected_proxy_violation": [],
        "filter_applied": [],
        "safe_action_exists": [],
    } if collect_filter_metrics else None

    for eval_ep in range(n_episodes):
        obs, _ = env.reset(seed=seed_base + eval_ep)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]

        ep_reward = 0.0
        ep_length = 0
        ep_has_violation = False
        ep_violation_steps = 0
        ep_constraint_sum = 0.0

        for step in range(env.max_steps):
            actions = system.act(obs_list, deterministic=True)
            action_dict = {
                agent: int(np.argmax(actions[i]))
                for i, agent in enumerate(sorted(obs.keys()))
            }
            next_obs, rewards, terminations, truncations, infos = env.step(
                action_dict,
                use_safety_filter=use_safety_filter,
                policy_actions={agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))},
            )

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get("constraint", 0.0) for agent in sorted(infos.keys())]
            step_has_violation = any(c > 0 for c in constraint_list)
            if filter_storage is not None:
                _append_filter_metrics(filter_storage, infos)

            ep_reward += float(sum(reward_list))
            ep_length = step + 1
            ep_has_violation = ep_has_violation or step_has_violation
            ep_violation_steps += int(step_has_violation)
            ep_constraint_sum += float(sum(max(0.0, c) for c in constraint_list))

            total_violation_steps += int(step_has_violation)
            total_steps += 1

            obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            if any(terminations.values()) or any(truncations.values()):
                break

        episode_rewards.append(ep_reward)
        episode_lengths.append(ep_length)
        episode_has_violation.append(float(ep_has_violation))
        episode_violation_steps.append(ep_violation_steps)
        episode_constraint_sum.append(ep_constraint_sum)

    results = {
        "eval_reward": float(np.mean(episode_rewards)),
        "eval_episode_violation_rate": float(np.mean(episode_has_violation)),
        "eval_step_violation_rate": float(total_violation_steps / max(total_steps, 1)),
        "eval_mean_episode_violation_steps": float(np.mean(episode_violation_steps)),
        "eval_mean_constraint_sum": float(np.mean(episode_constraint_sum)),
        "eval_length": float(np.mean(episode_lengths)),
        "eval_total_steps": int(total_steps),
        "eval_total_violation_steps": int(total_violation_steps),
    }
    results.update(summarize_filter_metrics(filter_storage))
    if collect_filter_metrics and return_filter_arrays and filter_storage is not None:
        results.update(
            {
                "filter_epsilon_values": np.asarray(filter_storage["epsilon"], dtype=np.float32),
                "filter_raw_proxy_violation_values": np.asarray(filter_storage["raw_proxy_violation"], dtype=np.float32),
                "filter_selected_proxy_violation_values": np.asarray(filter_storage["selected_proxy_violation"], dtype=np.float32),
                "filter_applied_values": np.asarray(filter_storage["filter_applied"], dtype=np.float32),
                "filter_safe_action_exists_values": np.asarray(filter_storage["safe_action_exists"], dtype=np.float32),
            }
        )
    return results
