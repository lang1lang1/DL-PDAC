"""
Baseline Training Script
Trains and evaluates:
  1. Safe MADDPG (MADDPG + CBF)
  2. Lagrangian PPO
  3. DL-PDAC v2 (for comparison)

Metrics: reward, violation_rate, episode_length
"""

import torch
import torch.nn.functional as F
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baselines import SafeMADDPG, MAPOSystem, LagrangianPPOAgent
from dl_pac_v2 import DLPACSystemV2
from envs import MultiAgentParticleEnv, ConstraintTracker


def parse_args():
    parser = argparse.ArgumentParser(description='Baseline Training')
    parser.add_argument('--algorithm', type=str, default='all',
                        choices=['all', 'safe_maddpg', 'lagrangian_ppo', 'dl_pac_v2'],
                        help='Which algorithm to train')
    parser.add_argument('--n_agents', type=int, default=4)
    parser.add_argument('--n_episodes', type=int, default=500)
    parser.add_argument('--max_steps', type=int, default=100)
    parser.add_argument('--k_hops', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--eval_interval', type=int, default=200)
    parser.add_argument('--eval_episodes', type=int, default=10)
    parser.add_argument('--update_interval', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lyapunov_coef', type=float, default=0.5)
    parser.add_argument('--dual_lr', type=float, default=0.01)
    parser.add_argument('--cbf_alpha', type=float, default=3.0)
    parser.add_argument('--save_dir', type=str, default='./logs/baselines')
    return parser.parse_args()


def create_adjacency_matrix(n_agents: int, seed: int = 0) -> np.ndarray:
    """Create chain-connected adjacency matrix."""
    np.random.seed(seed)
    adj = np.zeros((n_agents, n_agents))
    for i in range(n_agents - 1):
        adj[i, i + 1] = adj[i + 1, i] = 1
    for i in range(n_agents):
        for j in range(i + 1, n_agents):
            if np.random.random() < 0.3 and adj[i, j] == 0:
                adj[i, j] = adj[j, i] = 1
    np.fill_diagonal(adj, 1)
    return adj


def evaluate_safe_maddpg(
    system: SafeMADDPG,
    env: MultiAgentParticleEnv,
    n_episodes: int = 10,
    use_cbf: bool = True,
) -> Dict[str, float]:
    """Evaluate Safe MADDPG."""
    eval_rewards = []
    eval_violations = []
    eval_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100000))
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        positions = env.agent_positions.copy()

        episode_reward = 0
        step = 0

        for step in range(env.max_steps):
            actions = system.act(obs_list, positions, deterministic=True, use_cbf=use_cbf)
            action_dict = {agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))}

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            next_positions = env.agent_positions.copy()

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            episode_reward += sum(reward_list)
            obs_list = next_obs_list
            positions = next_positions

            if any(terminations.values()) or any(truncations.values()):
                break

        constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]
        eval_rewards.append(episode_reward)
        eval_violations.append(any(c > 0 for c in constraint_list))
        eval_lengths.append(step + 1)

    return {
        'eval_reward': np.mean(eval_rewards),
        'eval_violation_rate': np.mean(eval_violations),
        'eval_length': np.mean(eval_lengths),
    }


def evaluate_lagrangian_ppo(
    system: MAPOSystem,
    env: MultiAgentParticleEnv,
    n_episodes: int = 10,
    safety_filter: bool = True,
) -> Dict[str, float]:
    """Evaluate Lagrangian PPO."""
    eval_rewards = []
    eval_violations = []
    eval_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100000))
        obs_list = [obs[agent] for agent in sorted(obs.keys())]

        episode_reward = 0
        step = 0

        for step in range(env.max_steps):
            # Use shared policy
            obs_t = torch.FloatTensor(np.array(obs_list)).to(system.device)
            with torch.no_grad():
                logits, _ = system.policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                actions = torch.argmax(probs, dim=-1).cpu().numpy()

            action_dict = {agent: int(actions[i]) for i, agent in enumerate(sorted(obs.keys()))}

            next_obs, rewards, terminations, truncations, infos = env.step(
                action_dict, use_safety_filter=safety_filter,
                policy_actions={agent: probs.cpu().numpy()[i] for i, agent in enumerate(sorted(obs.keys()))}
            )
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            episode_reward += sum(reward_list)
            obs_list = next_obs_list

            if any(terminations.values()) or any(truncations.values()):
                break

        constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]
        eval_rewards.append(episode_reward)
        eval_violations.append(any(c > 0 for c in constraint_list))
        eval_lengths.append(step + 1)

    return {
        'eval_reward': np.mean(eval_rewards),
        'eval_violation_rate': np.mean(eval_violations),
        'eval_length': np.mean(eval_lengths),
    }


def evaluate_dl_pac_v2(
    system: DLPACSystemV2,
    env: MultiAgentParticleEnv,
    n_episodes: int = 10,
    safety_filter: bool = True,
) -> Dict[str, float]:
    """Evaluate DL-PDAC v2."""
    eval_rewards = []
    eval_violations = []
    eval_lengths = []

    for _ in range(n_episodes):
        obs, _ = env.reset(seed=np.random.randint(100000))
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        episode_reward = 0
        step = 0

        for step in range(env.max_steps):
            actions = system.act(obs_list, deterministic=True)
            action_dict = {
                agent: int(np.argmax(actions[i]))
                for i, agent in enumerate(sorted(obs.keys()))
            }
            next_obs, rewards, terminations, truncations, infos = env.step(
                action_dict, use_safety_filter=safety_filter,
                policy_actions={agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))}
            )
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            episode_reward += sum(reward_list)
            obs_list = next_obs_list

            if any(terminations.values()) or any(truncations.values()):
                break

        constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]
        eval_rewards.append(episode_reward)
        eval_violations.append(any(c > 0 for c in constraint_list))
        eval_lengths.append(step + 1)

    return {
        'eval_reward': np.mean(eval_rewards),
        'eval_violation_rate': np.mean(eval_violations),
        'eval_length': np.mean(eval_lengths),
    }


def train_safe_maddpg(args, seed: int) -> Dict:
    """Train Safe MADDPG."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiAgentParticleEnv(
        n_agents=args.n_agents,
        n_landmarks=2,
        max_steps=args.max_steps,
    )

    obs_dims = [env.obs_dim] * args.n_agents
    action_dims = [env.action_dim] * args.n_agents

    system = SafeMADDPG(
        n_agents=args.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=args.device,
        actor_lr=1e-3,
        critic_lr=1e-3,
        gamma=0.99,
        tau=0.01,
        collision_threshold=env.collision_threshold,
        cbf_alpha=args.cbf_alpha,
        batch_size=args.batch_size,
    )

    constraint_tracker = ConstraintTracker()
    episode_rewards = []
    episode_violations = []
    eval_rewards_curve = []
    eval_violations_curve = []
    step_count = 0

    print(f"\n{'Ep':>6} | {'TrainR':>8} | {'ViolR':>7} | {'CriticL':>8} | {'EvalR':>8} | {'EvalViol':>8}")
    print("-" * 70)

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        positions = env.agent_positions.copy()
        constraint_tracker.reset_episode()
        episode_reward = 0

        for step in range(args.max_steps):
            actions = system.act(obs_list, positions, deterministic=False, use_cbf=True)
            action_dict = {agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))}

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            next_positions = env.agent_positions.copy()

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]

            constraint_dict = {agent: infos[agent].get('constraint', 0) for agent in infos.keys()}
            constraint_tracker.add(constraint_dict)

            # Store in replay buffer (pack all agents' obs and actions)
            all_obs = np.array([obs_list[i] for i in range(args.n_agents)])
            all_actions_raw = np.array([actions[i] for i in range(args.n_agents)], dtype=np.int64)  # (n_agents,)
            all_next_obs = np.array([next_obs_list[i] for i in range(args.n_agents)])

            system.store(
                all_obs=all_obs,
                all_actions=all_actions_raw,  # raw integers; SafeMADDPG.update converts to one-hot
                all_rewards=reward_list,
                all_next_obs=all_next_obs,
                dones=[terminations[agent] or truncations[agent] for agent in sorted(terminations.keys())]
            )

            obs_list = next_obs_list
            positions = next_positions
            episode_reward += sum(reward_list)
            step_count += 1

            if (step + 1) % args.update_interval == 0 and len(system.replay_buffer['all_obs']) >= args.batch_size:
                system.update()

            if any(terminations.values()) or any(truncations.values()):
                break

        episode_rewards.append(episode_reward)
        episode_violations.append(constraint_tracker.get_stats()['violation_rate'])

        # Evaluation
        eval_stats = {}
        if (episode + 1) % args.eval_interval == 0:
            eval_stats = evaluate_safe_maddpg(system, env, args.eval_episodes, use_cbf=True)
            eval_rewards_curve.append(eval_stats['eval_reward'])
            eval_violations_curve.append(eval_stats['eval_violation_rate'])

        # Logging
        if (episode + 1) % args.log_interval == 0:
            recent_rewards = episode_rewards[-args.log_interval:]
            viol_rate = constraint_tracker.get_stats()['violation_rate']
            eval_str = (f"{eval_stats.get('eval_reward', 0):8.2f} | "
                        f"{eval_stats.get('eval_violation_rate', 0):7.0%}" if eval_stats
                        else f"{'N/A':>8} | {'N/A':>7}")
            print(f"{episode+1:>6} | {np.mean(recent_rewards):>8.2f} | {viol_rate:>7.0%} | "
                  f"{'N/A':>8} | {eval_str}")

    print(f"\nSafe MADDPG training complete. Steps: {step_count}")
    final_stats = constraint_tracker.get_stats()
    print(f"Final Violation Rate: {final_stats['violation_rate']:.2%}")

    return {
        'episode_rewards': episode_rewards,
        'episode_violations': episode_violations,
        'eval_rewards': eval_rewards_curve,
        'eval_violations': eval_violations_curve,
        'final_stats': final_stats,
        'step_count': step_count,
    }


def train_lagrangian_ppo(args, seed: int) -> Dict:
    """Train Lagrangian PPO."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = MultiAgentParticleEnv(
        n_agents=args.n_agents,
        n_landmarks=2,
        max_steps=args.max_steps,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_dim

    system = MAPOSystem(
        n_agents=args.n_agents,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=args.device,
    )

    constraint_tracker = ConstraintTracker()
    episode_rewards = []
    episode_violations = []
    eval_rewards_curve = []
    eval_violations_curve = []
    step_count = 0
    update_count = 0

    print(f"\n{'Ep':>6} | {'TrainR':>8} | {'ViolR':>7} | {'Dual':>6} | {'EvalR':>8} | {'EvalViol':>8}")
    print("-" * 70)

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=seed + episode)
        obs_list = [obs[agent] for agent in sorted(obs.keys())]
        constraint_tracker.reset_episode()
        episode_reward = 0

        for step in range(args.max_steps):
            # Act
            obs_t = torch.FloatTensor(np.array(obs_list)).to(args.device)
            with torch.no_grad():
                logits, values = system.policy(obs_t)
                probs = F.softmax(logits, dim=-1)
                dists = [torch.distributions.Categorical(p) for p in probs]
                actions_gpu = torch.stack([d.sample() for d in dists])
                actions = actions_gpu.cpu().numpy()
                log_probs = torch.stack([dists[i].log_prob(actions_gpu[i]) for i in range(len(dists))])
                values_list = values.squeeze(-1).detach().cpu().numpy()

            action_dict = {agent: int(actions[i]) for i, agent in enumerate(sorted(obs.keys()))}

            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)
            next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

            reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
            constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]

            constraint_dict = {agent: infos[agent].get('constraint', 0) for agent in infos.keys()}
            constraint_tracker.add(constraint_dict)

            for i, agent in enumerate(sorted(obs.keys())):
                system.store(
                    agent_id=i,
                    obs=obs_list[i],
                    action=int(actions[i]),
                    reward=reward_list[i],
                    constraint=constraint_list[i],
                    old_log_prob=log_probs[i].item(),
                    value=values_list[i],
                )

            obs_list = next_obs_list
            episode_reward += sum(reward_list)
            step_count += 1

            if (step + 1) % args.update_interval == 0:
                results = system.update()
                update_count += 1

            if any(terminations.values()) or any(truncations.values()):
                break

        episode_rewards.append(episode_reward)
        episode_violations.append(constraint_tracker.get_stats()['violation_rate'])

        # Evaluation
        eval_stats = {}
        if (episode + 1) % args.eval_interval == 0:
            eval_stats = evaluate_lagrangian_ppo(system, env, args.eval_episodes, safety_filter=False)
            eval_rewards_curve.append(eval_stats['eval_reward'])
            eval_violations_curve.append(eval_stats['eval_violation_rate'])

        # Logging
        if (episode + 1) % args.log_interval == 0:
            recent_rewards = episode_rewards[-args.log_interval:]
            viol_rate = constraint_tracker.get_stats()['violation_rate']
            mean_dual = np.mean([d.item() if hasattr(d, 'item') else d for d in system.duals])
            eval_str = (f"{eval_stats.get('eval_reward', 0):8.2f} | "
                        f"{eval_stats.get('eval_violation_rate', 0):7.0%}" if eval_stats
                        else f"{'N/A':>8} | {'N/A':>7}")
            print(f"{episode+1:>6} | {np.mean(recent_rewards):>8.2f} | {viol_rate:>7.0%} | "
                  f"{mean_dual:>6.3f} | {eval_str}")

    print(f"\nLagrangian PPO training complete. Steps: {step_count}, Updates: {update_count}")
    final_stats = constraint_tracker.get_stats()
    print(f"Final Violation Rate: {final_stats['violation_rate']:.2%}")

    return {
        'episode_rewards': episode_rewards,
        'episode_violations': episode_violations,
        'eval_rewards': eval_rewards_curve,
        'eval_violations': eval_violations_curve,
        'final_stats': final_stats,
        'step_count': step_count,
    }


def run_all_baselines(args):
    """Run all algorithms and compare."""
    seeds = args.seeds
    os.makedirs(args.save_dir, exist_ok=True)

    # DL-PDAC v2 evaluation points
    eval_points = list(range(0, args.n_episodes + 1, args.eval_interval))
    eval_points = [p for p in eval_points if p > 0]

    all_results = {
        'safe_maddpg': {'seeds': [], 'eval_rewards': [], 'eval_violations': []},
        'lagrangian_ppo': {'seeds': [], 'eval_rewards': [], 'eval_violations': []},
        'dl_pac_v2': {'seeds': [], 'eval_rewards': [], 'eval_violations': []},
    }

    # === Safe MADDPG ===
    if args.algorithm in ['all', 'safe_maddpg']:
        print("\n" + "=" * 80)
        print("TRAINING: Safe MADDPG (MADDPG + CBF)")
        print("=" * 80)

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            result = train_safe_maddpg(args, seed)

            save_path = os.path.join(args.save_dir, f'safe_maddpg_n{args.n_agents}_s{seed}.pt')
            torch.save({
                'result': result,
                'args': vars(args),
                'seed': seed,
            }, save_path)
            print(f"Saved: {save_path}")

            all_results['safe_maddpg']['seeds'].append(result)

    # === Lagrangian PPO ===
    if args.algorithm in ['all', 'lagrangian_ppo']:
        print("\n" + "=" * 80)
        print("TRAINING: Lagrangian PPO")
        print("=" * 80)

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")
            result = train_lagrangian_ppo(args, seed)

            save_path = os.path.join(args.save_dir, f'lagrangian_ppo_n{args.n_agents}_s{seed}.pt')
            torch.save({
                'result': result,
                'args': vars(args),
                'seed': seed,
            }, save_path)
            print(f"Saved: {save_path}")

            all_results['lagrangian_ppo']['seeds'].append(result)

    # === DL-PDAC v2 ===
    if args.algorithm in ['all', 'dl_pac_v2']:
        print("\n" + "=" * 80)
        print("TRAINING: DL-PDAC v2 (for comparison)")
        print("=" * 80)

        for seed in seeds:
            print(f"\n--- Seed {seed} ---")

            torch.manual_seed(seed)
            np.random.seed(seed)

            env = MultiAgentParticleEnv(
                n_agents=args.n_agents,
                n_landmarks=2,
                max_steps=args.max_steps,
            )
            adjacency = create_adjacency_matrix(args.n_agents, seed)
            obs_dims = [env.obs_dim] * args.n_agents
            action_dims = [env.action_dim] * args.n_agents

            system = DLPACSystemV2(
                n_agents=args.n_agents,
                obs_dims=obs_dims,
                action_dims=action_dims,
                adjacency_matrix=adjacency,
                k_hops=args.k_hops,
                device=args.device,
                lyapunov_coef=args.lyapunov_coef,
                dual_lr=args.dual_lr,
            )

            constraint_tracker = ConstraintTracker()
            episode_rewards = []
            episode_violations = []
            eval_rewards_curve = []
            eval_violations_curve = []
            step_count = 0

            print(f"\n{'Ep':>6} | {'TrainR':>8} | {'ViolR':>7} | {'Dual':>6} | {'EvalR':>8} | {'EvalViol':>8}")
            print("-" * 70)

            for episode in range(args.n_episodes):
                obs, _ = env.reset(seed=seed + episode)
                obs_list = [obs[agent] for agent in sorted(obs.keys())]
                constraint_tracker.reset_episode()
                episode_reward = 0

                for step in range(args.max_steps):
                    actions = system.act(obs_list, deterministic=False)
                    action_dict = {
                        agent: int(np.argmax(actions[i]))
                        for i, agent in enumerate(sorted(obs.keys()))
                    }

                    next_obs, rewards, terminations, truncations, infos = env.step(
                        action_dict, use_safety_filter=True,
                        policy_actions={agent: actions[i] for i, agent in enumerate(sorted(obs.keys()))}
                    )
                    next_obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]

                    reward_list = [rewards[agent] for agent in sorted(rewards.keys())]
                    constraint_list = [infos[agent].get('constraint', 0) for agent in sorted(infos.keys())]

                    constraint_dict = {agent: infos[agent].get('constraint', 0) for agent in infos.keys()}
                    constraint_tracker.add(constraint_dict)

                    system.step(
                        obs_list=obs_list,
                        action_list=actions,
                        reward_list=reward_list,
                        constraint_list=constraint_list,
                        next_obs_list=next_obs_list,
                        done_list=[terminations[agent] or truncations[agent]
                                   for agent in sorted(terminations.keys())]
                    )

                    if (step + 1) % args.update_interval == 0:
                        system.update_all()

                    obs_list = next_obs_list
                    episode_reward += sum(reward_list)
                    step_count += 1

                    if any(terminations.values()) or any(truncations.values()):
                        break

                episode_rewards.append(episode_reward)
                episode_violations.append(constraint_tracker.get_stats()['violation_rate'])

                eval_stats = {}
                if (episode + 1) % args.eval_interval == 0:
                    eval_stats = evaluate_dl_pac_v2(system, env, args.eval_episodes, safety_filter=True)
                    eval_rewards_curve.append(eval_stats['eval_reward'])
                    eval_violations_curve.append(eval_stats['eval_violation_rate'])

                if (episode + 1) % args.log_interval == 0:
                    recent_rewards = episode_rewards[-args.log_interval:]
                    viol_rate = constraint_tracker.get_stats()['violation_rate']
                    mean_dual = np.mean([a.dual.item() for a in system.agents])
                    eval_str = (f"{eval_stats.get('eval_reward', 0):8.2f} | "
                                f"{eval_stats.get('eval_violation_rate', 0):7.0%}" if eval_stats
                                else f"{'N/A':>8} | {'N/A':>7}")
                    print(f"{episode+1:>6} | {np.mean(recent_rewards):>8.2f} | {viol_rate:>7.0%} | "
                          f"{mean_dual:>6.3f} | {eval_str}")

            print(f"\nDL-PDAC v2 training complete. Steps: {step_count}")
            final_stats = constraint_tracker.get_stats()
            print(f"Final Violation Rate: {final_stats['violation_rate']:.2%}")

            result = {
                'episode_rewards': episode_rewards,
                'episode_violations': episode_violations,
                'eval_rewards': eval_rewards_curve,
                'eval_violations': eval_violations_curve,
                'final_stats': final_stats,
                'step_count': step_count,
            }

            save_path = os.path.join(args.save_dir, f'dl_pac_v2_n{args.n_agents}_k{args.k_hops}_s{seed}.pt')
            torch.save({
                'result': result,
                'args': vars(args),
                'seed': seed,
            }, save_path)
            print(f"Saved: {save_path}")

            all_results['dl_pac_v2']['seeds'].append(result)

    # === Summary ===
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for alg_name in ['safe_maddpg', 'lagrangian_ppo', 'dl_pac_v2']:
        results_list = all_results[alg_name]['seeds']
        if not results_list:
            continue

        # Use eval rewards at comparable points
        if results_list[0]['eval_rewards']:
            eval_rewards = np.array([r['eval_rewards'] for r in results_list])
            eval_violations = np.array([r['eval_violations'] for r in results_list])

            mean_reward = np.mean(eval_rewards, axis=0)
            std_reward = np.std(eval_rewards, axis=0)
            mean_viol = np.mean(eval_violations, axis=0)
            std_viol = np.std(eval_violations, axis=0)

            final_idx = -1
            print(f"\n{alg_name.upper()}:")
            print(f"  Final Eval Reward: {mean_reward[final_idx]:.2f} +/- {std_reward[final_idx]:.2f}")
            print(f"  Final Violation Rate: {mean_viol[final_idx]:.2%} +/- {std_viol[final_idx]:.2%}")
            print(f"  Best Eval Reward: {np.max(mean_reward):.2f} (ep {np.argmax(mean_reward) * args.eval_interval})")
            print(f"  Best Violation Rate: {np.min(mean_viol):.2%} (ep {np.argmin(mean_viol) * args.eval_interval})")
        else:
            # Use episode rewards
            ep_rewards = np.array([r['episode_rewards'] for r in results_list])
            mean_last100 = np.mean(ep_rewards[:, -100:], axis=1)
            mean_viol = np.array([r['final_stats']['violation_rate'] for r in results_list])
            print(f"\n{alg_name.upper()}:")
            print(f"  Final Train Reward (last 100): {np.mean(mean_last100):.2f} +/- {np.std(mean_last100):.2f}")
            print(f"  Final Violation Rate: {np.mean(mean_viol):.2%} +/- {np.std(mean_viol):.2%}")

    # Save summary
    summary_path = os.path.join(args.save_dir, f'summary_n{args.n_agents}_k{args.k_hops}.pt')
    torch.save({
        'all_results': all_results,
        'args': vars(args),
        'seeds': seeds,
        'timestamp': str(datetime.now()),
    }, summary_path)
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    args = parse_args()
    print(f"Device: {args.device}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    run_all_baselines(args)
