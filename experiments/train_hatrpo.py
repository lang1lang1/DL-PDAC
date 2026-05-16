"""
HATRPO Baseline Training Script
Runs HATRPO on Multi-Agent Particle Environment for n=4, 8, 20
Supports multiple seeds for statistical validation
"""

import torch
import numpy as np
import argparse
import os
import sys
from datetime import datetime
from typing import Dict, List

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from baselines import HATRPOSystem
from envs import MultiAgentParticleEnv, ConstraintTracker


def parse_args():
    parser = argparse.ArgumentParser(description='HATRPO Baseline Experiment')
    parser.add_argument('--n_agents', type=int, default=4, help='Number of agents')
    parser.add_argument('--n_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--max_steps', type=int, default=100, help='Max steps per episode')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda, mps, cpu')
    parser.add_argument('--seed', type=int, default=0, help='Random seed')
    parser.add_argument('--log_interval', type=int, default=50, help='Log interval')
    parser.add_argument('--update_interval', type=int, default=4, help='Update every N steps')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dim for networks')
    parser.add_argument('--delta', type=float, default=0.01, help='Trust region KL threshold')
    parser.add_argument('--use_safety_filter', type=int, default=1, help='Use safety filter (1=yes, 0=no)')
    parser.add_argument('--collision_threshold', type=float, default=0.5, help='Collision threshold')
    parser.add_argument('--safety_alpha', type=float, default=3.0, help='Safety filter alpha')
    parser.add_argument('--save_dir', type=str, default='./logs/baselines', help='Save directory')
    parser.add_argument('--tag', type=str, default='hatrpo', help='Experiment tag')
    return parser.parse_args()


def run_hatrpo(args):
    """Run HATRPO experiment."""
    print("=" * 60)
    print(f"HATRPO Baseline Experiment")
    print(f"n_agents={args.n_agents}, episodes={args.n_episodes}, seed={args.seed}")
    print(f"device={args.device}, delta={args.delta}, safety_filter={args.use_safety_filter}")
    print("=" * 60)

    # Set seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create environment
    env = MultiAgentParticleEnv(
        n_agents=args.n_agents,
        n_landmarks=2,
        max_steps=args.max_steps,
    )

    obs_dim = env.obs_dim
    action_dim = env.action_dim
    obs_dims = [obs_dim] * args.n_agents
    action_dims = [action_dim] * args.n_agents

    # Create HATRPO system
    system = HATRPOSystem(
        n_agents=args.n_agents,
        obs_dims=obs_dims,
        action_dims=action_dims,
        device=args.device,
        delta=args.delta,
        hidden_dim=args.hidden_dim,
        use_safety_filter=(args.use_safety_filter == 1),
        collision_threshold=args.collision_threshold,
        safety_alpha=args.safety_alpha,
    )

    # Constraint tracker
    constraint_tracker = ConstraintTracker()

    # Training
    episode_rewards = []
    episode_lengths = []
    step_count = 0

    print(f"\n{'Episode':>8} {'Reward':>10} {'ViolRate':>10} {'MeanViol':>10}")
    print("-" * 50)

    for episode in range(args.n_episodes):
        obs, _ = env.reset(seed=args.seed + episode)
        constraint_tracker.reset_episode()
        episode_reward = 0
        episode_steps = 0

        obs_list = [obs[agent] for agent in sorted(obs.keys())]

        for step in range(args.max_steps):
            # Get positions for safety filter
            positions = env.agent_positions if hasattr(env, 'agent_positions') else None

            # Get actions
            actions = system.act(obs_list, positions, deterministic=False)
            action_dict = {
                agent: actions[i]
                for i, agent in enumerate(sorted(obs.keys()))
            }

            # Environment step
            next_obs, rewards, terminations, truncations, infos = env.step(action_dict)

            # Get values for buffer
            with torch.no_grad():
                for i, (agent_id, agent_obs) in enumerate(zip(sorted(obs.keys()), obs_list)):
                    obs_t = torch.FloatTensor(agent_obs).to(args.device).unsqueeze(0)
                    _, log_prob, value = system.agents[i].actor_old.act(obs_t)
                    constraint_val = infos[agent_id].get('constraint', 0) if agent_id in infos else 0
                    system.store(i, agent_obs, actions[i], 0, log_prob.sum().item(), value.sum().item(), False)

            # Track constraints
            constraint_dict = {agent: infos[agent].get('constraint', 0) for agent in infos.keys()}
            constraint_tracker.add(constraint_dict)

            obs_list = [next_obs[agent] for agent in sorted(next_obs.keys())]
            episode_reward += sum([rewards[agent] for agent in sorted(rewards.keys())])
            episode_steps += 1
            step_count += 1

            # Update
            if (step + 1) % args.update_interval == 0:
                system.update_all()

        # Episode stats
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_steps)
        constraint_stats = constraint_tracker.get_stats()

        # Logging
        if (episode + 1) % args.log_interval == 0:
            mean_reward = np.mean(episode_rewards[-args.log_interval:])
            print(f"{episode+1:>8} {mean_reward:>10.2f} "
                  f"{constraint_stats['violation_rate']:>10.2%} "
                  f"{constraint_stats['episode_mean_violation']:>10.4f}")

    # Final results
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final Mean Reward: {np.mean(episode_rewards[-100:]):.2f}")
    print(f"Final Violation Rate: {constraint_tracker.get_stats()['violation_rate']:.2%}")
    print(f"Total Steps: {step_count}")

    # Save results
    sf_tag = "sf" if args.use_safety_filter else "nosf"
    save_dir = os.path.join(
        args.save_dir,
        f"{args.tag}_n{args.n_agents}_{sf_tag}_s{args.seed}"
    )
    os.makedirs(save_dir, exist_ok=True)

    final_stats = constraint_tracker.get_stats()

    results = {
        'result': {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'final_stats': final_stats,
        },
        'args': vars(args),
        'seed': args.seed,
    }

    torch.save(results, os.path.join(save_dir, f'{args.tag}_n{args.n_agents}_s{args.seed}.pt'))
    print(f"\nResults saved to: {save_dir}")

    return results


def run_sweep():
    """Run HATRPO sweep for multiple configurations."""
    import itertools

    configs = {
        'n_agents': [4, 8, 20],
        'n_episodes': [1000, 2000, 1500],
        'seeds': [0, 1, 2],
        'use_safety_filter': [1],  # With safety filter
    }

    results = {}
    total_runs = (3 + 3 + 3) * 3  # 9 configurations

    print(f"Running {total_runs} HATRPO experiments...")

    run_id = 0
    for n_agents in configs['n_agents']:
        n_eps = {4: 1000, 8: 2000, 20: 1500}[n_agents]

        for seed in configs['seeds']:
            run_id += 1
            print(f"\n[{run_id}/{total_runs}] HATRPO n={n_agents}, seed={seed}")

            args = parse_args()
            args.n_agents = n_agents
            args.n_episodes = n_eps
            args.seed = seed
            args.use_safety_filter = 1
            args.device = 'cuda'
            args.log_interval = 100
            args.delta = 0.01

            try:
                result = run_hatrpo(args)
                results[f"n{n_agents}_s{seed}"] = result['result']['final_stats']
            except Exception as e:
                print(f"Error: {e}")
                results[f"n{n_agents}_s{seed}"] = {'error': str(e)}

    # Summary
    print("\n" + "=" * 60)
    print("HATRPO SWEEP SUMMARY")
    print("=" * 60)
    for key, stats in results.items():
        if 'error' not in stats:
            print(f"{key:>15s}: ViolRate={stats['violation_rate']:.2%}, Reward={np.mean(stats.get('episode_rewards', [0])[-100:]):.1f}")
        else:
            print(f"{key:>15s}: ERROR")

    return results


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == '--sweep':
        run_sweep()
    else:
        args = parse_args()
        run_hatrpo(args)
