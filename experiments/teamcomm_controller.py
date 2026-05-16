"""
Minimal TeamComm integration for the warehouse benchmark.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import sys
import types
from typing import Dict, List, Tuple

import numpy as np
import torch

from teamcomm_warehouse_wrapper import TeamCommWarehouseWrapper


TEAMCOMM_REPO_ROOT = os.environ.get("TEAMCOMM_REPO_ROOT", "")


def _ensure_package(name: str, path: str):
    pkg = sys.modules.get(name)
    if pkg is None:
        pkg = types.ModuleType(name)
        pkg.__path__ = [path]
        sys.modules[name] = pkg
    return pkg


def _load_module(module_name: str, file_path: str):
    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load TeamComm module {module_name} from {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _import_teamcomm_components():
    if not TEAMCOMM_REPO_ROOT or not os.path.isdir(TEAMCOMM_REPO_ROOT):
        raise FileNotFoundError(
            "TeamComm repo not found. Set TEAMCOMM_REPO_ROOT to a local clone before using the TeamComm baseline."
        )

    if TEAMCOMM_REPO_ROOT not in sys.path:
        sys.path.insert(0, TEAMCOMM_REPO_ROOT)

    modules_dir = os.path.join(TEAMCOMM_REPO_ROOT, "modules")
    runner_dir = os.path.join(TEAMCOMM_REPO_ROOT, "runner")
    baselines_dir = os.path.join(TEAMCOMM_REPO_ROOT, "baselines")

    modules_pkg = _ensure_package("modules", modules_dir)
    runner_pkg = _ensure_package("runner", runner_dir)
    baselines_pkg = _ensure_package("baselines", baselines_dir)

    _load_module("modules.utils", os.path.join(modules_dir, "utils.py"))
    runner_core = _load_module("runner.runner", os.path.join(runner_dir, "runner.py"))
    setattr(runner_pkg, "Runner", runner_core.Runner)

    runner_teamcomm = _load_module(
        "runner.runner_teamcomm",
        os.path.join(runner_dir, "runner_teamcomm.py"),
    )
    baselines_teamcomm = _load_module(
        "baselines.teamcomm",
        os.path.join(baselines_dir, "teamcomm.py"),
    )
    setattr(baselines_pkg, "TeamCommAgent", baselines_teamcomm.TeamCommAgent)
    setattr(runner_pkg, "RunnerTeamComm", runner_teamcomm.RunnerTeamComm)
    return baselines_teamcomm.TeamCommAgent, runner_teamcomm.RunnerTeamComm


def build_teamcomm_config(args, env_info: Dict[str, int]) -> Dict:
    use_cuda = str(args.device).lower().startswith("cuda") and torch.cuda.is_available()
    return {
        "seed": args.seed,
        "env": "warehouse",
        "map": f"warehouse-n{args.n_agents}-s{args.n_shelves}",
        "n_agents": env_info["n_agents"],
        "nagents": env_info["n_agents"],
        "n_actions": env_info["n_actions"],
        "obs_shape": env_info["obs_shape"],
        "state_shape": env_info["state_shape"],
        "episode_length": env_info["episode_length"],
        "hid_size": 64,
        "att_head": 1,
        "block": "no",
        "interval": 1,
        "gamma": 0.9,
        "lr": 1e-3,
        "value_coeff": 0.01,
        "vib_coeff": 0.01,
        "modularity_coeff": 10.0,
        "batch_size": max(args.max_steps, 1),
        "normalize_rewards": True,
        "normalize_advantages": False,
        "use_cuda": use_cuda,
        "use_multiprocessing": False,
        "memo": "warehouse",
        "moduarity": False,
        "vib": True,
    }


class TeamCommController:
    def __init__(self, config: Dict, wrapper: TeamCommWarehouseWrapper, agent, runner):
        self.config = config
        self.wrapper = wrapper
        self.agent = agent
        self.runner = runner
        self.device = getattr(self.runner, "device", torch.device("cpu"))
        self.message_dim = int(getattr(self.agent.agent, "message_dim", 64))
        self.comm_stats = {
            "updates": 0,
            "actor_messages": 0,
            "actor_scalars": 0,
            "dual_messages": 0,
            "dual_scalars": 0,
            "neighborhood_entries": 0,
            "mean_neighborhood_size": 0.0,
            "comm_note": "TeamComm message counts are estimated from inferred inter/intra group edges.",
        }

    def _matrix_to_set(self, assignment: List[int]) -> List[List[int]]:
        sets = [[] for _ in range(self.config["n_agents"])]
        for agent_idx, group in enumerate(assignment):
            sets[int(group)].append(agent_idx)
        return [group for group in sets if group]

    def _choose_group_actions(self, team_action_out: torch.Tensor, deterministic: bool) -> List[int]:
        if deterministic:
            return torch.argmax(team_action_out, dim=-1).detach().cpu().tolist()
        sampled = self.runner.choose_action(team_action_out)
        if len(sampled) != 1:
            raise RuntimeError(f"Unexpected TeamComm group action format: {type(sampled)}")
        return np.asarray(sampled[0]).reshape(-1).tolist()

    def _estimate_comm(self, sets: List[List[int]]):
        intra_edges = sum(len(group) * max(len(group) - 1, 0) for group in sets)
        inter_groups = len(sets)
        inter_edges = inter_groups * max(inter_groups - 1, 0)
        total_edges = intra_edges + inter_edges
        self.comm_stats["actor_messages"] += int(total_edges)
        self.comm_stats["actor_scalars"] += int(total_edges * self.message_dim)
        self.comm_stats["neighborhood_entries"] += int(intra_edges)

    def act(self, obs_list: List[np.ndarray], deterministic: bool = True) -> List[int]:
        obs_tensor = torch.tensor(np.asarray(obs_list), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            team_action_out = self.agent.teaming(obs_tensor)
            assignment = self._choose_group_actions(team_action_out, deterministic=deterministic)
            sets = self._matrix_to_set(assignment)
            after_comm, _, _ = self.agent.communicate(obs_tensor, sets)
            action_outs, _ = self.agent.agent(after_comm)

        self._estimate_comm(sets)
        if deterministic:
            actions = torch.argmax(action_outs, dim=-1).detach().cpu().tolist()
        else:
            sampled = self.runner.choose_action(action_outs)
            actions = np.asarray(sampled[0]).reshape(-1).tolist()
        return [int(a) for a in actions]

    def train_epoch(self, batch_size: int) -> Dict:
        log = self.runner.train_batch(batch_size=batch_size)
        self.comm_stats["updates"] += 1
        return log

    def get_communication_stats(self):
        updates = max(int(self.comm_stats["updates"]), 1)
        neighborhood_entries = int(self.comm_stats["neighborhood_entries"])
        self.comm_stats["mean_neighborhood_size"] = neighborhood_entries / max(
            self.config["n_agents"] * updates, 1
        )
        return dict(self.comm_stats)

    def save_checkpoint(self, path: str):
        torch.save(
            {
                "config": self.config,
                "agent_state_dict": self.agent.state_dict(),
                "comm_stats": self.comm_stats,
            },
            path,
        )


def build_teamcomm_controller(args, env):
    teamcomm_agent_cls, teamcomm_runner_cls = _import_teamcomm_components()
    wrapper = TeamCommWarehouseWrapper(env=env, seed=args.seed)
    env_info = wrapper.get_env_info()
    config = build_teamcomm_config(args, env_info)
    agent = teamcomm_agent_cls(config)
    runner = teamcomm_runner_cls(config, wrapper, agent)
    controller = TeamCommController(config=config, wrapper=wrapper, agent=agent, runner=runner)
    return controller, None, {"teamcomm_repo_root": TEAMCOMM_REPO_ROOT, "runner": "RunnerTeamComm"}


def train_teamcomm(args, controller: TeamCommController, env, evaluate_fn):
    episode_rewards = []
    total_steps = 0
    total_episodes = 0
    eval_history = []

    while total_episodes < args.n_episodes:
        train_log = controller.train_epoch(batch_size=max(args.max_steps, 1))
        batch_episodes = int(train_log.get("num_episodes", 1))
        batch_steps = int(train_log.get("num_steps", args.max_steps))
        total_steps += batch_steps
        total_episodes += batch_episodes

        mean_batch_reward = float(train_log.get("episode_return", 0.0)) / max(batch_episodes, 1)
        episode_rewards.extend([mean_batch_reward] * batch_episodes)

        if total_episodes % args.eval_interval == 0:
            eval_stats, _ = evaluate_fn(
                algorithm="teamcomm",
                controller=controller,
                env=env,
                n_episodes=min(20, args.eval_episodes),
                seed_base=800000 + args.seed * 1000 + total_episodes * 20,
                deadlock_window=args.deadlock_window,
                deadlock_motion_eps=args.deadlock_motion_eps,
                record_failure_trace=False,
            )
            eval_stats["episode"] = total_episodes
            eval_history.append(eval_stats)

        if total_episodes % args.log_interval == 0:
            print(f"{total_episodes:>6} | reward={np.mean(episode_rewards[-args.log_interval:]):>8.2f}")

    return {
        "episode_rewards": episode_rewards[: args.n_episodes],
        "eval_history": eval_history,
        "total_steps": total_steps,
        "comm_stats": controller.get_communication_stats(),
    }
