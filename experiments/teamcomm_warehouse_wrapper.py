"""
Warehouse wrapper that matches the minimal API expected by TeamComm's runner.
"""

from __future__ import annotations

import numpy as np


class TeamCommWarehouseWrapper:
    """
    Adapts MultiRobotWarehouseEnv to the TeamComm wrapper protocol:

    - reset() -> (obs, state)
    - get_obs() -> list[np.ndarray]
    - step(actions) -> (rewards, done, info)
    - get_env_info() -> dict with state/obs/action sizes
    """

    def __init__(self, env, seed: int = 0):
        self._env = env
        self.n_agents = env.n_agents
        self.episode_limit = env.max_steps
        self._seed = seed
        self._obs = None
        self._last_info = {}

    def _ensure_initialized(self):
        if not hasattr(self._env, "robot_positions"):
            obs, _ = self._env.reset(seed=self._seed)
            self._obs = [obs[agent].astype(np.float32) for agent in self._sorted_agents()]
            self._last_info = {}

    def _sorted_agents(self):
        return sorted(self._env.possible_agents)

    def _build_state(self) -> np.ndarray:
        self._ensure_initialized()
        robot_pos = np.asarray(self._env.robot_positions, dtype=np.float32).reshape(-1)
        robot_vel = np.asarray(self._env.robot_velocities, dtype=np.float32).reshape(-1)
        shelves = np.asarray(self._env.shelves, dtype=np.float32).reshape(-1)
        picked = np.asarray(self._env.shelf_picked, dtype=np.float32).reshape(-1)
        return np.concatenate([robot_pos, robot_vel, shelves, picked], axis=0).astype(np.float32)

    def reset(self, seed: int | None = None):
        obs, _ = self._env.reset(seed=self._seed if seed is None else seed)
        self._obs = [obs[agent].astype(np.float32) for agent in self._sorted_agents()]
        self._last_info = {}
        return self.get_obs(), self.get_state()

    def step(self, actions):
        if isinstance(actions, np.ndarray):
            actions = actions.tolist()
        if len(actions) == 1 and isinstance(actions[0], np.ndarray):
            actions = actions[0].tolist()
        action_dict = {
            agent: int(actions[i])
            for i, agent in enumerate(self._sorted_agents())
        }
        next_obs, rewards, terminations, truncations, infos = self._env.step(action_dict)
        self._obs = [next_obs[agent].astype(np.float32) for agent in self._sorted_agents()]
        done = bool(
            any(terminations[agent] or truncations[agent] for agent in self._sorted_agents())
        )
        reward_vec = np.asarray(
            [float(rewards[agent]) for agent in self._sorted_agents()],
            dtype=np.float32,
        )
        collisions = int(
            any(float(infos[agent].get("constraint", 0.0)) > 0.0 for agent in self._sorted_agents())
        )
        completed_agent = [
            bool(terminations[agent] or truncations[agent])
            for agent in self._sorted_agents()
        ]
        self._last_info = {
            "completed_agent": completed_agent,
            "num_collisions": collisions,
            "items_picked": max(int(infos[agent].get("items_picked", 0)) for agent in self._sorted_agents()),
        }
        return reward_vec, done, self._last_info

    def get_obs(self):
        self._ensure_initialized()
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return int(self._env.obs_dim)

    def get_state(self):
        return self._build_state()

    def get_state_size(self):
        return int(self._build_state().shape[0])

    def get_avail_actions(self):
        return [self.get_avail_agent_actions(i) for i in range(self.n_agents)]

    def get_avail_agent_actions(self, agent_id):
        return [1] * int(self._env.action_dim)

    def get_total_actions(self):
        return int(self._env.action_dim)

    def close(self):
        if hasattr(self._env, "close"):
            self._env.close()

    def seed(self):
        return self._seed

    def get_graph(self):
        return None

    def get_env_info(self):
        return {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_length": self.episode_limit,
        }
