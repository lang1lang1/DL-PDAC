"""Scal-MAPPO-L style baseline for the paper navigation benchmark.

This is a PyTorch adaptation of the practical Scal-MAPPO-L idea for the
project's discrete MultiAgentParticleEnv.  It intentionally stays close to the
paper-facing comparison need:

- local k-hop observations, implemented by masking non-neighbor feature blocks;
- MAPPO-Lagrangian style reward/cost actor-critic objective;
- per-agent Lagrange multipliers;
- sequential per-agent PPO updates over a shared homogeneous policy.

It is not a drop-in reproduction of the NeurIPS 2024 Safe-MAMuJoCo codebase.
The implementation is scoped to the current repository's discrete navigation
environment so it can be compared against DL-PDAC under the same metrics.
"""

from __future__ import annotations

import copy
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _as_tensor(array, device: str, dtype=torch.float32) -> torch.Tensor:
    return torch.as_tensor(array, dtype=dtype, device=device)


class LocalActorCritic(nn.Module):
    """Shared local policy with separate reward and cost value heads."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor = nn.Linear(hidden_dim, action_dim)
        self.reward_value = nn.Linear(hidden_dim, 1)
        self.cost_value = nn.Linear(hidden_dim, 1)

    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.encoder(obs)
        return self.actor(h), self.reward_value(h).squeeze(-1), self.cost_value(h).squeeze(-1)


class ScalMAPPOLSystem:
    """Scalable MAPPO-Lagrangian style baseline.

    The public API matches `revision_metrics.evaluate_navigation`: `act` returns
    per-agent action probability vectors, and deterministic evaluation uses the
    argmax externally.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        adjacency_matrix: np.ndarray,
        k_hops: int = 1,
        device: str = "cuda",
        hidden_dim: int = 128,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_eps: float = 0.2,
        ppo_epochs: int = 4,
        minibatch_size: int = 256,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        cost_value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        dual_lr: float = 0.01,
        dual_max: float = 20.0,
        cost_limit: float = 0.05,
        n_landmarks: int = 2,
    ):
        self.n_agents = int(n_agents)
        self.obs_dim = int(obs_dim)
        self.action_dim = int(action_dim)
        self.adjacency = np.asarray(adjacency_matrix, dtype=np.float32)
        self.k_hops = int(k_hops)
        self.device = device
        self.gamma = float(gamma)
        self.gae_lambda = float(gae_lambda)
        self.clip_eps = float(clip_eps)
        self.ppo_epochs = int(ppo_epochs)
        self.minibatch_size = int(minibatch_size)
        self.entropy_coef = float(entropy_coef)
        self.value_coef = float(value_coef)
        self.cost_value_coef = float(cost_value_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.dual_lr = float(dual_lr)
        self.dual_max = float(dual_max)
        self.cost_limit = float(cost_limit)
        self.n_landmarks = int(n_landmarks)

        self.policy = LocalActorCritic(obs_dim, action_dim, hidden_dim=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.duals = torch.ones(self.n_agents, dtype=torch.float32, device=device)
        self.neighborhoods = self._compute_k_hop_neighborhoods(self.k_hops)

        self.buffer: List[Dict[str, list]] = [
            {
                "obs": [],
                "actions": [],
                "log_probs": [],
                "reward_values": [],
                "cost_values": [],
                "rewards": [],
                "costs": [],
                "dones": [],
            }
            for _ in range(self.n_agents)
        ]

        self.comm_stats = {
            "updates": 0,
            "rollout_steps": 0,
            "neighbor_messages": 0,
            "neighbor_scalars": 0,
            "mean_neighborhood_size_sum": 0.0,
        }

    def _compute_k_hop_neighborhoods(self, k: int) -> List[List[int]]:
        neighborhoods = []
        for i in range(self.n_agents):
            visited = {i}
            frontier = {i}
            for _ in range(max(k, 0)):
                nxt = set()
                for j in frontier:
                    nxt.update(np.flatnonzero(self.adjacency[j] > 0.0).astype(int).tolist())
                nxt -= visited
                visited.update(nxt)
                frontier = nxt
            neighborhoods.append(sorted(int(x) for x in visited))
        return neighborhoods

    def _mask_obs(self, agent_idx: int, obs: np.ndarray) -> np.ndarray:
        """Zero neighbor feature blocks outside the k-hop set.

        MultiAgentParticleEnv observations are:
          self(4) + all other agents in fixed order ((n-1) * 4) + landmarks.
        """
        masked = np.asarray(obs, dtype=np.float32).copy()
        neighbor_start = 4
        block = 4
        keep = set(self.neighborhoods[agent_idx])
        slot = 0
        for other in range(self.n_agents):
            if other == agent_idx:
                continue
            start = neighbor_start + slot * block
            end = start + block
            if other not in keep:
                masked[start:end] = 0.0
            slot += 1
        return masked

    def mask_obs_list(self, obs_list: List[np.ndarray]) -> np.ndarray:
        return np.stack([self._mask_obs(i, obs) for i, obs in enumerate(obs_list)], axis=0)

    def _record_rollout_comm(self):
        peer_links = sum(max(len(nbrs) - 1, 0) for nbrs in self.neighborhoods)
        self.comm_stats["rollout_steps"] += 1
        self.comm_stats["neighbor_messages"] += int(peer_links)
        self.comm_stats["neighbor_scalars"] += int(peer_links * self.obs_dim)
        self.comm_stats["mean_neighborhood_size_sum"] += float(
            np.mean([len(nbrs) for nbrs in self.neighborhoods])
        )

    def act(self, obs_list: List[np.ndarray], deterministic: bool = False) -> List[np.ndarray]:
        obs = _as_tensor(self.mask_obs_list(obs_list), self.device)
        with torch.no_grad():
            logits, _, _ = self.policy(obs)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                actions = torch.argmax(probs, dim=-1)
                one_hot = F.one_hot(actions, num_classes=self.action_dim).float()
                return one_hot.cpu().numpy()
            return probs.cpu().numpy()

    def sample_actions(self, obs_list: List[np.ndarray]) -> Dict[str, np.ndarray]:
        self._record_rollout_comm()
        masked_obs = self.mask_obs_list(obs_list)
        obs_t = _as_tensor(masked_obs, self.device)
        with torch.no_grad():
            logits, reward_values, cost_values = self.policy(obs_t)
            dist = torch.distributions.Categorical(logits=logits)
            actions = dist.sample()
            log_probs = dist.log_prob(actions)
            probs = F.softmax(logits, dim=-1)
        return {
            "masked_obs": masked_obs,
            "actions": actions.cpu().numpy().astype(np.int64),
            "log_probs": log_probs.cpu().numpy().astype(np.float32),
            "reward_values": reward_values.cpu().numpy().astype(np.float32),
            "cost_values": cost_values.cpu().numpy().astype(np.float32),
            "probs": probs.cpu().numpy().astype(np.float32),
        }

    def store(
        self,
        masked_obs: np.ndarray,
        actions: np.ndarray,
        log_probs: np.ndarray,
        reward_values: np.ndarray,
        cost_values: np.ndarray,
        rewards: List[float],
        costs: List[float],
        dones: List[bool],
    ):
        for i in range(self.n_agents):
            b = self.buffer[i]
            b["obs"].append(masked_obs[i])
            b["actions"].append(int(actions[i]))
            b["log_probs"].append(float(log_probs[i]))
            b["reward_values"].append(float(reward_values[i]))
            b["cost_values"].append(float(cost_values[i]))
            b["rewards"].append(float(rewards[i]))
            b["costs"].append(float(costs[i]))
            b["dones"].append(float(dones[i]))

    def _clear_buffer(self):
        for b in self.buffer:
            for values in b.values():
                values.clear()

    def _gae(self, rewards: torch.Tensor, values: torch.Tensor, dones: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        advantages = torch.zeros_like(rewards)
        last_gae = torch.tensor(0.0, device=self.device)
        next_value = torch.tensor(0.0, device=self.device)
        for t in reversed(range(rewards.numel())):
            nonterminal = 1.0 - dones[t]
            delta = rewards[t] + self.gamma * next_value * nonterminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * nonterminal * last_gae
            advantages[t] = last_gae
            next_value = values[t]
        returns = advantages + values
        return advantages, returns

    def update(self, min_rollout_steps: int = 128) -> Dict[str, float]:
        if any(len(b["obs"]) < min_rollout_steps for b in self.buffer):
            return {}

        metrics = []
        for agent_id in range(self.n_agents):
            b = self.buffer[agent_id]
            obs = _as_tensor(np.asarray(b["obs"], dtype=np.float32), self.device)
            actions = _as_tensor(np.asarray(b["actions"], dtype=np.int64), self.device, dtype=torch.long)
            old_log_probs = _as_tensor(np.asarray(b["log_probs"], dtype=np.float32), self.device)
            reward_values_old = _as_tensor(np.asarray(b["reward_values"], dtype=np.float32), self.device)
            cost_values_old = _as_tensor(np.asarray(b["cost_values"], dtype=np.float32), self.device)
            rewards = _as_tensor(np.asarray(b["rewards"], dtype=np.float32), self.device)
            costs = _as_tensor(np.asarray(b["costs"], dtype=np.float32), self.device)
            dones = _as_tensor(np.asarray(b["dones"], dtype=np.float32), self.device)

            with torch.no_grad():
                reward_adv, reward_returns = self._gae(rewards, reward_values_old, dones)
                cost_adv, cost_returns = self._gae(costs, cost_values_old, dones)
                lag_adv = reward_adv - self.duals[agent_id] * cost_adv
                lag_adv = (lag_adv - lag_adv.mean()) / (lag_adv.std(unbiased=False) + 1e-8)

            n = obs.shape[0]
            last_policy_loss = torch.tensor(0.0, device=self.device)
            last_value_loss = torch.tensor(0.0, device=self.device)
            last_entropy = torch.tensor(0.0, device=self.device)

            for _ in range(self.ppo_epochs):
                perm = torch.randperm(n, device=self.device)
                for start in range(0, n, self.minibatch_size):
                    idx = perm[start:start + self.minibatch_size]
                    logits, reward_value, cost_value = self.policy(obs[idx])
                    dist = torch.distributions.Categorical(logits=logits)
                    new_log_probs = dist.log_prob(actions[idx])
                    entropy = dist.entropy().mean()

                    ratio = torch.exp(new_log_probs - old_log_probs[idx])
                    surr1 = ratio * lag_adv[idx]
                    surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * lag_adv[idx]
                    policy_loss = -torch.min(surr1, surr2).mean()
                    reward_value_loss = F.mse_loss(reward_value, reward_returns[idx])
                    cost_value_loss = F.mse_loss(cost_value, cost_returns[idx])
                    value_loss = self.value_coef * reward_value_loss + self.cost_value_coef * cost_value_loss
                    loss = policy_loss + value_loss - self.entropy_coef * entropy

                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                    self.optimizer.step()

                    last_policy_loss = policy_loss.detach()
                    last_value_loss = value_loss.detach()
                    last_entropy = entropy.detach()

            with torch.no_grad():
                dual_update = self.dual_lr * (costs.mean() - self.cost_limit)
                self.duals[agent_id] = torch.clamp(self.duals[agent_id] + dual_update, 0.0, self.dual_max)

            metrics.append(
                {
                    "policy_loss": float(last_policy_loss.item()),
                    "value_loss": float(last_value_loss.item()),
                    "entropy": float(last_entropy.item()),
                    "mean_cost": float(costs.mean().item()),
                    "dual": float(self.duals[agent_id].item()),
                }
            )

        self._clear_buffer()
        self.comm_stats["updates"] += 1
        return {
            "policy_loss": float(np.mean([m["policy_loss"] for m in metrics])),
            "value_loss": float(np.mean([m["value_loss"] for m in metrics])),
            "entropy": float(np.mean([m["entropy"] for m in metrics])),
            "mean_cost": float(np.mean([m["mean_cost"] for m in metrics])),
            "mean_dual": float(np.mean([m["dual"] for m in metrics])),
        }

    def get_communication_stats(self) -> Dict[str, float]:
        rollout_steps = max(int(self.comm_stats["rollout_steps"]), 1)
        updates = max(int(self.comm_stats["updates"]), 1)
        return {
            **self.comm_stats,
            "neighbor_messages_per_step": self.comm_stats["neighbor_messages"] / rollout_steps,
            "neighbor_scalars_per_step": self.comm_stats["neighbor_scalars"] / rollout_steps,
            "mean_neighborhood_size": self.comm_stats["mean_neighborhood_size_sum"] / rollout_steps,
            "updates": self.comm_stats["updates"],
            "updates_safe_denominator": updates,
        }

    def get_checkpoint_payload(self) -> Dict:
        return {
            "policy_state_dict": copy.deepcopy(self.policy.state_dict()),
            "duals": self.duals.detach().cpu().numpy(),
            "neighborhoods": self.neighborhoods,
            "comm_stats": self.get_communication_stats(),
        }

    def load_checkpoint_payload(self, payload: Dict):
        self.policy.load_state_dict(payload["policy_state_dict"])
        self.duals = torch.as_tensor(
            payload["duals"],
            dtype=torch.float32,
            device=self.device,
        )
