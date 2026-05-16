"""
Baseline Algorithms for DL-PDAC Comparison
Includes: Lagrangian PPO, Safe MADDPG (MADDPG + CBF Safety Layer)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


class PPOPolicy(nn.Module):
    """Actor-Critic policy for PPO."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.actor_head = nn.Linear(hidden_dim, action_dim)

        self.critic = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.actor(x)
        logits = self.actor_head(h)
        value = self.critic(x)
        return logits, value

    def act(self, x, deterministic=False):
        with torch.no_grad():
            logits, value = self.forward(x)
            probs = F.softmax(logits, dim=-1)
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
            return action, probs, value


class LagrangianPPOAgent:
    """Single-agent Lagrangian PPO with constraint satisfaction."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        device: str = "mps",
        lr: float = 3e-4,
        gamma: float = 0.99,
        eps: float = 0.2,
        dual_lr: float = 0.01,
        dual_max: float = 10.0,
        ent_coef: float = 0.01,
    ):
        self.device = device
        self.gamma = gamma
        self.eps = eps
        self.dual_lr = dual_lr
        self.dual_max = dual_max
        self.ent_coef = ent_coef

        self.policy = PPOPolicy(obs_dim, action_dim).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=lr)

        # Dual variable for constraint
        self.dual = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        self.buffer = {
            'obs': [], 'actions': [], 'rewards': [],
            'constraints': [], 'old_log_probs': [], 'values': [], 'dones': []
        }

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            action, probs, value = self.policy.act(obs_t, deterministic)
            return action.cpu().numpy()[0], probs.cpu().numpy()[0], value.item()

    def store(self, obs, action, reward, constraint, old_log_prob, value, done):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['constraints'].append(constraint)
        self.buffer['old_log_probs'].append(old_log_prob)
        self.buffer['values'].append(value)
        self.buffer['dones'].append(done)

    def _clear(self):
        for k in self.buffer:
            self.buffer[k].clear()

    def update(self, batch_size: int = 64) -> Dict[str, float]:
        if len(self.buffer['obs']) < batch_size:
            return {}

        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.LongTensor(np.array(self.buffer['actions'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device)
        constraints = torch.FloatTensor(np.array(self.buffer['constraints'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['old_log_probs'])).to(self.device)
        values = torch.FloatTensor(np.array(self.buffer['values'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)

        self._clear()

        # PPO update with Lagrangian
        with torch.no_grad():
            returns = torch.zeros_like(rewards)
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] - self.dual.item() * constraints[t] - values[t]
                gae = gae * self.gamma + delta
                returns[t] = gae + values[t]
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        for _ in range(4):  # PPO epochs
            indices = torch.randperm(len(obs))[:batch_size]
            batch_obs = obs[indices]
            batch_actions = actions[indices]
            batch_returns = returns[indices]
            batch_constraints = constraints[indices]
            batch_old_log = old_log_probs[indices]

            logits, values_pred = self.policy(batch_obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-8)
            selected_log_prob = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(selected_log_prob - batch_old_log)
            surr1 = ratio * batch_returns
            surr2 = torch.clamp(ratio, 1 - self.eps, 1 + self.eps) * batch_returns
            policy_loss = -torch.min(surr1, surr2).mean()

            entropy = -(probs * log_probs).sum(dim=1).mean()
            value_loss = F.mse_loss(values_pred.squeeze(-1), batch_returns)

            constrained_reward = rewards - self.dual.item() * constraints
            advantages = (constrained_reward - values.detach()).detach()
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            actor_loss = -(selected_log_prob * advantages).mean()

            total_loss = policy_loss + 0.5 * value_loss - self.ent_coef * entropy

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

        # Update dual
        mean_con = constraints.mean().item()
        new_dual = self.dual.item() + self.dual_lr * mean_con
        new_dual = float(np.clip(new_dual, 0.0, self.dual_max))
        self.dual.data = torch.tensor(new_dual, device=self.device)

        return {
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'entropy': entropy.item(),
            'dual': self.dual.item(),
            'mean_constraint': mean_con,
        }


class MAPOSystem:
    """Multi-Agent PPO with shared policy (centralized training)."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        action_dim: int,
        device: str = "mps",
        **kwargs,
    ):
        self.n_agents = n_agents
        self.device = device

        # Shared policy for all agents (MADDPG-style centralized training)
        self.policy = PPOPolicy(obs_dim, action_dim).to(device)
        self.opt = torch.optim.Adam(self.policy.parameters(), lr=3e-4)

        # One dual per agent
        self.duals = [nn.Parameter(torch.tensor(1.0), requires_grad=False) for _ in range(n_agents)]
        self.dual_lrs = [0.01] * n_agents

        self.buffer = {i: {
            'obs': [], 'actions': [], 'rewards': [], 'constraints': [],
            'old_log_probs': [], 'values': []
        } for i in range(n_agents)}

    def act(self, obs_list: List[np.ndarray], deterministic: bool = False):
        results = []
        for obs in obs_list:
            obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            with torch.no_grad():
                logits, value = self.policy.forward(obs_t)
                probs = F.softmax(logits, dim=-1)
                if deterministic:
                    action = torch.argmax(probs, dim=-1)
                else:
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
            results.append(action.cpu().numpy()[0])
        return results

    def store(self, agent_id: int, obs, action, reward, constraint, old_log_prob, value):
        b = self.buffer[agent_id]
        b['obs'].append(obs)
        b['actions'].append(action)
        b['rewards'].append(reward)
        b['constraints'].append(constraint)
        b['old_log_probs'].append(old_log_prob)
        b['values'].append(value)

    def update(self) -> List[Dict]:
        results = []
        for i in range(self.n_agents):
            b = self.buffer[i]
            if len(b['obs']) < 64:
                results.append({})
                continue

            obs = torch.FloatTensor(np.array(b['obs'])).to(self.device)
            actions = torch.LongTensor(np.array(b['actions'])).to(self.device)
            rewards = torch.FloatTensor(np.array(b['rewards'])).to(self.device)
            constraints = torch.FloatTensor(np.array(b['constraints'])).to(self.device)
            old_log_probs = torch.FloatTensor(np.array(b['old_log_probs'])).to(self.device)
            values = torch.FloatTensor(np.array(b['values'])).to(self.device)

            for k in b:
                b[k].clear()

            # GAE
            with torch.no_grad():
                returns = torch.zeros_like(rewards)
                gae = 0
                for t in reversed(range(len(rewards))):
                    delta = rewards[t] - self.duals[i].item() * constraints[t] - values[t]
                    gae = gae * 0.99 + delta
                    returns[t] = gae + values[t]
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # PPO update
            logits, values_pred = self.policy(obs)
            probs = F.softmax(logits, dim=-1)
            log_probs = torch.log(probs + 1e-8)
            selected_log_prob = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)

            ratio = torch.exp(selected_log_prob - old_log_probs)
            advantages = returns.detach()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 0.8, 1.2) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values_pred.squeeze(-1), returns)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            total_loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            self.opt.zero_grad()
            total_loss.backward()
            self.opt.step()

            # Update dual
            mean_con = constraints.mean().item()
            new_dual = self.duals[i].item() + 0.01 * mean_con
            new_dual = float(np.clip(new_dual, 0.0, 10.0))
            self.duals[i].data = torch.tensor(new_dual, device=self.device)

            results.append({
                'policy_loss': policy_loss.item(),
                'value_loss': value_loss.item(),
                'dual': self.duals[i].item(),
            })

        return results


# =============================================================================
# Safe MADDPG: MADDPG + Control Barrier Function Safety Layer
# =============================================================================


class SafeMADDPGCritic(nn.Module):
    """
    Centralized Q-network for MADDPG.
    Takes all agents' observations and actions as input.
    """

    def __init__(self, total_obs_dim: int, total_action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(total_obs_dim + total_action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        """all_obs: (batch, n_agents * obs_dim), all_actions: (batch, n_agents * action_dim)"""
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)


class SafeMADDPGActor(nn.Module):
    """
    Decentralized policy network for MADDPG.
    Each agent has its own actor that only sees local observation.
    Outputs Q-values for each action (for discrete actions).
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        # For discrete actions: output Q-values for each action
        self.q_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for each action."""
        h = self.net(obs)
        q_values = self.q_head(h)
        return q_values

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns action (int) and Q-values."""
        q_values = self.forward(obs)
        if deterministic:
            action = torch.argmax(q_values, dim=-1)
        else:
            probs = F.softmax(q_values, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
        return action, q_values


class CBFSafetyLayer:
    """
    Control Barrier Function (CBF) safety layer.

    Safety constraint: h(x) = d - d_min >= 0
    where d = collision_threshold, d_min = min distance to any other agent

    CBF condition: Delta h >= -alpha * h  (exponential safety)
    Equivalent to: safe actions must keep h(t+1) >= 0

    For discrete actions, we select the safe action with highest Q-value
    among those satisfying the CBF condition.
    """

    def __init__(
        self,
        agent_id: int,
        n_agents: int,
        collision_threshold: float = 0.5,
        alpha: float = 3.0,
        agent_size: float = 0.1,
        dt: float = 0.1,
    ):
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.collision_threshold = collision_threshold
        self.alpha = alpha  # CBF gain (higher = more aggressive safety)
        self.agent_size = agent_size
        self.dt = dt
        # Safety margin: keep extra buffer beyond collision threshold
        self.safety_margin = collision_threshold + agent_size * 2

    def _get_action_vectors(self) -> np.ndarray:
        """Get velocity vectors for each discrete action."""
        vectors = np.zeros((5, 2), dtype=np.float32)
        vectors[0] = [0, 1]    # up
        vectors[1] = [0, -1]   # down
        vectors[2] = [-1, 0]  # left
        vectors[3] = [1, 0]    # right
        vectors[4] = [0, 0]    # stay
        return vectors

    def compute_safety_score(self, current_pos: np.ndarray, other_positions: np.ndarray, action: int) -> float:
        """
        Compute safety score for an action.
        Returns: h(x') = min_dist_to_other - threshold
        Positive = safe, Negative = unsafe.
        """
        vectors = self._get_action_vectors()
        next_pos = current_pos + vectors[action] * self.dt

        min_dist = float('inf')
        for other_pos in other_positions:
            dist = np.linalg.norm(next_pos - other_pos)
            min_dist = min(min_dist, dist)

        safety_value = min_dist - self.safety_margin
        return safety_value

    def get_safe_action(
        self,
        current_pos: np.ndarray,
        other_positions: np.ndarray,
        policy_q_values: np.ndarray,
        fallback_to_safe: bool = True,
    ) -> int:
        """
        Select the safest action using CBF-filtered Q-values.

        CBF filtering: Only consider actions that satisfy h(x') >= -alpha * h(x)
        For simplicity, we select the highest-Q action among those with positive safety.
        """
        n_actions = len(policy_q_values)
        vectors = self._get_action_vectors()
        current_safety = self.compute_safety_score(current_pos, other_positions, 4)  # stay action as reference

        safe_actions = []
        for a in range(n_actions):
            h_next = self.compute_safety_score(current_pos, other_positions, a)
            # CBF condition: h(t+1) >= -alpha * h(t)
            # If current h is positive (safe), we need h_next >= -alpha * h_current
            # But simpler: just require h_next >= 0 (absolute safety)
            if h_next >= 0:
                safe_actions.append(a)

        if safe_actions:
            # Among safe actions, pick the one with highest Q-value
            safe_q = [(a, policy_q_values[a]) for a in safe_actions]
            return max(safe_q, key=lambda x: x[1])[0]

        if fallback_to_safe:
            # No safe action: use Q-learning style - pick argmax but trust env filter
            return int(np.argmax(policy_q_values))

        # Fallback: stay (action 4)
        return 4

    def project_to_safe_action(
        self,
        current_pos: np.ndarray,
        other_positions: np.ndarray,
        raw_action_probs: np.ndarray,
    ) -> np.ndarray:
        """
        Project action distribution onto safe set.
        Returns modified probs that put all mass on safest available actions.
        """
        n_actions = len(raw_action_probs)
        safe_mask = np.zeros(n_actions)
        for a in range(n_actions):
            h_next = self.compute_safety_score(current_pos, other_positions, a)
            if h_next >= 0:
                safe_mask[a] = 1.0

        if safe_mask.sum() > 0:
            safe_probs = raw_action_probs * safe_mask
            total = safe_probs.sum()
            if total > 0:
                return safe_probs / total
        # No safe action: return uniform over all (let env handle)
        return np.ones(n_actions) / n_actions


class SafeMADDPGAgent:
    """Single agent for Safe MADDPG."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id: int,
        n_agents: int,
        device: str = "mps",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        collision_threshold: float = 0.5,
        cbf_alpha: float = 3.0,
    ):
        self.device = device
        self.agent_id = agent_id
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.obs_dim = obs_dim  # Store per-agent obs dim
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.actor = SafeMADDPGActor(obs_dim, action_dim).to(device)
        self.critic = SafeMADDPGCritic(obs_dim * n_agents, action_dim * n_agents).to(device)
        self.target_critic = SafeMADDPGCritic(obs_dim * n_agents, action_dim * n_agents).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        # CBF Safety Layer
        self.cbf = CBFSafetyLayer(
            agent_id=agent_id,
            n_agents=n_agents,
            collision_threshold=collision_threshold,
            alpha=cbf_alpha,
        )

        # Replay buffer
        self.buffer = {
            'all_obs': [], 'all_actions': [], 'all_rewards': [],
            'all_next_obs': [], 'dones': []
        }

    def _pack(self, obs_list: List[np.ndarray], action_list: List[np.ndarray]) -> Tuple[torch.Tensor, torch.Tensor]:
        """Pack observations and actions from all agents."""
        obs_t = torch.FloatTensor(np.array(obs_list)).to(self.device)  # (n_agents, obs_dim)
        act_t = torch.FloatTensor(np.array(action_list)).to(self.device)  # (n_agents, act_dim)
        return obs_t, act_t

    def act_with_cbf(
        self,
        obs: np.ndarray,
        current_pos: np.ndarray,
        other_positions: List[np.ndarray],
        deterministic: bool = False,
    ) -> Tuple[int, np.ndarray]:
        """Act with CBF safety filtering."""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, q_values = self.actor.act(obs_t, deterministic)
            q_values_np = q_values.cpu().numpy()[0]

        # Apply CBF safety filter
        safe_action = self.cbf.get_safe_action(
            current_pos=current_pos,
            other_positions=other_positions,
            policy_q_values=q_values_np,
        )

        return int(safe_action), q_values_np

    def act(self, obs: np.ndarray, deterministic: bool = False) -> Tuple[int, np.ndarray]:
        """Act without safety filtering (for critic training)."""
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, q_values = self.actor.act(obs_t, deterministic)
        return int(action.cpu().numpy()[0]), q_values.cpu().numpy()[0]

    def store(self, all_obs: np.ndarray, all_actions: np.ndarray, reward: float,
              all_next_obs: np.ndarray, done: bool):
        self.buffer['all_obs'].append(all_obs)
        self.buffer['all_actions'].append(all_actions)
        self.buffer['all_rewards'].append(reward)
        self.buffer['all_next_obs'].append(all_next_obs)
        self.buffer['dones'].append(done)

    def update(self) -> Dict[str, float]:
        """Update using data from this agent's buffer. Uses discrete DQN-style target."""
        if len(self.buffer['all_obs']) < 32:
            return {}

        obs = torch.FloatTensor(np.array(self.buffer['all_obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['all_actions'])).to(self.device)  # (batch, n_agents, action_dim)
        rewards = torch.FloatTensor(np.array(self.buffer['all_rewards'])).to(self.device)
        next_obs = torch.FloatTensor(np.array(self.buffer['all_next_obs'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)

        for k in self.buffer:
            self.buffer[k].clear()

        batch_size = obs.shape[0]
        n_ag = self.n_agents
        adim = self.action_dim
        obs_dim_per = self.obs_dim  # Use stored per-agent obs dim

        # Reshape for critic: obs (batch, n_agents * obs_dim), actions (batch, n_agents * action_dim)
        obs_flat = obs.reshape(batch_size, n_ag * obs_dim_per)
        actions_flat = actions.reshape(batch_size, n_ag * adim)

        # === Critic Update: Double DQN-style target ===
        with torch.no_grad():
            next_obs_flat = next_obs.reshape(batch_size, n_ag * obs_dim_per)
            # Use online actor to select next actions
            next_obs_i = next_obs[:, self.agent_id]  # (batch, obs_dim)
            next_q_vals = self.actor(next_obs_i)  # (batch, action_dim) - Q-values for each action
            next_actions_idx = torch.argmax(next_q_vals, dim=-1)  # (batch,)

            # Build one-hot next actions for all agents (use argmax for each)
            next_actions_all = torch.zeros(batch_size, n_ag * adim, device=self.device)
            for b in range(batch_size):
                for ag in range(n_ag):
                    if ag == self.agent_id:
                        act_idx = next_actions_idx[b]
                    else:
                        # Use random for other agents (simplified)
                        act_idx = torch.randint(0, adim, (1,)).item()
                    next_actions_all[b, ag * adim + act_idx] = 1.0

            next_q = self.target_critic(next_obs_flat, next_actions_all).squeeze(-1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.critic(obs_flat, actions_flat).squeeze(-1)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # === Actor Update: Policy gradient with actual Q-values ===
        # Actor outputs Q-values; we want to maximize Q for chosen actions
        obs_i = obs[:, self.agent_id]  # (batch, obs_dim)
        q_vals = self.actor(obs_i)  # (batch, action_dim)

        # Policy gradient: maximize expected Q under current policy
        # Use softmax policy: gradient = sum_a pi(a) * (Q(a) - V) * grad_pi(a)
        # Simplified: just maximize Q for the actions that would be chosen
        probs = F.softmax(q_vals, dim=-1)
        v_vals = (q_vals * probs).sum(dim=-1)  # State value under current policy

        # Actor loss: maximize Q under policy (minimize negative Q)
        # Use actual taken actions for policy gradient
        action_one_hot = actions_flat[:, self.agent_id * adim:(self.agent_id + 1) * adim]
        q_under_policy = (q_vals * action_one_hot).sum(dim=-1).mean()
        actor_loss = -q_under_policy

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # === Soft Update Target Critic ===
        self._soft_update()

        return {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'mean_q': current_q.mean().item(),
        }

    def _soft_update(self):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)


class SafeMADDPG:
    """
    Safe MADDPG: MADDPG with CBF Safety Layer.

    Key components:
    1. Centralized critics (one per agent, seeing all observations/actions)
    2. Decentralized actors (each sees only local observation)
    3. CBF safety layer that filters unsafe actions

    Reference: "Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments"
                Lowe et al., NeurIPS 2017
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        device: str = "mps",
        actor_lr: float = 1e-3,
        critic_lr: float = 1e-3,
        gamma: float = 0.99,
        tau: float = 0.01,
        collision_threshold: float = 0.5,
        cbf_alpha: float = 3.0,
        batch_size: int = 32,
        buffer_size: int = 100000,
    ):
        self.n_agents = n_agents
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.obs_dims = obs_dims
        self.action_dims = action_dims
        self.collision_threshold = collision_threshold

        # Centralized replay buffer (shared across agents)
        self.replay_buffer = {
            'all_obs': [], 'all_actions': [], 'all_rewards': [],
            'all_next_obs': [], 'dones': []
        }
        self.max_buffer_size = buffer_size

        # Create agents
        self.agents = []
        for i in range(n_agents):
            agent = SafeMADDPGAgent(
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                agent_id=i,
                n_agents=n_agents,
                device=device,
                actor_lr=actor_lr,
                critic_lr=critic_lr,
                gamma=gamma,
                tau=tau,
                collision_threshold=collision_threshold,
                cbf_alpha=cbf_alpha,
            )
            self.agents.append(agent)

        print(f"SafeMADDPG initialized: {n_agents} agents, collision_threshold={collision_threshold}, cbf_alpha={cbf_alpha}")

    def _get_other_positions(self, agent_id: int, agent_positions: np.ndarray) -> List[np.ndarray]:
        """Get positions of all other agents."""
        return [agent_positions[j] for j in range(self.n_agents) if j != agent_id]

    def act(
        self,
        obs_list: List[np.ndarray],
        agent_positions: np.ndarray,
        deterministic: bool = False,
        use_cbf: bool = True,
    ) -> List[int]:
        """Select actions for all agents with optional CBF safety filter."""
        actions = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_list)):
            if use_cbf:
                other_pos = self._get_other_positions(i, agent_positions)
                action, _ = agent.act_with_cbf(
                    obs=obs,
                    current_pos=agent_positions[i],
                    other_positions=other_pos,
                    deterministic=deterministic,
                )
            else:
                action, _ = agent.act(obs, deterministic=deterministic)
            actions.append(action)
        return actions

    def store(
        self,
        all_obs: np.ndarray,
        all_actions: np.ndarray,
        all_rewards: List[float],
        all_next_obs: np.ndarray,
        dones: List[bool],
    ):
        """Store ONE transition (for all agents together) in shared replay buffer.

        all_obs: (n_agents, obs_dim) - observations of all agents
        all_actions: (n_agents,) - raw action indices per agent
        all_rewards: List[float] - rewards per agent
        all_next_obs: (n_agents, obs_dim)
        dones: List[bool] - done flags per agent
        """
        self.replay_buffer['all_obs'].append(all_obs.copy())
        self.replay_buffer['all_actions'].append(all_actions.copy().astype(np.int64))
        self.replay_buffer['all_rewards'].append(list(all_rewards))
        self.replay_buffer['all_next_obs'].append(all_next_obs.copy())
        self.replay_buffer['dones'].append(any(dones))

        # Trim buffer if too large
        if len(self.replay_buffer['all_obs']) > self.max_buffer_size:
            trim_size = len(self.replay_buffer['all_obs']) - self.max_buffer_size
            for k in self.replay_buffer:
                self.replay_buffer[k] = self.replay_buffer[k][trim_size:]

    def update(self) -> List[Dict[str, float]]:
        """Update all agents using sampled batch from replay buffer."""
        if len(self.replay_buffer['all_obs']) < self.batch_size:
            return [{} for _ in range(self.n_agents)]

        # Sample batch indices
        buf_len = len(self.replay_buffer['all_obs'])
        indices = np.random.choice(buf_len, self.batch_size, replace=False)

        # Build batch tensors: all_obs (batch, n_agents, obs_dim), all_actions (batch, n_agents)
        batch_obs = np.array([self.replay_buffer['all_obs'][idx] for idx in indices])
        batch_actions_raw = np.array([self.replay_buffer['all_actions'][idx] for idx in indices])  # (batch, n_agents)
        batch_rewards = np.array([self.replay_buffer['all_rewards'][idx] for idx in indices])  # (batch, n_agents)
        batch_next_obs = np.array([self.replay_buffer['all_next_obs'][idx] for idx in indices])
        batch_dones = np.array([self.replay_buffer['dones'][idx] for idx in indices])

        # Convert raw actions (integers) to one-hot, then flatten for critic
        batch_obs_flat = batch_obs.reshape(self.batch_size, -1)  # (batch, n_agents * obs_dim)
        batch_actions_onehot = np.zeros((self.batch_size, self.n_agents, self.action_dims[0]), dtype=np.float32)
        for b in range(self.batch_size):
            for a in range(self.n_agents):
                action_idx = int(batch_actions_raw[b, a])
                batch_actions_onehot[b, a, action_idx] = 1.0
        batch_actions_flat = batch_actions_onehot.reshape(self.batch_size, -1)  # (batch, n_agents * action_dim)

        # Each agent updates using its own portion of the batch
        results = []
        for i, agent in enumerate(self.agents):
            agent.buffer['all_obs'] = list(batch_obs)
            agent.buffer['all_actions'] = list(batch_actions_onehot)  # one-hot for critic
            agent.buffer['all_rewards'] = list(batch_rewards[:, i])
            agent.buffer['all_next_obs'] = list(batch_next_obs)
            agent.buffer['dones'] = list(batch_dones)

            result = agent.update()
            results.append(result)

        return results

    def save(self, path: str):
        """Save all agent states."""
        torch.save({
            'agents': [
                {
                    'actor': agent.actor.state_dict(),
                    'critic': agent.critic.state_dict(),
                    'target_critic': agent.target_critic.state_dict(),
                }
                for agent in self.agents
            ]
        }, path)

    def load(self, path: str):
        """Load all agent states."""
        data = torch.load(path, map_location=self.device)
        for i, agent_data in enumerate(data['agents']):
            self.agents[i].actor.load_state_dict(agent_data['actor'])
            self.agents[i].critic.load_state_dict(agent_data['critic'])
            self.agents[i].target_critic.load_state_dict(agent_data['target_critic'])


# =============================================================================
# HATRPO: Heterogeneous Agent Trust Region Policy Optimization
# Reference: "Heterogeneous-Agent Trust Region Policy Optimization"
#            Han et al., NeurIPS 2022
# =============================================================================


class HATRPOActor(nn.Module):
    """Policy network for HATRPO with trust region."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x):
        h = self.net(x)
        mean = self.mean_head(h)
        return mean, self.log_std.exp()

    def act(self, x, deterministic=False):
        mean, std = self.forward(x)
        if deterministic:
            action = mean
        else:
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class HATRPOCritic(nn.Module):
    """Value network for HATRPO."""

    def __init__(self, obs_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        return self.net(x)


class HATRPOAgent:
    """
    Single agent for HATRPO.

    Key features:
    1. Trust region constraint: KL(old_pi || new_pi) <= delta
    2. Natural gradient for stable updates
    3. Adaptive step size via line search
    4. GAE for advantage estimation
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        agent_id: int,
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        lam: float = 0.95,
        delta: float = 0.01,  # Trust region KL threshold
        hidden_dim: int = 128,
    ):
        self.device = device
        self.agent_id = agent_id
        self.gamma = gamma
        self.lam = lam
        self.delta = delta

        self.actor = HATRPOActor(obs_dim, action_dim, hidden_dim).to(device)
        self.critic = HATRPOCritic(obs_dim, hidden_dim).to(device)
        self.actor_old = HATRPOActor(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_old.load_state_dict(self.actor.state_dict())

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Buffer for PPO-style updates
        self.buffer = {
            'obs': [], 'actions': [], 'rewards': [], 'values': [],
            'log_probs': [], 'dones': []
        }

    def act(self, obs: np.ndarray, deterministic: bool = False) -> tuple:
        obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action, log_prob, _ = self.actor_old.act(obs_t, deterministic)
            value = self.critic(obs_t)
        return action.cpu().numpy()[0], log_prob.sum().item(), value.sum().item()

    def store(self, obs, action, reward, value, log_prob, done):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['values'].append(value)
        self.buffer['log_probs'].append(log_prob)
        self.buffer['dones'].append(done)

    def _compute_gae(self, rewards, values, dones):
        """Compute GAE (Generalized Advantage Estimation)."""
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        advantages = torch.FloatTensor(advantages).to(self.device)
        returns = advantages + torch.FloatTensor(values).to(self.device)
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return advantages, returns

    def _kl_divergence(self, obs, action_old, action_new):
        """Compute KL divergence between old and new action distributions."""
        mean_old, std_old = self.actor_old(obs)
        mean_new, std_new = self.actor(obs)

        # KL divergence for Gaussian distributions
        kl = torch.log(std_new / std_old) + (std_old ** 2 + (mean_old - mean_new) ** 2) / (2 * std_new ** 2) - 0.5
        return kl.sum(dim=-1).mean()

    def update(self, batch_size: int = 64) -> dict:
        """Update using trust region policy optimization."""
        if len(self.buffer['obs']) < batch_size:
            return {}

        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.buffer['log_probs'])).to(self.device)
        values = torch.FloatTensor(np.array(self.buffer['values'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device)

        # Clear buffer
        for k in self.buffer:
            self.buffer[k].clear()

        # Compute advantages
        advantages, returns = self._compute_gae(
            rewards.cpu().numpy(), values.cpu().numpy(), dones.cpu().numpy()
        )

        # === Critic Update ===
        for _ in range(10):  # Multiple critic updates
            values_pred = self.critic(obs).squeeze(-1)
            critic_loss = F.mse_loss(values_pred, returns.detach())
            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

        # === Actor Update with Trust Region ===
        # Compute surrogate loss and KL constraint
        mean, std = self.actor(obs)
        dist = torch.distributions.Normal(mean, std)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        ratio = torch.exp(log_probs - old_log_probs)

        # PPO-style surrogate
        surr1 = ratio * advantages.detach()
        surr2 = torch.clamp(ratio, 1 - 0.2, 1 + 0.2) * advantages.detach()
        policy_loss = -torch.min(surr1, surr2).mean()

        # === Natural Gradient Update ===
        # Compute gradient of policy loss
        self.actor_opt.zero_grad()
        policy_loss.backward(retain_graph=True)

        # Compute Fisher information (approximation)
        grads = torch.cat([p.grad.view(-1) for p in self.actor.parameters() if p.grad is not None])

        # Natural gradient direction
        with torch.no_grad():
            fisher_inv = 1.0 / (grads.pow(2).mean() + 1e-8)
            nat_grad = grads * fisher_inv

        # Apply natural gradient with line search
        old_params = [p.clone() for p in self.actor.parameters()]

        # Line search for step size
        for step_frac in [0.5, 0.25, 0.125, 0.0625]:
            # Apply natural gradient step
            idx = 0
            for p in self.actor.parameters():
                if p.grad is not None:
                    p.data = old_params[idx] - step_frac * nat_grad[idx:idx + p.numel()].view(p.shape)
                    idx += p.numel()

            # Check KL constraint
            with torch.no_grad():
                kl = self._kl_divergence(obs, None, None)

            if kl <= self.delta:
                break
            else:
                # Revert
                for i, p in enumerate(self.actor.parameters()):
                    p.data = old_params[i]

        # Update old policy
        self.actor_old.load_state_dict(self.actor.state_dict())

        return {
            'policy_loss': policy_loss.item(),
            'critic_loss': critic_loss.item(),
            'kl': kl.item() if 'kl' in dir() else 0.0,
            'mean_value': values_pred.mean().item(),
        }


class HATRPOSystem:
    """
    HATRPO: Heterogeneous Agent Trust Region Policy Optimization.

    Each agent has its own actor-critic pair.
    Trust region ensures stable policy updates.

    Reference: Han et al., NeurIPS 2022
    """

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        device: str = "cuda",
        lr: float = 3e-4,
        gamma: float = 0.99,
        delta: float = 0.01,  # Trust region KL threshold
        hidden_dim: int = 128,
        use_safety_filter: bool = False,
        collision_threshold: float = 0.5,
        safety_alpha: float = 3.0,
    ):
        self.n_agents = n_agents
        self.device = device
        self.gamma = gamma
        self.use_safety_filter = use_safety_filter

        # Create one agent per agent (heterogeneous)
        self.agents = []
        for i in range(n_agents):
            agent = HATRPOAgent(
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                agent_id=i,
                device=device,
                lr=lr,
                gamma=gamma,
                delta=delta,
                hidden_dim=hidden_dim,
            )
            self.agents.append(agent)

        # Optional safety filter
        if use_safety_filter:
            self.safety_filters = [
                CBFSafetyLayer(
                    agent_id=i,
                    n_agents=n_agents,
                    collision_threshold=collision_threshold,
                    alpha=safety_alpha,
                )
                for i in range(n_agents)
            ]
        else:
            self.safety_filters = [None] * n_agents

        # Shared or per-agent buffers
        self.episode_buffer = [[] for _ in range(n_agents)]

        print(f"HATRPO initialized: {n_agents} agents, delta={delta}, safety_filter={use_safety_filter}")

    def _get_other_positions(self, agent_id: int, positions: np.ndarray) -> List[np.ndarray]:
        """Get positions of other agents."""
        return [positions[j] for j in range(self.n_agents) if j != agent_id]

    def act(
        self,
        obs_list: List[np.ndarray],
        positions: np.ndarray,
        deterministic: bool = False,
    ) -> List[int]:
        """Act for all agents, optionally with safety filter."""
        actions = []
        for i, (agent, obs) in enumerate(zip(self.agents, obs_list)):
            action, log_prob, value = agent.act(obs, deterministic)

            # Apply safety filter if enabled
            if self.use_safety_filter and self.safety_filters[i] is not None:
                sf = self.safety_filters[i]
                other_pos = self._get_other_positions(i, positions)
                # For continuous action: discretize and apply filter
                action_np = action.cpu().numpy() if hasattr(action, 'cpu') else action
                q_values_dummy = np.zeros(5)
                safe_action = sf.get_safe_action(
                    current_pos=positions[i],
                    other_positions=other_pos,
                    policy_q_values=q_values_dummy,
                )
                actions.append(safe_action)
            else:
                # Convert continuous to discrete (argmax over direction)
                action_np = action.cpu().numpy() if hasattr(action, 'cpu') else action
                if len(action_np.shape) > 0:
                    action_idx = int(np.argmax(action_np))
                else:
                    action_idx = int(action_np > 0)
                actions.append(action_idx)

        return actions

    def store(self, agent_id: int, obs, action, reward, value, log_prob, done):
        """Store transition for an agent."""
        self.episode_buffer[agent_id].append({
            'obs': obs,
            'action': action,
            'reward': reward,
            'value': value,
            'log_prob': log_prob,
            'done': done,
        })

    def update_all(self) -> List[dict]:
        """Update all agents."""
        results = []
        for agent in self.agents:
            # Convert episode buffer to update
            buffer = self.episode_buffer[agent.agent_id]
            if len(buffer) < 32:
                results.append({})
                continue

            # Pack buffer
            agent.buffer['obs'] = [b['obs'] for b in buffer]
            agent.buffer['actions'] = [b['action'] for b in buffer]
            agent.buffer['rewards'] = [b['reward'] for b in buffer]
            agent.buffer['values'] = [b['value'] for b in buffer]
            agent.buffer['log_probs'] = [b['log_prob'] for b in buffer]
            agent.buffer['dones'] = [b['done'] for b in buffer]

            result = agent.update()
            results.append(result)

            # Clear episode buffer
            self.episode_buffer[agent.agent_id].clear()

        return results

    def reset_buffers(self):
        """Reset all episode buffers."""
        self.episode_buffer = [[] for _ in range(self.n_agents)]

    def save(self, path: str):
        torch.save({
            'agents': [{'actor': a.actor.state_dict(), 'critic': a.critic.state_dict()}
                       for a in self.agents]
        }, path)

    def load(self, path: str):
        data = torch.load(path, map_location=self.device)
        for i, agent_data in enumerate(data['agents']):
            self.agents[i].actor.load_state_dict(agent_data['actor'])
            self.agents[i].critic.load_state_dict(agent_data['critic'])
            self.agents[i].actor_old.load_state_dict(agent_data['actor'])
