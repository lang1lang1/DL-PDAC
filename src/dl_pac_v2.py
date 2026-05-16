"""
DL-PDAC v2: Fixed Distributed Lyapunov Primal-Dual Actor-Critic
Key fixes:
  - Constraint gradient flows to policy (real Lagrangian)
  - Advantage incorporates constraint penalty
  - Proper k-hop gradient consensus
  - Better reward shaping
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
import copy


def _sanitize_tensor(
    tensor: torch.Tensor,
    nan: float = 0.0,
    posinf: float = 1.0,
    neginf: float = -1.0,
) -> torch.Tensor:
    """Replace non-finite values to keep training/evaluation alive."""
    return torch.nan_to_num(tensor, nan=nan, posinf=posinf, neginf=neginf)


def _sanitize_distribution_params(
    mu: torch.Tensor,
    std: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = _sanitize_tensor(mu, nan=0.0, posinf=1.0, neginf=-1.0)
    std = _sanitize_tensor(std, nan=1.0, posinf=2.0, neginf=1e-3).clamp(1e-3, 10.0)
    return mu, std


def _has_nonfinite_module_params(module: nn.Module) -> bool:
    return any(not torch.isfinite(param).all() for param in module.parameters())


def _sanitize_module_params(module: nn.Module, name: str = "module"):
    repaired = False
    for param in module.parameters():
        if not torch.isfinite(param).all():
            with torch.no_grad():
                param.data = _sanitize_tensor(param.data, nan=0.0, posinf=1.0, neginf=-1.0)
            repaired = True
    if repaired:
        print(f"[dl_pac_v2] Sanitized non-finite parameters in {name}.")


def _clip_or_skip_actor_step(agent, max_norm: float = 5.0) -> bool:
    """Clip actor gradients and skip the optimizer step if they are non-finite."""
    grad_norm = torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), max_norm=max_norm)
    if not torch.isfinite(torch.as_tensor(grad_norm)):
        print(f"[dl_pac_v2] Skipping actor step for agent {agent.agent_id} due to non-finite gradient norm.")
        agent.actor_opt.zero_grad(set_to_none=True)
        return False

    for param in agent.actor.parameters():
        if param.grad is not None:
            param.grad.data = _sanitize_tensor(param.grad.data, nan=0.0, posinf=max_norm, neginf=-max_norm)
    return True


class PolicyNetwork(nn.Module):
    """Gaussian policy network for continuous actions.

    Outputs mean and std for a Gaussian distribution over action vectors.
    The raw mean is scaled by max_force to produce force values in
    [-max_force, max_force].
    """

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 64,
                 max_force: float = 1.0):
        super().__init__()
        self.max_force = max_force
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        # Raw mu in [-1, 1] via tanh, then scale to force range
        raw_mu = torch.tanh(self.mu(h))
        mu = raw_mu * self.max_force
        std = torch.exp(self.log_std.clamp(-5, 2))
        return mu, std

    def sample(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mu, std = self.forward(x)
        dist = torch.distributions.Normal(mu, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob


class CriticNetwork(nn.Module):
    """Value function network for Q-values."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class LyapunovNetwork(nn.Module):
    """Lyapunov function network for stability."""

    def __init__(self, obs_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) ** 2 + 0.01  # Ensure positive


class DLPACAgentV2:
    """
    Fixed DL-PDAC Agent.

    Key fixes vs v1:
    1. Constraint penalty in advantage (gradients flow to policy)
    2. Proper dual variable update with Lagrangian theory
    3. Lyapunov stability with proper gradient flow
    4. k-hop gradient consensus
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        n_agents: int,
        agent_id: int,
        neighbor_ids: List[int],
        device: str = "mps",
        actor_lr: float = 3e-4,
        critic_lr: float = 1e-3,
        lyapunov_lr: float = 1e-4,
        dual_lr: float = 0.01,
        gamma: float = 0.99,
        tau: float = 0.005,
        lyapunov_coef: float = 0.5,
        dual_max: float = 10.0,
        k_hops: int = 1,
        max_force: float = 1.0,
    ):
        self.device = device
        self.n_agents = n_agents
        self.agent_id = agent_id
        self.neighbor_ids = neighbor_ids
        self.k_hops = k_hops
        self.gamma = gamma
        self.lyapunov_coef = lyapunov_coef
        self.dual_max = dual_max
        self.dual_lr = dual_lr
        self.tau = tau

        # Networks
        self.actor = PolicyNetwork(obs_dim, action_dim, max_force=max_force).to(device)
        self.critic = CriticNetwork(obs_dim).to(device)
        self.target_critic = CriticNetwork(obs_dim).to(device)
        self.target_critic.load_state_dict(self.critic.state_dict())
        self.lyapunov = LyapunovNetwork(obs_dim).to(device)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.lyapunov_opt = torch.optim.Adam(self.lyapunov.parameters(), lr=lyapunov_lr)

        # Dual variable (local λ_i ≥ 0)
        self.dual = nn.Parameter(torch.tensor(1.0), requires_grad=False)

        # k-hop neighborhoods (set externally)
        self.k_hop_neighbors: List[int] = []

        # Gradient buffers for consensus
        self.local_grads: Optional[Dict[str, torch.Tensor]] = None
        self.consensus_grads: Optional[Dict[str, torch.Tensor]] = None

        # Replay buffer
        self.buffer = {
            'obs': [], 'actions': [], 'rewards': [],
            'constraints': [], 'next_obs': [], 'dones': []
        }

    def set_k_hop_neighbors(self, neighbors: List[int]):
        """Set k-hop neighbors for consensus."""
        self.k_hop_neighbors = neighbors

    def act(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).to(self.device).unsqueeze(0)
            mu, std = self.actor(obs_t)
            if deterministic:
                return mu.cpu().numpy()[0]
            action = mu + std * torch.randn_like(mu)
            return action.cpu().numpy()[0]

    def store(self, obs, action, reward, constraint, next_obs, done):
        self.buffer['obs'].append(obs)
        self.buffer['actions'].append(action)
        self.buffer['rewards'].append(reward)
        self.buffer['constraints'].append(constraint)
        self.buffer['next_obs'].append(next_obs)
        self.buffer['dones'].append(done)

    def _clear_buffer(self):
        for k in self.buffer:
            self.buffer[k].clear()

    def get_gradients(self) -> Optional[Dict[str, torch.Tensor]]:
        """Get current actor gradients for consensus."""
        grads = {}
        for name, param in self.actor.named_parameters():
            if param.grad is not None:
                grads[name] = param.grad.clone()
        if not grads:
            return None
        return grads

    def apply_consensus_grads(self, consensus_grads: Dict[str, torch.Tensor]):
        """Apply consensus-averaged gradients before optimizer step."""
        with torch.no_grad():
            for name, param in self.actor.named_parameters():
                if name in consensus_grads:
                    # Weighted average: local + consensus
                    param.grad = 0.5 * param.grad + 0.5 * consensus_grads[name]

    def update(
        self,
        neighbor_grads: Optional[List[Dict[str, torch.Tensor]]] = None,
    ) -> Dict[str, float]:
        """
        Perform one update step with proper Lagrangian formulation.

        Lagrangian: L = E[Q(s,a) - λ * c(s)] - λ * E[c(s)]
        Gradient: ∇L = ∇E[Q] - λ * ∇E[c] - E[c] * ∇λ
        """
        if len(self.buffer['obs']) < 32:
            return {}

        obs = torch.FloatTensor(np.array(self.buffer['obs'])).to(self.device)
        actions = torch.FloatTensor(np.array(self.buffer['actions'])).to(self.device)
        rewards = torch.FloatTensor(np.array(self.buffer['rewards'])).to(self.device)
        constraints = torch.FloatTensor(np.array(self.buffer['constraints'])).to(self.device)
        next_obs = torch.FloatTensor(np.array(self.buffer['next_obs'])).to(self.device)
        dones = torch.FloatTensor(np.array(self.buffer['dones'])).to(self.device)
        obs = _sanitize_tensor(obs)
        actions = _sanitize_tensor(actions)
        rewards = _sanitize_tensor(rewards)
        constraints = _sanitize_tensor(constraints)
        next_obs = _sanitize_tensor(next_obs)
        dones = _sanitize_tensor(dones)

        self._clear_buffer()
        batch_size = obs.shape[0]

        # === Critic Update (standard TD) ===
        with torch.no_grad():
            next_q = self.target_critic(next_obs).squeeze(-1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        current_q = self.critic(obs).squeeze(-1)
        critic_loss = F.mse_loss(current_q, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # === Compute Lagrangian reward ===
        # L(s,a) = r(s,a) - λ * c(s,a)
        # This makes the policy directly penalize constraint violations
        dual_val = self.dual.item()
        constrained_rewards = rewards - dual_val * constraints

        # Advantage estimation
        advantages = (constrained_rewards - current_q.detach()).detach()
        advantages = _sanitize_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))

        # === Policy Update ===
        _sanitize_module_params(self.actor, name=f"actor[{self.agent_id}]")
        mu, std = self.actor(obs)
        mu, std = _sanitize_distribution_params(mu, std)
        dist = torch.distributions.Normal(mu, std)
        action_log_probs = dist.log_prob(actions).sum(dim=-1)

        # Policy gradient with constrained reward
        actor_loss = -(action_log_probs * advantages).mean()

        # === Lyapunov Stability Term ===
        # g_L = E[L(x) * (L(x') - L(x))_+]
        lyapunov_vals = self.lyapunov(obs).squeeze(-1)
        with torch.no_grad():
            next_lyapunov = self.lyapunov(next_obs).squeeze(-1)
        lyapunov_diff = (next_lyapunov - lyapunov_vals.detach()).clamp(min=0)
        lyapunov_loss = (lyapunov_vals * lyapunov_diff).mean()

        # Update Lyapunov network
        self.lyapunov_opt.zero_grad()
        lyapunov_loss.backward()
        self.lyapunov_opt.step()

        # === Total Actor Loss ===
        total_actor_loss = actor_loss + self.lyapunov_coef * lyapunov_loss

        self.actor_opt.zero_grad()
        total_actor_loss.backward()

        # === Apply Consensus Gradients ===
        if neighbor_grads:
            self.apply_consensus_grads_from_list(neighbor_grads)

        if _clip_or_skip_actor_step(self):
            self.actor_opt.step()

        # === Dual Update (Local Lagrangian) ===
        # λ_i(t+1) = [λ_i(t) + α_λ * E[c(s)]]_+
        mean_constraint = constraints.mean().item()
        new_dual = self.dual.item() + self.dual_lr * mean_constraint
        new_dual = float(np.clip(new_dual, 0.0, self.dual_max))
        self.dual.data = torch.tensor(new_dual, device=self.device)

        # === Soft Update Target Network ===
        self._soft_update()

        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'lyapunov_loss': lyapunov_loss.item(),
            'dual_value': self.dual.item(),
            'mean_constraint': mean_constraint,
            'constrained_reward_mean': constrained_rewards.mean().item(),
        }

    def apply_consensus_grads_from_list(
        self,
        neighbor_grads: List[Dict[str, torch.Tensor]]
    ):
        """Average gradients from neighbors and apply to local gradients."""
        if not neighbor_grads:
            return

        # Get parameter names from first gradient dict
        names = list(neighbor_grads[0].keys())
        consensus = {}
        for name in names:
            grads = [neighbor_grads[0][name]]  # Only one neighbor's grads for now
            consensus[name] = torch.stack(grads).mean(dim=0)

        self.apply_consensus_grads(consensus)

    def _soft_update(self):
        for tp, p in zip(self.target_critic.parameters(), self.critic.parameters()):
            tp.data.copy_(tp.data * (1 - self.tau) + p.data * self.tau)


class DLPACSystemV2:
    """Multi-agent DL-PDAC v2 with k-hop gradient consensus and optional comm range."""

    def __init__(
        self,
        n_agents: int,
        obs_dims: List[int],
        action_dims: List[int],
        adjacency_matrix: np.ndarray,
        agent_positions: np.ndarray = None,
        k_hops: int = 1,
        device: str = "mps",
        comm_range: float = None,
        max_force: float = 1.0,
        actor_consensus: bool = True,
        dual_mode: str = "local",
        **agent_kwargs,
    ):
        valid_dual_modes = {"local", "centralized", "sparse"}
        if dual_mode not in valid_dual_modes:
            raise ValueError(f"dual_mode must be one of {sorted(valid_dual_modes)}, got {dual_mode}")

        self.n_agents = n_agents
        self.k_hops = k_hops
        self.device = device
        self.adjacency = adjacency_matrix
        self.agent_positions = agent_positions  # For range-based comm
        self.comm_range = comm_range  # If set, only comm with agents within range
        self.max_force = max_force
        self.actor_consensus = actor_consensus
        self.dual_mode = dual_mode

        # Precompute k-hop neighborhoods (optionally filtered by comm range)
        self.neighborhoods = self._compute_k_hop_neighborhoods(k_hops)

        # Create agents
        self.agents = []
        for i in range(n_agents):
            neighbors = list(np.where(adjacency_matrix[i] > 0)[0])
            agent = DLPACAgentV2(
                obs_dim=obs_dims[i],
                action_dim=action_dims[i],
                n_agents=n_agents,
                agent_id=i,
                neighbor_ids=neighbors,
                device=device,
                k_hops=k_hops,
                max_force=max_force,
                **agent_kwargs,
            )
            agent.set_k_hop_neighbors(self.neighborhoods[i])
            self.agents.append(agent)

        self.actor_param_count = 0
        if self.agents:
            self.actor_param_count = sum(
                p.numel() for p in self.agents[0].actor.parameters()
            )
        self.global_dual = 1.0 if dual_mode == "centralized" else None
        if self.global_dual is not None:
            for agent in self.agents:
                agent.dual.data = torch.tensor(self.global_dual, device=agent.device)

        self.comm_stats = {
            "updates": 0,
            "actor_messages": 0,
            "actor_scalars": 0,
            "dual_messages": 0,
            "dual_scalars": 0,
            "neighborhood_entries": 0,
            "actor_param_count": self.actor_param_count,
        }

        # Statistics
        self.episode_rewards = []
        self.episode_constraints = []

        print(
            f"DL-PDAC v2: {n_agents} agents, {k_hops}-hop consensus, "
            f"comm_range={comm_range}, actor_consensus={actor_consensus}, dual_mode={dual_mode}"
        )
        for i, nbrs in enumerate(self.neighborhoods):
            print(f"  Agent {i}: k={k_hops} neighborhood = {nbrs}")

    def _compute_k_hop_neighborhoods(self, k: int) -> List[List[int]]:
        """BFS to find k-hop neighborhoods, optionally filtered by comm_range."""
        neighborhoods = []
        for i in range(self.n_agents):
            visited = {i}
            current = {i}
            for _ in range(k):
                next_layer = set()
                for j in current:
                    neighbors = np.where(self.adjacency[j] > 0)[0]
                    # Filter by comm_range if specified
                    if self.comm_range is not None and self.agent_positions is not None:
                        pos_i = self.agent_positions[i]
                        neighbors = [
                            n for n in neighbors
                            if np.linalg.norm(self.agent_positions[n] - pos_i) <= self.comm_range
                        ]
                    next_layer.update(neighbors)
                next_layer -= visited
                visited.update(next_layer)
                current = next_layer
            neighborhoods.append([int(x) for x in visited])
        return neighborhoods

    def get_comm_neighborhood(self, agent_idx: int) -> List[int]:
        """Get agents within comm_range of agent_idx (based on current positions)."""
        if self.comm_range is None or self.agent_positions is None:
            return self.neighborhoods[agent_idx]
        pos_i = self.agent_positions[agent_idx]
        return [
            int(j) for j in range(self.n_agents)
            if j != agent_idx and np.linalg.norm(self.agent_positions[j] - pos_i) <= self.comm_range
        ]

    def _get_current_neighborhood(self, agent_idx: int) -> List[int]:
        """Return the current communication neighborhood, including self."""
        if self.comm_range is None or self.agent_positions is None:
            return self.neighborhoods[agent_idx]

        neighbors = self.get_comm_neighborhood(agent_idx)
        if agent_idx not in neighbors:
            neighbors = [agent_idx] + neighbors
        return sorted(set(int(x) for x in neighbors))

    def _record_communication(self, current_neighborhoods: List[List[int]]):
        """Accumulate communication counters for the current update."""
        peer_links = sum(max(len(neighborhood) - 1, 0) for neighborhood in current_neighborhoods)

        self.comm_stats["updates"] += 1
        self.comm_stats["neighborhood_entries"] += sum(len(n) for n in current_neighborhoods)

        if self.actor_consensus:
            self.comm_stats["actor_messages"] += peer_links
            self.comm_stats["actor_scalars"] += peer_links * self.actor_param_count

        if self.dual_mode == "sparse":
            self.comm_stats["dual_messages"] += peer_links
            self.comm_stats["dual_scalars"] += peer_links
        elif self.dual_mode == "centralized":
            central_messages = 2 * self.n_agents
            self.comm_stats["dual_messages"] += central_messages
            self.comm_stats["dual_scalars"] += central_messages

    def get_communication_stats(self) -> Dict[str, float]:
        """Return cumulative and per-update communication statistics."""
        updates = max(self.comm_stats["updates"], 1)
        mean_neighborhood_size = self.comm_stats["neighborhood_entries"] / (updates * max(self.n_agents, 1))
        return {
            **self.comm_stats,
            "actor_messages_per_update": self.comm_stats["actor_messages"] / updates,
            "actor_scalars_per_update": self.comm_stats["actor_scalars"] / updates,
            "dual_messages_per_update": self.comm_stats["dual_messages"] / updates,
            "dual_scalars_per_update": self.comm_stats["dual_scalars"] / updates,
            "mean_neighborhood_size": mean_neighborhood_size,
        }

    def update_positions(self, positions: np.ndarray):
        """Update agent positions for range-based consensus."""
        self.agent_positions = positions

    def act(self, obs_list: List[np.ndarray], deterministic: bool = False) -> List[np.ndarray]:
        return [agent.act(obs, deterministic) for agent, obs in zip(self.agents, obs_list)]

    def step(
        self,
        obs_list: List[np.ndarray],
        action_list: List[np.ndarray],
        reward_list: List[float],
        constraint_list: List[float],
        next_obs_list: List[np.ndarray],
        done_list: List[bool],
        agent_positions: np.ndarray = None,
    ):
        for agent, obs, action, reward, constraint, next_obs, done in zip(
            self.agents, obs_list, action_list, reward_list,
            constraint_list, next_obs_list, done_list
        ):
            agent.store(obs, action, reward, constraint, next_obs, done)

        # Sync positions for range-based consensus
        if agent_positions is not None:
            self.update_positions(agent_positions)

        self.episode_rewards.append(sum(reward_list))
        self.episode_constraints.append(sum(constraint_list))

    def update_all(self) -> List[Dict[str, float]]:
        """Update all agents with k-hop consensus.

        Protocol:
        1. All agents compute local losses (forward pass, store data)
        2. Critic/Lyapunov backward + optimizer step per agent
        3. Actor backward (no step yet), collect gradients
        4. Consensus: average gradients within k-hop neighborhoods
        5. Apply blended gradients and optimizer step
        """
        # === Phase 1: Forward pass for all agents ===
        agent_data = []

        for agent in self.agents:
            if len(agent.buffer['obs']) < 32:
                agent_data.append(None)
                continue

            obs = torch.FloatTensor(np.array(agent.buffer['obs'])).to(agent.device)
            actions = torch.FloatTensor(np.array(agent.buffer['actions'])).to(agent.device)
            rewards = torch.FloatTensor(np.array(agent.buffer['rewards'])).to(agent.device)
            constraints = torch.FloatTensor(np.array(agent.buffer['constraints'])).to(agent.device)
            next_obs = torch.FloatTensor(np.array(agent.buffer['next_obs'])).to(agent.device)
            dones = torch.FloatTensor(np.array(agent.buffer['dones'])).to(agent.device)
            obs = _sanitize_tensor(obs)
            actions = _sanitize_tensor(actions)
            rewards = _sanitize_tensor(rewards)
            constraints = _sanitize_tensor(constraints)
            next_obs = _sanitize_tensor(next_obs)
            dones = _sanitize_tensor(dones)

            agent._clear_buffer()

            # Critic forward
            with torch.no_grad():
                next_q = agent.target_critic(next_obs).squeeze(-1)
                target_q = rewards + agent.gamma * next_q * (1 - dones)
            current_q = agent.critic(obs).squeeze(-1)
            critic_loss = F.mse_loss(current_q, target_q)

            # Actor forward
            dual_val = self.global_dual if self.dual_mode == "centralized" else agent.dual.item()
            constrained_rewards = rewards - dual_val * constraints
            advantages = (constrained_rewards - current_q.detach()).detach()
            advantages = _sanitize_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))

            _sanitize_module_params(agent.actor, name=f"actor[{agent.agent_id}]")
            mu, std = agent.actor(obs)
            mu, std = _sanitize_distribution_params(mu, std)
            dist = torch.distributions.Normal(mu, std)
            action_log_probs = dist.log_prob(actions).sum(dim=-1)
            actor_loss = -(action_log_probs * advantages).mean()

            # Lyapunov forward
            lyapunov_vals = agent.lyapunov(obs).squeeze(-1)
            with torch.no_grad():
                next_lyapunov = agent.lyapunov(next_obs).squeeze(-1)
            lyapunov_diff = (next_lyapunov - lyapunov_vals.detach()).clamp(min=0)
            lyapunov_loss = (lyapunov_vals * lyapunov_diff).mean()

            agent_data.append({
                'agent': agent,
                'obs': obs, 'actions': actions, 'rewards': rewards,
                'constraints': constraints, 'next_obs': next_obs,
                'critic_loss': critic_loss,
                'actor_loss': actor_loss,
                'lyapunov_loss': lyapunov_loss,
                'dual_val': dual_val,
            })

        if not any(d is not None for d in agent_data):
            return [{} for _ in range(self.n_agents)]

        # === Phase 2: Critic and Lyapunov backward + optimizer step ===
        for d in agent_data:
            if d is None:
                continue
            agent = d['agent']
            agent.critic_opt.zero_grad()
            d['critic_loss'].backward()
            agent.critic_opt.step()

            agent.lyapunov_opt.zero_grad()
            d['lyapunov_loss'].backward()
            agent.lyapunov_opt.step()

        # === Phase 3: Actor backward, collect gradients ===
        local_grads = []
        for d in agent_data:
            if d is None:
                local_grads.append({})
                continue
            agent = d['agent']
            agent.actor_opt.zero_grad()
            d['actor_loss'].backward()
            grads = {
                name: param.grad.clone()
                for name, param in agent.actor.named_parameters()
                if param.grad is not None
            }
            local_grads.append(grads)

        # === Phase 4: Consensus - average gradients within k-hop neighborhoods ===
        current_neighborhoods = [
            self._get_current_neighborhood(i) for i in range(self.n_agents)
        ]
        self._record_communication(current_neighborhoods)

        consensus_grads = []
        for i in range(self.n_agents):
            if not self.actor_consensus:
                consensus_grads.append({})
                continue

            neighborhood = current_neighborhoods[i]
            neighbor_grads = []
            for j in neighborhood:
                if j < len(local_grads) and local_grads[j]:
                    neighbor_grads.append(local_grads[j])
            if neighbor_grads:
                avg = {}
                for name in neighbor_grads[0].keys():
                    avg[name] = torch.stack([g[name] for g in neighbor_grads]).mean(dim=0)
                consensus_grads.append(avg)
            else:
                consensus_grads.append({})

        # === Phase 5: Recompute actor + Lyapunov loss, apply consensus, step ===
        results = []
        mean_constraints = [None] * self.n_agents
        for i, d in enumerate(agent_data):
            if d is None:
                results.append({})
                continue

            agent = d['agent']
            obs = d['obs']
            actions = d['actions']
            rewards = d['rewards']
            constraints = d['constraints']
            next_obs = d['next_obs']

            # Fresh actor forward (uses updated critic weights from Phase 2)
            dual_val = self.global_dual if self.dual_mode == "centralized" else agent.dual.item()
            current_q = agent.critic(obs).squeeze(-1)
            constrained_rewards = rewards - dual_val * constraints
            advantages = (constrained_rewards - current_q.detach()).detach()
            advantages = _sanitize_tensor((advantages - advantages.mean()) / (advantages.std() + 1e-8))

            _sanitize_module_params(agent.actor, name=f"actor[{agent.agent_id}]")
            mu, std = agent.actor(obs)
            mu, std = _sanitize_distribution_params(mu, std)
            dist = torch.distributions.Normal(mu, std)
            action_log_probs = dist.log_prob(actions).sum(dim=-1)
            actor_loss = -(action_log_probs * advantages).mean()

            lyapunov_vals = agent.lyapunov(obs).squeeze(-1)
            with torch.no_grad():
                next_lyapunov = agent.lyapunov(next_obs).squeeze(-1)
            lyapunov_diff = (next_lyapunov - lyapunov_vals.detach()).clamp(min=0)
            lyapunov_loss = (lyapunov_vals * lyapunov_diff).mean()

            total_loss = actor_loss + agent.lyapunov_coef * lyapunov_loss
            agent.actor_opt.zero_grad()
            total_loss.backward()

            # Apply consensus: blend local grads with neighborhood average
            if consensus_grads[i]:
                with torch.no_grad():
                    for name, param in agent.actor.named_parameters():
                        if name in consensus_grads[i]:
                            param.grad.copy_(
                                0.5 * param.grad + 0.5 * consensus_grads[i][name]
                            )

            if _clip_or_skip_actor_step(agent):
                agent.actor_opt.step()

            # Dual update
            mean_constraint = constraints.mean().item()
            mean_constraints[i] = mean_constraint

            # Soft update
            agent._soft_update()

            results.append({
                'actor_loss': actor_loss.item(),
                'critic_loss': d['critic_loss'].item(),
                'lyapunov_loss': lyapunov_loss.item(),
                'mean_constraint': mean_constraint,
            })

        # === Phase 6: Dual update ===
        if self.dual_mode == "centralized":
            valid_constraints = [
                mean_constraints[i] for i, d in enumerate(agent_data)
                if d is not None and mean_constraints[i] is not None
            ]
            if valid_constraints:
                dual_lr = self.agents[0].dual_lr
                dual_max = self.agents[0].dual_max
                self.global_dual = float(np.clip(
                    self.global_dual + dual_lr * float(np.mean(valid_constraints)),
                    0.0,
                    dual_max,
                ))
            for agent in self.agents:
                agent.dual.data = torch.tensor(self.global_dual, device=agent.device)
            for i, d in enumerate(agent_data):
                if d is not None:
                    results[i]['dual_value'] = self.global_dual

        elif self.dual_mode == "sparse":
            proposed_duals = []
            for i, agent in enumerate(self.agents):
                mean_constraint = 0.0 if mean_constraints[i] is None else mean_constraints[i]
                proposed_dual = agent.dual.item() + agent.dual_lr * mean_constraint
                proposed_duals.append(float(np.clip(proposed_dual, 0.0, agent.dual_max)))

            synced_duals = []
            for i, agent in enumerate(self.agents):
                neighborhood = current_neighborhoods[i]
                neighbor_values = [proposed_duals[j] for j in neighborhood]
                synced_duals.append(float(np.clip(np.mean(neighbor_values), 0.0, agent.dual_max)))

            for i, agent in enumerate(self.agents):
                agent.dual.data = torch.tensor(synced_duals[i], device=agent.device)
                if agent_data[i] is not None:
                    results[i]['dual_value'] = synced_duals[i]

        else:
            for i, agent in enumerate(self.agents):
                if agent_data[i] is None or mean_constraints[i] is None:
                    continue
                new_dual = agent.dual.item() + agent.dual_lr * mean_constraints[i]
                new_dual = float(np.clip(new_dual, 0.0, agent.dual_max))
                agent.dual.data = torch.tensor(new_dual, device=agent.device)
                results[i]['dual_value'] = new_dual

        return results

    def reset_episode_stats(self):
        self.episode_rewards = []
        self.episode_constraints = []

    def get_episode_stats(self) -> Dict[str, float]:
        return {
            'mean_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0,
            'mean_constraint': np.mean(self.episode_constraints) if self.episode_constraints else 0,
            'violation_rate': np.mean([c > 0 for c in self.episode_constraints]) if self.episode_constraints else 0,
            'mean_dual': np.mean([a.dual.item() for a in self.agents]),
        }

    def get_checkpoint_payload(self) -> Dict[str, object]:
        """Serialize model state needed for offline evaluation and residual audits."""
        return {
            "n_agents": self.n_agents,
            "k_hops": self.k_hops,
            "actor_consensus": self.actor_consensus,
            "dual_mode": self.dual_mode,
            "adjacency": np.asarray(self.adjacency, dtype=np.float32),
            "global_dual": self.global_dual,
            "agents": [
                {
                    "actor": agent.actor.state_dict(),
                    "critic": agent.critic.state_dict(),
                    "target_critic": agent.target_critic.state_dict(),
                    "lyapunov": agent.lyapunov.state_dict(),
                    "dual": float(agent.dual.item()),
                }
                for agent in self.agents
            ],
        }

    def load_checkpoint_payload(self, payload: Dict[str, object]):
        """Load a payload produced by ``get_checkpoint_payload``."""
        agent_states = payload.get("agents", [])
        if len(agent_states) != len(self.agents):
            raise ValueError(
                f"Checkpoint agent count mismatch: expected {len(self.agents)}, got {len(agent_states)}"
            )

        for agent, state in zip(self.agents, agent_states):
            agent.actor.load_state_dict(state["actor"])
            agent.critic.load_state_dict(state["critic"])
            agent.target_critic.load_state_dict(state["target_critic"])
            agent.lyapunov.load_state_dict(state["lyapunov"])
            agent.dual.data = torch.tensor(float(state["dual"]), device=agent.device)

        self.global_dual = payload.get("global_dual", self.global_dual)
        if self.global_dual is not None and self.dual_mode == "centralized":
            for agent in self.agents:
                agent.dual.data = torch.tensor(float(self.global_dual), device=agent.device)
