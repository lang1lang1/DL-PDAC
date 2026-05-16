"""
Multi-Agent Particle Environment with Collision Avoidance Constraint.
Built on PettingZoo but with constraint tracking.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import gymnasium as gym
from gymnasium import spaces
from pettingzoo import ParallelEnv
import math


class MultiAgentParticleEnv(ParallelEnv):
    """
    N-agent particle environment with collision avoidance.

    Constraint: distance between any two agents must be >= 0.5m.
    """

    metadata = {"render_modes": [], "name": "multiagent_particle_v0"}

    @staticmethod
    def compute_boundary_size(n_agents: int, base_size: float = 2.0) -> float:
        """Dynamically compute boundary size to keep agent density constant.

        Area scales with sqrt(n_agents / 4) to maintain constant density.
        This ensures collision probability stays roughly constant across agent counts.
        """
        return base_size * math.sqrt(n_agents / 4)

    def __init__(
        self,
        n_agents: int = 4,
        n_landmarks: int = 2,
        max_steps: int = 100,
        agent_size: float = 0.1,
        collision_threshold: float = 0.5,
        boundary_size: float = None,
        seed: int = 42,
        vision_range: float = 3.0,
    ):
        super().__init__()

        self.n_agents = n_agents
        self.n_landmarks = n_landmarks
        self.max_steps = max_steps
        self.agent_size = agent_size
        self.collision_threshold = collision_threshold
        # Use dynamic boundary if not specified
        self.boundary_size = boundary_size or self.compute_boundary_size(n_agents)
        self.seed = seed
        self.vision_range = vision_range

        self.possible_agents = [f"agent_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents.copy()

        # Obs: self_pos(2) + self_vel(2) + vision((n-1)*4) + landmarks(n_lm*2)
        # vision: [norm_rel_x, norm_rel_y, norm_dist, presence] per agent
        # max neighbors = n_agents - 1
        self.vision_dim = (n_agents - 1) * 4
        self.obs_dim = 4 + self.vision_dim + n_landmarks * 2
        self.action_dim = 5  # up, down, left, right, stay

        self.observation_spaces = {
            agent: spaces.Box(low=-20, high=20, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.action_dim)
            for agent in self.agents
        }

        self._agent_ids = {agent: i for i, agent in enumerate(self.agents)}

    def reset(self, seed: int = None) -> Dict[str, np.ndarray]:
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)

        self.agents = self.possible_agents.copy()
        self.timestep = 0

        # Initialize agents with minimum safe distance
        self.agent_positions = np.zeros((self.n_agents, 2))
        for i in range(self.n_agents):
            for attempt in range(100):
                pos = np.random.uniform(
                    -self.boundary_size / 2, self.boundary_size / 2, size=2
                )
                min_dist = min([
                    np.linalg.norm(pos - self.agent_positions[j])
                    for j in range(i) if i > 0
                ]) if i > 0 else float('inf')
                if min_dist >= self.collision_threshold:
                    self.agent_positions[i] = pos
                    break
            else:
                self.agent_positions[i] = np.random.uniform(
                    -self.boundary_size / 2, self.boundary_size / 2, size=2
                )

        self.agent_velocities = np.zeros((self.n_agents, 2))

        # Initialize landmarks (targets)
        self.landmarks = np.random.uniform(
            -self.boundary_size / 2, self.boundary_size / 2,
            size=(self.n_landmarks, 2)
        )

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def predict_collision_risk(self, agent_idx: int, action: int) -> float:
        """Return minimum distance to any other agent if taking this action."""
        action_vector = self._action_to_velocity(action)
        next_pos = self.agent_positions[agent_idx] + action_vector * 0.1
        min_dist = float('inf')
        for j in range(self.n_agents):
            if j != agent_idx:
                dist = np.linalg.norm(next_pos - self.agent_positions[j])
                min_dist = min(min_dist, dist)
        return min_dist

    def compute_proxy_violation(self, agent_idx: int, action: int) -> float:
        """Deterministic one-step proxy violation used by the discrete runtime filter."""
        return float(self.collision_threshold - self.predict_collision_risk(agent_idx, action))

    def evaluate_filter_metrics(self, agent_idx: int, policy_action: np.ndarray) -> Dict[str, object]:
        """Return discrete runtime-filter metrics for one agent at the current state."""
        chosen = int(np.argmax(policy_action))
        proxy_scores = np.asarray(
            [self.compute_proxy_violation(agent_idx, a) for a in range(self.action_dim)],
            dtype=np.float32,
        )
        feasible_actions = np.flatnonzero(proxy_scores <= 0.0).astype(int).tolist()
        epsilon = float(np.min(np.clip(proxy_scores, a_min=0.0, a_max=None)))

        if proxy_scores[chosen] <= 0.0:
            selected = chosen
            branch = "raw_safe"
        elif feasible_actions:
            selected = 4 if 4 in feasible_actions else feasible_actions[0]
            branch = "fallback_safe"
        else:
            # All candidates violate the proxy. Choose the least risky action, i.e.
            # the candidate with the smallest positive proxy violation.
            selected = int(np.argmin(proxy_scores))
            branch = "least_risky"

        return {
            "raw_action": chosen,
            "selected_action": selected,
            "filter_applied": bool(selected != chosen),
            "branch": branch,
            "safe_action_exists": bool(feasible_actions),
            "safe_action_count": int(len(feasible_actions)),
            "raw_proxy_violation": float(proxy_scores[chosen]),
            "selected_proxy_violation": float(proxy_scores[selected]),
            "epsilon": epsilon,
        }

    def get_safe_action(
        self,
        agent_idx: int,
        policy_action: np.ndarray,
        return_metrics: bool = False,
    ):
        """Override policy action to be safe. Returns safest action index."""
        metrics = self.evaluate_filter_metrics(agent_idx, policy_action)
        if return_metrics:
            return int(metrics["selected_action"]), metrics
        return int(metrics["selected_action"])

    def get_state_snapshot(self) -> Dict[str, object]:
        """Capture the mutable simulator state for one-step counterfactual audits."""
        return {
            "agents": self.agents.copy(),
            "timestep": int(self.timestep),
            "agent_positions": self.agent_positions.copy(),
            "agent_velocities": self.agent_velocities.copy(),
            "landmarks": self.landmarks.copy(),
            "seed": int(self.seed),
        }

    def set_state_snapshot(self, snapshot: Dict[str, object]):
        """Restore a state previously captured by ``get_state_snapshot``."""
        self.agents = list(snapshot["agents"])
        self.timestep = int(snapshot["timestep"])
        self.agent_positions = np.asarray(snapshot["agent_positions"], dtype=np.float32).copy()
        self.agent_velocities = np.asarray(snapshot["agent_velocities"], dtype=np.float32).copy()
        self.landmarks = np.asarray(snapshot["landmarks"], dtype=np.float32).copy()
        self.seed = int(snapshot["seed"])

    def step(
        self,
        actions: Dict[str, int],
        use_safety_filter: bool = False,
        policy_actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        if not actions:
            raise ValueError("Actions dictionary is empty")

        # Apply safety filter if enabled
        filter_metrics = {}
        if use_safety_filter and policy_actions:
            for agent in actions:
                idx = self._agent_ids[agent]
                safe_act, metrics = self.get_safe_action(
                    idx,
                    policy_actions[agent],
                    return_metrics=True,
                )
                actions[agent] = safe_act
                filter_metrics[agent] = metrics

        # Apply actions
        velocities = np.zeros((self.n_agents, 2))
        for agent, action in actions.items():
            idx = self._agent_ids[agent]
            action_vector = self._action_to_velocity(action)
            velocities[idx] = action_vector

        # Update positions
        self.agent_positions += velocities * 0.1
        self.agent_velocities = velocities

        # Boundary check
        self.agent_positions = np.clip(
            self.agent_positions,
            -self.boundary_size / 2,
            self.boundary_size / 2
        )

        # Compute rewards and constraints
        rewards = {}
        constraints = {}
        for agent in self.agents:
            idx = self._agent_ids[agent]
            pos = self.agent_positions[idx]

            # Reward: distance to nearest landmark
            dist_to_landmark = np.min([
                np.linalg.norm(pos - lm) for lm in self.landmarks
            ])
            reward = -dist_to_landmark * 0.05  # Lower weight for exploration

            # Positive safety reward: bonus for staying collision-free
            reward += 0.5  # Step survival bonus (encourage longer episodes)

            # Compute collision penalty for constraint tracking
            collision_penalty = 0.0
            for j in range(self.n_agents):
                if j != idx:
                    dist = np.linalg.norm(self.agent_positions[idx] - self.agent_positions[j])
                    if dist < self.collision_threshold:
                        collision_penalty += (self.collision_threshold - dist)
                        reward -= 1.0  # Heavy collision penalty
            # Small positive reward for maintaining safe distance
            if collision_penalty == 0.0:
                reward += 0.1  # Bonus for staying safe

            rewards[agent] = reward
            constraints[agent] = collision_penalty  # > 0 if violated

        # Check termination
        self.timestep += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_steps for agent in self.agents}

        observations = self._get_obs()
        infos = {
            agent: {
                'constraint': constraints[agent],
                'filter_metrics': filter_metrics.get(agent),
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def _get_obs(self) -> Dict[str, np.ndarray]:
        """Get observation for each agent. Uses vision range to limit visible neighbors."""
        observations = {}
        for agent in self.agents:
            idx = self._agent_ids[agent]
            pos = self.agent_positions[idx]
            vel = self.agent_velocities[idx]

            # Self: pos(2) + vel(2)
            self_obs = np.concatenate([pos, vel])

            # Vision: for each OTHER agent, if within vision_range, store [rel_x, rel_y, dist, 1.0]
            # Otherwise store [0, 0, 0, 0]
            vision_obs = []
            visible_count = 0
            for j in range(self.n_agents):
                if j == idx:
                    continue
                other_pos = self.agent_positions[j]
                rel = other_pos - pos
                dist = np.linalg.norm(rel)
                if dist <= self.vision_range:
                    # Normalized relative position + distance + presence flag
                    norm_rel = rel / (dist + 1e-6)
                    vision_obs.extend([norm_rel[0], norm_rel[1], dist / (self.vision_range + 1e-6), 1.0])
                    visible_count += 1
                else:
                    vision_obs.extend([0.0, 0.0, 0.0, 0.0])

            # If fewer neighbors in range, remaining slots stay zero
            while len(vision_obs) < self.vision_dim:
                vision_obs.extend([0.0, 0.0, 0.0, 0.0])

            # Landmark observations
            lm_obs = self.landmarks.flatten()

            obs = np.concatenate([self_obs, np.array(vision_obs[:self.vision_dim]), lm_obs])
            observations[agent] = obs.astype(np.float32)
        return observations

    def _action_to_velocity(self, action: int) -> np.ndarray:
        """Convert discrete action to velocity vector."""
        if action == 0:  # up
            return np.array([0, 1])
        elif action == 1:  # down
            return np.array([0, -1])
        elif action == 2:  # left
            return np.array([-1, 0])
        elif action == 3:  # right
            return np.array([1, 0])
        else:  # stay
            return np.array([0, 0])

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class MultiAgentParticleEnvContinuous(MultiAgentParticleEnv):
    """
    Continuous action version: 2D force vector.

    The policy outputs a 2D continuous force vector [-max_force, max_force].
    The environment clips it to valid range and uses it directly as velocity.

    Key differences from discrete version:
    - action_dim = 2 (force vector components)
    - _action_to_velocity returns clipped action directly
    - predict_collision_risk takes continuous vector
    - get_safe_action projects to zero-force when dangerous
    """

    metadata = {"render_modes": [], "name": "multiagent_particle_continuous_v0"}

    def __init__(
        self,
        n_agents: int = 4,
        max_force: float = 1.0,
        **kwargs,
    ):
        super().__init__(n_agents=n_agents, **kwargs)
        self.max_force = max_force
        self.action_dim = 2  # 2D continuous force vector

        # Override action spaces with continuous Box
        self.action_spaces = {
            agent: spaces.Box(
                low=-max_force, high=max_force, shape=(2,), dtype=np.float32
            )
            for agent in self.agents
        }

        self._agent_ids = {agent: i for i, agent in enumerate(self.agents)}

    def _action_to_velocity(self, action) -> np.ndarray:
        """Continuous action is already a velocity vector (force)."""
        a = np.asarray(action, dtype=np.float32)
        return np.clip(a, -self.max_force, self.max_force)

    def predict_collision_risk(self, agent_idx: int, action: np.ndarray) -> float:
        """
        Continuous collision risk: minimum distance after taking force action.

        The action is a 2D force vector. We simulate stepping with it
        and return the minimum distance to any other agent.
        """
        vel = np.clip(action, -self.max_force, self.max_force)
        next_pos = self.agent_positions[agent_idx] + vel * 0.1
        min_dist = float('inf')
        for j in range(self.n_agents):
            if j != agent_idx:
                dist = np.linalg.norm(next_pos - self.agent_positions[j])
                min_dist = min(min_dist, dist)
        return min_dist

    def get_safe_action(self, agent_idx: int, policy_action: np.ndarray) -> np.ndarray:
        """
        Project policy action to safe region if needed.

        If the policy action would cause a collision, project onto the
        zero-force vector (most conservative safe action).
        """
        safe_action = np.asarray(policy_action, dtype=np.float32)
        if self.predict_collision_risk(agent_idx, safe_action) < self.collision_threshold:
            return np.array([0.0, 0.0], dtype=np.float32)
        return safe_action

    def step(
        self,
        actions: Dict[str, np.ndarray],
        use_safety_filter: bool = False,
        policy_actions: Optional[Dict[str, np.ndarray]] = None,
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, float], Dict[str, bool], Dict[str, bool], Dict[str, dict]]:
        """
        Step with continuous actions.

        actions: dict of agent -> 2D force vectors (float32)
        """
        if not actions:
            raise ValueError("Actions dictionary is empty")

        # Apply safety filter if enabled
        if use_safety_filter and policy_actions:
            safe_actions = {}
            for agent in actions:
                idx = self._agent_ids[agent]
                safe_actions[agent] = self.get_safe_action(idx, policy_actions[agent])
            actions = safe_actions

        # Apply actions as velocity
        velocities = np.zeros((self.n_agents, 2), dtype=np.float32)
        for agent, action in actions.items():
            idx = self._agent_ids[agent]
            velocities[idx] = self._action_to_velocity(action)

        # Update positions
        self.agent_positions = self.agent_positions + velocities * 0.1
        self.agent_velocities = velocities

        # Boundary check
        self.agent_positions = np.clip(
            self.agent_positions,
            -self.boundary_size / 2,
            self.boundary_size / 2,
        )

        # Compute rewards and constraints
        rewards = {}
        constraints = {}
        for agent in self.agents:
            idx = self._agent_ids[agent]
            pos = self.agent_positions[idx]

            dist_to_landmark = np.min([
                np.linalg.norm(pos - lm) for lm in self.landmarks
            ])
            reward = -dist_to_landmark * 0.05
            reward += 0.5  # Step survival bonus

            collision_penalty = 0.0
            for j in range(self.n_agents):
                if j != idx:
                    dist = np.linalg.norm(
                        self.agent_positions[idx] - self.agent_positions[j]
                    )
                    if dist < self.collision_threshold:
                        collision_penalty += (self.collision_threshold - dist)
                        reward -= 1.0  # Heavy collision penalty
            if collision_penalty == 0.0:
                reward += 0.1  # Bonus for staying safe

            rewards[agent] = reward
            constraints[agent] = collision_penalty

        self.timestep += 1
        terminations = {agent: False for agent in self.agents}
        truncations = {agent: self.timestep >= self.max_steps for agent in self.agents}

        observations = self._get_obs()
        infos = {agent: {'constraint': constraints[agent]} for agent in self.agents}

        return observations, rewards, terminations, truncations, infos


class ConstraintTracker:
    """Track constraint violations during training."""

    def __init__(self):
        self.violations = []
        self.episode_violations = []

    def add(self, constraints: Dict[str, float]):
        """Add constraint values for current step."""
        total_violation = sum(max(0, c) for c in constraints.values())
        self.violations.append(total_violation)
        self.episode_violations.append(total_violation)

    def reset_episode(self):
        """Reset for new episode."""
        self.episode_violations = []

    def get_stats(self) -> Dict[str, float]:
        """Get constraint statistics."""
        if not self.violations:
            return {
                'mean_violation': 0.0,
                'max_violation': 0.0,
                'violation_rate': 0.0,
                'episode_mean_violation': 0.0,
            }
        return {
            'mean_violation': np.mean(self.violations),
            'max_violation': np.max(self.violations),
            'violation_rate': np.mean([v > 0 for v in self.violations]),
            'episode_mean_violation': np.mean(self.episode_violations) if self.episode_violations else 0,
        }


# =============================================================================
# Multi-Robot Warehouse Environment
# =============================================================================


class MultiRobotWarehouseEnv(ParallelEnv):
    """
    Multi-robot warehouse environment for safe MARL.

    Scenario: N robots navigate a warehouse floor to pick items from shelves.
    Each robot must avoid collisions with other robots while maximizing throughput.

    Safety Constraints:
    1. Robot-robot collision avoidance (primary)
    2. Robot-shelf collision avoidance
    3. Boundary constraints

    Communication: k-hop sparse (robots only communicate with neighbors)

    Key difference from MPE:
    - Grid layout with aisles and shelves
    - Target locations (shelves) are stationary
    - Collaborative task: maximize total items picked
    - More realistic for industrial applications (Amazon Kiva-style)

    Reference: Inspired by Amazon Kiva/Scott fulfillment robots
    """

    metadata = {"render_modes": [], "name": "warehouse_v0"}

    @staticmethod
    def compute_warehouse_size(n_agents: int, base_size: float = 10.0) -> float:
        """Compute warehouse size to maintain constant robot density."""
        # Warehouse area scales linearly with n_agents
        return base_size * math.sqrt(n_agents / 4)

    @staticmethod
    def compute_layout_extent(
        n_shelves: int,
        shelf_size: float,
        aisle_width: float,
        dock_depth: float,
        outer_margin: float,
    ) -> float:
        """Minimum side length needed for a shelf grid plus a shared dock area."""
        n_cols = max(1, int(math.sqrt(n_shelves)))
        n_rows = (n_shelves + n_cols - 1) // n_cols
        width = n_cols * shelf_size + max(0, n_cols - 1) * aisle_width + 2 * outer_margin
        height = (
            n_rows * shelf_size
            + max(0, n_rows - 1) * aisle_width
            + 2 * outer_margin
            + dock_depth
        )
        return max(width, height)

    def __init__(
        self,
        n_agents: int = 10,
        n_shelves: int = 8,
        max_steps: int = 100,
        agent_size: float = 0.4,
        collision_threshold: float = 0.8,
        warehouse_size: float = None,
        shelf_size: float = 1.5,
        aisle_width: float = 2.5,
        use_structured_layout: bool = False,
        use_shelf_obstacles: bool = False,
        spawn_mode: str = "uniform",
        dock_depth: float = 2.0,
        pickup_radius: float = 0.0,
        outer_margin: float = 1.2,
        seed: int = 42,
    ):
        """
        Args:
            n_agents: Number of robots
            n_shelves: Number of shelf units (items to pick)
            max_steps: Maximum steps per episode
            agent_size: Robot radius (m)
            collision_threshold: Minimum safe distance between robots
            warehouse_size: Warehouse floor size (auto-computed if None)
            shelf_size: Size of each shelf unit
            aisle_width: Width of aisles between shelves
            use_structured_layout: If True, arrange shelves as blocking aisle cells
            use_shelf_obstacles: If True, shelves block motion
            spawn_mode: "uniform" or "dock"
            dock_depth: Height of the shared start area near the bottom boundary
            pickup_radius: Clearance from a shelf face needed to pick an item
            outer_margin: Free margin between outer boundary and shelf blocks
            seed: Random seed
        """
        super().__init__()

        self.n_agents = n_agents
        self.n_shelves = n_shelves
        self.max_steps = max_steps
        self.agent_size = agent_size
        self.collision_threshold = collision_threshold
        self.shelf_size = shelf_size
        self.aisle_width = aisle_width
        self.use_structured_layout = use_structured_layout
        self.use_shelf_obstacles = use_shelf_obstacles
        self.spawn_mode = spawn_mode
        self.dock_depth = dock_depth
        self.pickup_radius = pickup_radius
        self.outer_margin = outer_margin
        self.seed = seed
        default_size = self.compute_warehouse_size(n_agents)
        if self.use_structured_layout:
            default_size = max(
                default_size,
                self.compute_layout_extent(
                    n_shelves=n_shelves,
                    shelf_size=shelf_size,
                    aisle_width=aisle_width,
                    dock_depth=dock_depth,
                    outer_margin=outer_margin,
                ),
            )
        self.warehouse_size = warehouse_size or default_size

        # Safety threshold: robots must maintain this distance
        self.safety_margin = collision_threshold

        # Robot configuration
        self.possible_agents = [f"robot_{i}" for i in range(n_agents)]
        self.agents = self.possible_agents.copy()

        # Observation: robot_pos(2) + robot_vel(2) + shelf_obs + neighbor_obs
        # 2D position and velocity
        self.robot_pos_dim = 2
        self.robot_vel_dim = 2

        # Neighbor awareness: k-nearest neighbors info (up to 3 neighbors)
        self.max_neighbors = 3
        self.neighbor_obs_dim = self.max_neighbors * 4  # rel_x, rel_y, dist, v_rel for each

        # Shelf awareness: nearest shelf info
        self.shelf_obs_dim = 4  # rel_x, rel_y, dist_to_pick, picked (if any)

        self.obs_dim = self.robot_pos_dim + self.robot_vel_dim + self.neighbor_obs_dim + self.shelf_obs_dim
        self.action_dim = 5  # up, down, left, right, stay

        self.observation_spaces = {
            agent: spaces.Box(low=-50, high=50, shape=(self.obs_dim,), dtype=np.float32)
            for agent in self.agents
        }
        self.action_spaces = {
            agent: spaces.Discrete(self.action_dim)
            for agent in self.agents
        }

        self._agent_ids = {agent: i for i, agent in enumerate(self.agents)}

    def _create_shelf_layout(self) -> list:
        """Create shelf positions in a grid layout."""
        shelves = []
        n_cols = int(math.sqrt(self.n_shelves))
        n_rows = (self.n_shelves + n_cols - 1) // n_cols
        if self.use_structured_layout:
            grid_width = n_cols * self.shelf_size + max(0, n_cols - 1) * self.aisle_width
            grid_height = n_rows * self.shelf_size + max(0, n_rows - 1) * self.aisle_width
            x_start = -grid_width / 2 + self.shelf_size / 2
            y_start = (
                -self.warehouse_size / 2
                + self.outer_margin
                + self.dock_depth
                + self.shelf_size / 2
            )
            for i in range(self.n_shelves):
                col = i % n_cols
                row = i // n_cols
                x = x_start + col * (self.shelf_size + self.aisle_width)
                y = y_start + row * (self.shelf_size + self.aisle_width)
                shelves.append(np.array([x, y], dtype=np.float32))
            return shelves

        shelf_spacing_x = (self.warehouse_size - 2) / (n_cols + 1)
        shelf_spacing_y = (self.warehouse_size - 2) / (n_rows + 1)

        for i in range(self.n_shelves):
            col = i % n_cols
            row = i // n_cols
            x = -self.warehouse_size/2 + shelf_spacing_x * (col + 1)
            y = -self.warehouse_size/2 + shelf_spacing_y * (row + 1)
            shelves.append(np.array([x, y], dtype=np.float32))

        return shelves

    def _shelf_clearance(self, pos: np.ndarray, shelf: np.ndarray) -> float:
        """Euclidean clearance from a point to the face of a square shelf obstacle."""
        dx = max(abs(float(pos[0] - shelf[0])) - self.shelf_size / 2, 0.0)
        dy = max(abs(float(pos[1] - shelf[1])) - self.shelf_size / 2, 0.0)
        return math.hypot(dx, dy)

    def _inside_shelf_obstacle(self, pos: np.ndarray, padding: float = 0.0) -> bool:
        if not self.use_shelf_obstacles:
            return False
        threshold = self.shelf_size / 2 + padding
        for shelf in self.shelves:
            if abs(float(pos[0] - shelf[0])) <= threshold and abs(float(pos[1] - shelf[1])) <= threshold:
                return True
        return False

    def _sample_spawn_position(self) -> np.ndarray:
        half_size = self.warehouse_size / 2 - self.agent_size
        if self.spawn_mode == "dock":
            y_low = -half_size
            y_high = min(
                half_size,
                -self.warehouse_size / 2 + self.outer_margin + self.dock_depth,
            )
            return np.array(
                [
                    np.random.uniform(-half_size, half_size),
                    np.random.uniform(y_low, y_high),
                ],
                dtype=np.float32,
            )
        return np.random.uniform(-half_size, half_size, size=2).astype(np.float32)

    def reset(self, seed: int = None) -> tuple:
        """Reset environment to initial state."""
        if seed is not None:
            self.seed = seed
        np.random.seed(self.seed)

        self.agents = self.possible_agents.copy()
        self.timestep = 0

        # Create shelf layout
        self.shelves = self._create_shelf_layout()
        self.shelf_picked = [False] * self.n_shelves

        # Initialize robot positions (spread out to avoid initial collisions)
        self.robot_positions = np.zeros((self.n_agents, 2), dtype=np.float32)
        self.robot_velocities = np.zeros((self.n_agents, 2), dtype=np.float32)

        for i in range(self.n_agents):
            for attempt in range(100):
                pos = self._sample_spawn_position()
                # Check collision with existing robots
                min_dist = float('inf')
                for j in range(i):
                    dist = np.linalg.norm(pos - self.robot_positions[j])
                    min_dist = min(min_dist, dist)
                # Check collision with shelves
                for shelf in self.shelves:
                    dist = self._shelf_clearance(pos, shelf)
                    min_dist = min(min_dist, dist)
                if min_dist >= self.collision_threshold and not self._inside_shelf_obstacle(pos, padding=self.agent_size):
                    self.robot_positions[i] = pos
                    break
            else:
                self.robot_positions[i] = pos

        observations = self._get_obs()
        infos = {agent: {} for agent in self.agents}

        return observations, infos

    def _get_obs(self) -> dict:
        """Get observations for all agents."""
        observations = {}
        for agent in self.agents:
            idx = self._agent_ids[agent]
            obs = self._get_single_obs(idx)
            observations[agent] = obs
        return observations

    def _get_single_obs(self, robot_idx: int) -> np.ndarray:
        """Get observation for a single robot.

        Observation components:
        1. Self position (2D)
        2. Self velocity (2D)
        3. Nearest neighbors (up to 3) relative info
        4. Nearest unpicked shelf info
        """
        obs_parts = []

        # 1. Self position (normalized)
        pos = self.robot_positions[robot_idx]
        obs_parts.append(pos / self.warehouse_size)

        # 2. Self velocity
        vel = self.robot_velocities[robot_idx]
        obs_parts.append(vel)

        # 3. Nearest neighbor observations
        neighbor_dists = []
        for j in range(self.n_agents):
            if j != robot_idx:
                dist = np.linalg.norm(self.robot_positions[robot_idx] - self.robot_positions[j])
                neighbor_dists.append((j, dist))

        # Sort by distance and take nearest 3
        neighbor_dists.sort(key=lambda x: x[1])
        nearest_neighbors = neighbor_dists[:self.max_neighbors]

        for j, dist in nearest_neighbors:
            rel_pos = self.robot_positions[j] - self.robot_positions[robot_idx]
            rel_vel = self.robot_velocities[j] - self.robot_velocities[robot_idx]
            obs_parts.append(rel_pos / self.warehouse_size)  # 2
            obs_parts.append(rel_vel / 2.0)  # 2

        # Pad if fewer than max_neighbors
        while len(nearest_neighbors) < self.max_neighbors:
            obs_parts.append(np.zeros(4, dtype=np.float32))  # 4
            nearest_neighbors.append(None)

        # 4. Nearest unpicked shelf info
        nearest_shelf = None
        nearest_shelf_dist = float('inf')
        for i, shelf in enumerate(self.shelves):
            if not self.shelf_picked[i]:
                dist = np.linalg.norm(self.robot_positions[robot_idx] - shelf)
                if dist < nearest_shelf_dist:
                    nearest_shelf_dist = dist
                    nearest_shelf = (i, shelf)

        if nearest_shelf is not None:
            shelf_idx, shelf_pos = nearest_shelf
            rel_pos = shelf_pos - self.robot_positions[robot_idx]
            obs_parts.append(rel_pos / self.warehouse_size)  # 2
            obs_parts.append(np.array([nearest_shelf_dist / self.warehouse_size, 0.0], dtype=np.float32))  # dist, picked=0
        else:
            # All shelves picked
            obs_parts.append(np.zeros(4, dtype=np.float32))

        return np.concatenate(obs_parts).astype(np.float32)

    def _action_to_velocity(self, action: int) -> np.ndarray:
        """Convert discrete action to 2D velocity vector."""
        dt = 0.1  # Time step
        if action == 0:  # up
            return np.array([0.0, 1.0 * dt], dtype=np.float32)
        elif action == 1:  # down
            return np.array([0.0, -1.0 * dt], dtype=np.float32)
        elif action == 2:  # left
            return np.array([-1.0 * dt, 0.0], dtype=np.float32)
        elif action == 3:  # right
            return np.array([1.0 * dt, 0.0], dtype=np.float32)
        else:  # stay
            return np.array([0.0, 0.0], dtype=np.float32)

    def _check_shelf_collision(self, pos: np.ndarray) -> int:
        """Check if robot is at a shelf (can pick item). Returns shelf index or -1."""
        for i, shelf in enumerate(self.shelves):
            if self.use_shelf_obstacles:
                dist = self._shelf_clearance(pos, shelf)
                if dist <= self.pickup_radius + 1e-6:
                    return i
                continue
            dist = np.linalg.norm(pos - shelf)
            if dist < self.shelf_size / 2 + self.agent_size:
                return i
        return -1

    def step(
        self,
        actions: dict
    ) -> tuple:
        """Execute one step in the environment.

        Args:
            actions: dict of agent -> action index

        Returns:
            observations, rewards, terminations, truncations, infos
        """
        if not actions:
            raise ValueError("Actions dictionary is empty")

        self.timestep += 1

        # Apply actions
        new_positions = np.copy(self.robot_positions)
        new_velocities = np.copy(self.robot_velocities)

        for agent, action in actions.items():
            idx = self._agent_ids[agent]
            vel = self._action_to_velocity(action)
            new_positions[idx] += vel
            new_velocities[idx] = vel / 0.1 if 0.1 > 0 else vel

        # Boundary enforcement
        half_size = self.warehouse_size / 2 - self.agent_size
        new_positions = np.clip(
            new_positions,
            -half_size,
            half_size
        )

        shelf_contact = np.zeros(self.n_agents, dtype=bool)
        if self.use_shelf_obstacles:
            for i in range(self.n_agents):
                if self._inside_shelf_obstacle(new_positions[i], padding=self.agent_size):
                    new_positions[i] = self.robot_positions[i]
                    new_velocities[i] = 0.0
                    shelf_contact[i] = True

        # Collision detection and resolution
        # First: check robot-robot collisions
        collision_detected = np.zeros(self.n_agents, dtype=bool)
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                dist = np.linalg.norm(new_positions[i] - new_positions[j])
                if dist < self.collision_threshold:
                    # Push robots apart
                    overlap = self.collision_threshold - dist
                    direction = (new_positions[i] - new_positions[j]) / (dist + 1e-8)
                    new_positions[i] += direction * overlap / 2
                    new_positions[j] -= direction * overlap / 2
                    collision_detected[i] = True
                    collision_detected[j] = True

        self.robot_positions = new_positions
        self.robot_velocities = new_velocities

        # Compute rewards and constraints
        rewards = {}
        constraints = {}

        for agent in self.agents:
            idx = self._agent_ids[agent]
            pos = self.robot_positions[idx]

            # Check shelf pick
            shelf_idx = self._check_shelf_collision(pos)
            item_picked = False
            if shelf_idx >= 0 and not self.shelf_picked[shelf_idx]:
                self.shelf_picked[shelf_idx] = True
                item_picked = True

            # Base reward: negative distance to nearest unpicked shelf
            min_dist = float('inf')
            for i, shelf in enumerate(self.shelves):
                if not self.shelf_picked[i]:
                    dist = np.linalg.norm(pos - shelf)
                    min_dist = min(min_dist, dist)
            if min_dist == float('inf'):
                min_dist = 0.0

            reward = -min_dist * 0.05

            if item_picked:
                reward += 5.0  # Big reward for picking item

            # Collision penalty
            constraint_violation = 0.0
            if collision_detected[idx]:
                reward -= 1.0  # Collision penalty
                constraint_violation = 1.0

            # Boundary penalty
            if (abs(pos[0]) >= half_size or abs(pos[1]) >= half_size):
                reward -= 0.5
            if shelf_contact[idx]:
                reward -= 0.25

            rewards[agent] = reward
            constraints[agent] = constraint_violation

        # Check termination
        terminations = {agent: False for agent in self.agents}
        truncations = {
            agent: self.timestep >= self.max_steps
            for agent in self.agents
        }

        # All shelves picked = success
        if all(self.shelf_picked):
            # Bonus for completing all picks
            for agent in self.agents:
                rewards[agent] += 10.0
                terminations[agent] = True

        observations = self._get_obs()
        infos = {
            agent: {
                'constraint': constraints[agent],
                'items_picked': sum(self.shelf_picked),
                'total_items': self.n_shelves,
                'shelf_contact': bool(shelf_contact[self._agent_ids[agent]]),
            }
            for agent in self.agents
        }

        return observations, rewards, terminations, truncations, infos

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class WarehouseConstraintTracker(ConstraintTracker):
    """Extended constraint tracker for warehouse environment."""

    def __init__(self):
        super().__init__()
        self.items_picked_history = []
        self.total_items = 0

    def reset_episode(self):
        super().reset_episode()
        self.items_picked_history = []

    def add(self, constraints: dict, infos: dict = None):
        super().add(constraints)
        if infos:
            total_picked = sum(i.get('items_picked', 0) for i in infos.values()) // len(infos) if infos else 0
            self.items_picked_history.append(total_picked)

    def get_stats(self) -> dict:
        stats = super().get_stats()
        stats['items_picked'] = sum(self.shelf_picked) if hasattr(self, 'shelf_picked') else 0
        return stats
