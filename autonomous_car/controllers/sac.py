"""Soft Actor-Critic (SAC) Implementation for Autonomous Driving.

SAC is an off-policy actor-critic algorithm that uses:
- Maximum entropy RL: Optimizes J(π) = E[Σ γᵗ(rₜ + α·H(π))]
- Twin Q-networks to reduce overestimation bias
- Automatic temperature tuning for entropy coefficient α
- Reparameterization trick for policy gradient

This is TRUE reinforcement learning that optimizes reward, unlike LWR
which is just supervised imitation learning.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from collections import deque
import random
import json


@dataclass
class SACConfig:
    """SAC hyperparameters."""
    # Network architecture
    hidden_dim: int = 256
    num_hidden_layers: int = 2
    use_layer_norm: bool = True  # LayerNorm for stability
    
    # Training
    learning_rate: float = 3e-4
    # Stability sweep: separate critic (value-head) LR. The value head is the
    # half that re-fits to the drifting replay mix and loses its hazard-gradient;
    # a slower q_lr slows that re-fit directly. None = use learning_rate.
    q_lr: Optional[float] = None
    alpha_lr: float = 1e-4  # Slower alpha learning for stability
    gamma: float = 0.99
    tau: float = 0.005  # Soft update coefficient
    batch_size: int = 256
    buffer_size: int = 1_000_000
    grad_clip: float = 1.0  # Gradient clipping for stability
    
    # Entropy
    init_alpha: float = 0.3   # V15: Fixed alpha (was 0.2)
    auto_tune_alpha: bool = False  # V15: Disabled — auto-tune causes alpha collapse
    target_entropy: Optional[float] = None  # Used if auto_tune is True
    # Stability (v21-v21b post-peak degradation fix): deterministic anneal of
    # exploration temperature and learning rate so a good policy settles in
    # instead of being perturbed out of its basin late in training.
    alpha_start: Optional[float] = None   # if set: anneal alpha_start -> alpha_end
    alpha_end: float = 0.05
    alpha_anneal_steps: int = 0           # env steps over which alpha anneals; 0 = fixed
    lr_anneal_end_frac: float = 0.3       # final effective LR = lr * this frac
    lr_anneal_steps: Optional[int] = None  # env steps for LR decay; default = alpha_anneal_steps
    
    # Training schedule
    warmup_steps: int = 10000  # Random actions before training
    update_every: int = 1
    gradient_steps: int = 1

    # Behavior cloning (BC-SAC imitation term)
    bc_coef: float = 0.1           # lambda: strength of BC term in policy loss
    bc_demo_fraction: float = 0.5  # Reserved: fraction of demo transitions per update
    bc_coef_start: float = 1.0     # lambda at env_steps=0 (curriculum anneal start)
    bc_anneal_steps: int = 0       # dist over which lambda anneals start -> bc_coef; 0 = fixed
    bc_coef_floor: float = 0.5     # Never let the BC anchor fall below this value
    bc_update_interval: int = 8    # One separate IL update after N RL updates
    bc_learning_rate: float = 5e-5
    q_scale_floor: float = 1.0     # Prevent near-zero Q from amplifying RL gradients
    # Hazard-margin value anchoring (root-cause fix, Aug 29): an auxiliary
    # state-only safety-value head is regressed on a FIXED proximity target so
    # the actor always receives a danger signal even if the main critic's
    # hazard-gradient washes out under off-policy drift.
    safety_value_coef: float = 0.5   # weight of safety head in the actor objective
    q_clip_actor: float = 5.0        # clamp |normalized Q| in the actor objective
    safety_near_dist: float = 15.0   # hazard within this many meters is 'dangerous'


class ReplayBuffer:
    """Experience replay buffer for off-policy learning."""
    
    def __init__(self, capacity: int, obs_dim: int, action_dim: int):
        self.capacity = capacity
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Pre-allocate arrays for efficiency
        self.observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_observations = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.idx = 0
        self.size = 0
    
    def push(self, obs: np.ndarray, action: np.ndarray, reward: float,
             next_obs: np.ndarray, done: bool):
        """Add a transition to the buffer."""
        self.observations[self.idx] = obs
        self.actions[self.idx] = action
        self.rewards[self.idx] = reward
        self.next_observations[self.idx] = next_obs
        self.dones[self.idx] = float(done)
        
        self.idx = (self.idx + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """Sample a batch of transitions."""
        indices = np.random.randint(0, self.size, size=batch_size)
        
        return {
            'obs': torch.FloatTensor(self.observations[indices]),
            'actions': torch.FloatTensor(self.actions[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'next_obs': torch.FloatTensor(self.next_observations[indices]),
            'dones': torch.FloatTensor(self.dones[indices])
        }
    
    def __len__(self):
        return self.size


class MLP(nn.Module):
    """Multi-layer perceptron with configurable architecture."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int,
                 num_hidden: int, activation: nn.Module = nn.ReLU,
                 use_layer_norm: bool = True):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for _ in range(num_hidden):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))  # Stability improvement
            layers.append(activation())
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, output_dim))
        self.net = nn.Sequential(*layers)
        
        # Initialize final layer with small weights for stability
        nn.init.uniform_(self.net[-1].weight, -3e-3, 3e-3)
        nn.init.uniform_(self.net[-1].bias, -3e-3, 3e-3)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Stochastic policy that outputs a Gaussian distribution over actions.
    
    Uses tanh squashing to bound actions to [-1, 1] and reparameterization
    for differentiable sampling.
    """
    
    LOG_STD_MIN = -20
    LOG_STD_MAX = 2
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int,
                 num_hidden: int, use_layer_norm: bool = True):
        super().__init__()
        
        # Shared feature extractor with LayerNorm for stability
        self.feature_net = MLP(obs_dim, hidden_dim, hidden_dim, num_hidden - 1,
                               use_layer_norm=use_layer_norm)
        
        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        
        # Initialize output layers with small weights for stability
        nn.init.uniform_(self.mean_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.mean_head.bias, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.log_std_head.bias, -3e-3, 3e-3)
        
        self.action_dim = action_dim
    
    def forward(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get mean and log_std of action distribution."""
        features = F.relu(self.feature_net(obs))
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std
    
    def sample(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action using reparameterization trick.
        
        Returns:
            action: Sampled action (squashed to [-1, 1])
            log_prob: Log probability of the action
        """
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        
        # Reparameterization: action = tanh(mean + std * noise)
        normal = Normal(mean, std)
        x_t = normal.rsample()  # Reparameterized sample
        action = torch.tanh(x_t)
        
        # Compute log probability with tanh correction
        # log π(a|s) = log μ(u|s) - Σ log(1 - tanh²(uᵢ))
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Get action for inference."""
        mean, log_std = self.forward(obs)
        
        if deterministic:
            return torch.tanh(mean)
        else:
            std = log_std.exp()
            normal = Normal(mean, std)
            x_t = normal.rsample()
            return torch.tanh(x_t)


class TwinQNetwork(nn.Module):
    """Twin Q-networks for reducing overestimation bias."""
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int,
                 num_hidden: int, use_layer_norm: bool = True):
        super().__init__()
        
        input_dim = obs_dim + action_dim
        self.q1 = MLP(input_dim, 1, hidden_dim, num_hidden, use_layer_norm=use_layer_norm)
        self.q2 = MLP(input_dim, 1, hidden_dim, num_hidden, use_layer_norm=use_layer_norm)
    
    def forward(self, obs: torch.Tensor, action: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Q-values from both networks."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)
    
    def q1_forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compute Q-value from first network only (for policy optimization)."""
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x)


class SAC:
    """Soft Actor-Critic agent.
    
    Optimizes: J(π) = E[Σ γᵗ(rₜ + α·H(π(·|sₜ)))]
    
    Where α is the temperature parameter that balances reward vs entropy.
    """
    
    def __init__(self, obs_dim: int, action_dim: int, config: SACConfig,
                 action_scale: np.ndarray, action_bias: np.ndarray,
                 device: str = 'cpu'):
        """Initialize SAC agent.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            config: SAC hyperparameters
            action_scale: Scale to convert [-1,1] to actual action range
            action_bias: Bias to convert [-1,1] to actual action range
            device: 'cpu' or 'cuda'
        """
        self.config = config
        self.device = torch.device(device)
        self.action_dim = action_dim
        
        # Action scaling: actual_action = scale * tanh_action + bias
        self.action_scale = torch.FloatTensor(action_scale).to(self.device)
        self.action_bias = torch.FloatTensor(action_bias).to(self.device)
        
        # Stability parameters
        self.grad_clip = config.grad_clip
        
        # Networks with LayerNorm for stability
        self.policy = GaussianPolicy(
            obs_dim, action_dim, config.hidden_dim, config.num_hidden_layers,
            use_layer_norm=config.use_layer_norm
        ).to(self.device)
        
        self.q_network = TwinQNetwork(
            obs_dim, action_dim, config.hidden_dim, config.num_hidden_layers,
            use_layer_norm=config.use_layer_norm
        ).to(self.device)
        
        self.q_target = TwinQNetwork(
            obs_dim, action_dim, config.hidden_dim, config.num_hidden_layers,
            use_layer_norm=config.use_layer_norm
        ).to(self.device)
        self.q_target.load_state_dict(self.q_network.state_dict())

        # Hazard-margin value anchor (root-cause fix): state-only net mapping
        # observation -> signed proximity affinity (0 = clear, -1 = contact),
        # trained by regression (NOT Bellman), so it can never un-learn danger.
        self.safety_head = MLP(
            obs_dim, 1, config.hidden_dim, config.num_hidden_layers,
            use_layer_norm=config.use_layer_norm,
        ).to(self.device)
        self.safety_optimizer = torch.optim.Adam(
            self.safety_head.parameters(), lr=config.learning_rate
        )
        
        # Optimizers
        self.policy_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.learning_rate
        )
        self.bc_optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=config.bc_learning_rate
        )
        self.q_optimizer = torch.optim.Adam(
            self.q_network.parameters(),
            lr=config.q_lr if config.q_lr is not None else config.learning_rate,
        )
        
        # Entropy temperature (alpha) with slower learning rate for stability
        self.auto_tune_alpha = config.auto_tune_alpha
        # target_entropy only needed for auto-tune (disabled in V15+)
        if self.auto_tune_alpha:
            if config.target_entropy is None:
                self.target_entropy = -action_dim
            else:
                self.target_entropy = config.target_entropy
        
        if self.auto_tune_alpha:
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            # Use slower learning rate for alpha to prevent oscillation
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=config.alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        elif config.alpha_start is not None and config.alpha_anneal_steps > 0:
            self.alpha = config.alpha_start
        else:
            self.alpha = config.init_alpha
        self._alpha_start = config.alpha_start if config.alpha_start is not None else None
        self._alpha_anneal_steps = config.alpha_anneal_steps
        
        # Replay buffer
        self.buffer = ReplayBuffer(config.buffer_size, obs_dim, action_dim)
        
        # Training stats
        self.total_steps = 0
        # env_steps: counter in env-transition units (NOT decisions). Synced by the
        # train_sac loop each decision; used for BC coef curriculum + resume.
        self.env_steps = 0
        
        # Behavior-cloning demonstrations (obs, actions in NORMALIZED [-1,1] space)
        self.demo_obs = None
        self.demo_actions = None
        self.update_count = 0
    
    def select_action(self, obs: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """Select action given observation.
        
        Args:
            obs: Observation array
            deterministic: If True, use mean action (no exploration)
        
        Returns:
            action: Action array in actual action space
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if self.total_steps < self.config.warmup_steps and not deterministic:
                # Random exploration during warmup
                action = torch.FloatTensor(1, self.action_dim).uniform_(-1, 1).to(self.device)
            else:
                action = self.policy.get_action(obs_tensor, deterministic)
            
            # Scale to actual action space
            actual_action = action * self.action_scale + self.action_bias
            
            return actual_action.cpu().numpy()[0]
    
    def sync_schedule(self, env_steps: int) -> None:
        """Deterministic alpha anneal + LR decay based on live env_steps.

        Called each env step so exploration temperature eases off once the
        policy has found a good basin, and learning rate shrinks to settle
        instead of overshooting past the peak (v21/v21b degradation fix).
        Does nothing unless alpha_start / anneal configured.
        """
        if (self.config.alpha_start is not None
                and self.config.alpha_anneal_steps > 0):
            t = min(1.0, env_steps / max(1, self.config.alpha_anneal_steps))
            self.alpha = (
                self.config.alpha_start
                + (self.config.alpha_end - self.config.alpha_start) * t
            )
            self._last_alpha = self.alpha

        # LR decay: cosine from lr -> lr*lr_anneal_end_frac over the SAME window
        # (or over alpha_anneal_steps if set, else over a fixed horizon).
        decay_steps = self.config.lr_anneal_steps
        if decay_steps is None:
            decay_steps = self.config.alpha_anneal_steps
        if decay_steps and decay_steps > 0:
            t = min(1.0, env_steps / decay_steps)
            # cosine schedule
            cos = 0.5 * (1.0 + np.cos(np.pi * t))
            frac = self.config.lr_anneal_end_frac + (1.0 - self.config.lr_anneal_end_frac) * cos
            lr = self.config.learning_rate * frac
            for opt in (self.policy_optimizer, self.q_optimizer):
                for g in opt.param_groups:
                    g['lr'] = lr

    def _current_bc_coef(self) -> float:
        """Return the BC lambda for the current env_steps.

        Linear curriculum anneal: bc_coef_start -> bc_coef over bc_anneal_steps
        env-transition units. bc_anneal_steps=0 -> fixed lambda = bc_coef.
        """
        if self.config.bc_anneal_steps <= 0:
            return max(self.config.bc_coef, getattr(self.config, 'bc_coef_floor', 0.5))
        t = min(1.0, self.env_steps / self.config.bc_anneal_steps)
        lam = self.config.bc_coef_start + (self.config.bc_coef - self.config.bc_coef_start) * t
        return max(lam, getattr(self.config, 'bc_coef_floor', 0.5))
    
    def set_demos(self, obs: np.ndarray, actions: np.ndarray,
                  action_scale: np.ndarray = None, action_bias: np.ndarray = None,
                  completed_only: bool = False,
                  episode_ids: np.ndarray = None,
                  episode_completed: np.ndarray = None):
        """Register BC demonstrations.

        Args:
            obs: Demo observations [N, obs_dim]
            actions: Demo actions in ACTUAL action space [N, action_dim]
            action_scale/bias: If given, normalize actions to [-1,1] (else assume already normalized)
            completed_only: If True, keep only transitions whose episode completed.
            episode_ids: per-transition episode index [N] (int)
            episode_completed: per-episode completion flag [num_episodes] (bool)
        """
        mask = np.ones(len(actions), dtype=bool)
        if completed_only:
            if episode_ids is None or episode_completed is None:
                raise ValueError("completed_only=True requires episode_ids and episode_completed")
            mask = episode_completed[episode_ids]
        self.demo_obs = obs[mask].astype(np.float32)
        demo_act = actions[mask].astype(np.float32)
        if action_scale is not None and action_bias is not None:
            demo_act = (demo_act - action_bias) / action_scale
        self.demo_actions = demo_act.astype(np.float32)
        print(f"[BC] Registered {self.demo_obs.shape[0]:,} completed-only demos "
              f"(action normalized to [-1,1], range [{demo_act.min():.2f},{demo_act.max():.2f}])")

    def _safety_target(self, obs: torch.Tensor) -> torch.Tensor:
        """Fixed proximity affinity target from the obs distance-sensor block.

        Sensors occupy obs[7:12], normalized to [0..1] over sensor_range (default
        50 m; near = 0, far = 1). Affinity = -1 when the nearest sensor reads
        contact, rising to 0 when clear beyond safety_near_dist. This is a
        SUPERVISED target, detached from the Bellman loop, so the actor always
        sees a stable danger gradient even if the main critic's hazard-signal
        washes out under off-policy drift (root-cause finding, Aug 29).
        """
        near = getattr(self.config, 'safety_near_dist', 15.0)
        sensor_range = 50.0
        sensors = obs[:, 7:12].clamp(0.0, 1.0)
        nearest = sensors.min(dim=-1, keepdim=True).values  # 0 close .. 1 far
        clearance_m = nearest * sensor_range
        danger = (1.0 - (clearance_m / near)).clamp(0.0, 1.0)  # 0 clear .. 1 at contact
        return -danger  # -1 dangerous, 0 clear

    def _safety_loss(self, obs: torch.Tensor) -> torch.Tensor:
        """Regression loss pinning the safety head to the fixed target."""
        target = self._safety_target(obs)
        return torch.nn.functional.mse_loss(self.safety_head(obs), target)

    def _bc_loss(self, demo_obs: torch.Tensor, demo_actions: torch.Tensor) -> torch.Tensor:
        """Compute BC loss in the latent space of the tanh Gaussian policy."""
        demo_mean, demo_log_std = self.policy.forward(demo_obs)
        demo_std = demo_log_std.exp()
        demo_dist = Normal(demo_mean, demo_std)
        bounded_actions = torch.clamp(demo_actions, -1 + 1e-6, 1 - 1e-6)
        latent_actions = torch.atanh(bounded_actions)
        log_prob = demo_dist.log_prob(latent_actions).sum(dim=-1, keepdim=True)
        return -log_prob.mean()
    
    def update(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Perform one gradient update step.
        
        Args:
            batch: Dictionary with 'obs', 'actions', 'rewards', 'next_obs', 'dones'
        
        Returns:
            Dictionary of training metrics
        """
        obs = batch['obs'].to(self.device)
        actions = batch['actions'].to(self.device)
        rewards = batch['rewards'].unsqueeze(-1).to(self.device)
        next_obs = batch['next_obs'].to(self.device)
        dones = batch['dones'].unsqueeze(-1).to(self.device)
        
        # Convert actions from actual space to [-1, 1] for Q-network
        normalized_actions = (actions - self.action_bias) / self.action_scale
        
        # --- Update Q-networks ---
        with torch.no_grad():
            # Sample next actions and their log probs
            next_actions, next_log_probs = self.policy.sample(next_obs)
            
            # Compute target Q-values using target networks
            target_q1, target_q2 = self.q_target(next_obs, next_actions)
            target_q = torch.min(target_q1, target_q2)
            
            # Bellman backup with entropy bonus
            # y = r + γ(1-d)(Q_target - α·log_prob)
            target_value = target_q - self.alpha * next_log_probs
            q_target = rewards + self.config.gamma * (1 - dones) * target_value
        
        # Current Q estimates
        q1, q2 = self.q_network(obs, normalized_actions)
        
        # Q-network loss (MSE)
        q1_loss = F.mse_loss(q1, q_target)
        q2_loss = F.mse_loss(q2, q_target)
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), self.grad_clip)
        self.q_optimizer.step()
        
        # --- Update policy ---
        # Sample new actions for current observations
        new_actions, log_probs = self.policy.sample(obs)
        
        # Q-value of new actions (use Q1 only to reduce computation)
        q1_new = self.q_network.q1_forward(obs, new_actions)
        
        # Normalize only Q, and floor the scale. Dividing the entire actor
        # objective by a near-zero Q estimate amplified an uninformative critic.
        q_scale = max(
            q1_new.abs().mean().detach().item(),
            getattr(self.config, 'q_scale_floor', 1.0),
        )
        # Hazard-margin anchor: fold the state-only safety affinity into the
        # actor's maximized value so braking-near-danger stays incentivized even
        # if q1_new's hazard-gradient recedes.
        # Bound the normalized Q so a value-head explosion cannot drown the
        # (bounded) safety anchor: scale by the FLOORED mean-abs (as before) then
        # clamp to [-q_clip_actor, +q_clip_actor]. Keeps risk/avoid ordering
        # while preventing the degraded critic's growing magnitude from winning.
        q_norm = (q1_new / q_scale).clamp(
            -getattr(self.config, 'q_clip_actor', 5.0),
            getattr(self.config, 'q_clip_actor', 5.0),
        )
        rl_policy_loss = self.alpha * log_probs - q_norm
        rl_policy_loss = rl_policy_loss - self.config.safety_value_coef * \
            self.safety_head(obs).squeeze(-1).detach()
        policy_loss = rl_policy_loss.mean()

        # Supervised safety-head regression (every update) keeps the anchor
        # pinned to the fixed proximity target so it cannot un-learn danger.
        safety_loss = self._safety_loss(obs)
        self.safety_optimizer.zero_grad()
        safety_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.safety_head.parameters(), self.grad_clip)
        self.safety_optimizer.step()

        self.update_count += 1
        bc_metrics = {'bc_coef': 0.0, 'bc_loss': 0.0}
        if (self.demo_obs is not None and
                self.update_count % max(1, getattr(self.config, 'bc_update_interval', 8)) == 0):
            lam = self._current_bc_coef()
            bc_metrics['bc_coef'] = lam

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
        self.policy_optimizer.step()

        # Keep IL as a separate, lower-rate update as in BC-SAC rather than
        # assuming the RL and BC gradients have compatible scales.
        if self.demo_obs is not None and bc_metrics['bc_coef'] > 0:
            n = min(self.config.batch_size, len(self.demo_obs))
            idx = torch.randint(0, len(self.demo_obs), (n,))
            demo_obs_t = torch.FloatTensor(self.demo_obs[idx]).to(self.device)
            demo_act_t = torch.FloatTensor(self.demo_actions[idx]).to(self.device)
            bc_loss = self._bc_loss(demo_obs_t, demo_act_t)
            bc_metrics['bc_loss'] = bc_loss.item()
            self.bc_optimizer.zero_grad()
            (bc_metrics['bc_coef'] * bc_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.grad_clip)
            self.bc_optimizer.step()
        
        # --- Update temperature (alpha) with bounds ---
        alpha_loss = 0.0
        if self.auto_tune_alpha:
            # α_loss = -E[α · (log π(a|s) + H_target)]
            alpha_loss = -(self.log_alpha.exp() * (log_probs.detach() + self.target_entropy)).mean()
            
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            
            self.alpha = self.log_alpha.exp().item()
        
        # --- Soft update target networks ---
        with torch.no_grad():
            for param, target_param in zip(self.q_network.parameters(),
                                           self.q_target.parameters()):
                target_param.data.copy_(
                    self.config.tau * param.data + (1 - self.config.tau) * target_param.data
                )
        
        return {
            'q_loss': q_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'alpha_loss': alpha_loss if isinstance(alpha_loss, float) else alpha_loss.item(),
            'q1_mean': q1.mean().item(),
            'log_prob_mean': log_probs.mean().item(),
            **bc_metrics,
        }
    
    def train_step(self, obs: np.ndarray, action: np.ndarray, reward: float,
                   next_obs: np.ndarray, done: bool) -> Optional[Dict[str, float]]:
        """Add experience to buffer and optionally update.
        
        Args:
            obs, action, reward, next_obs, done: Transition tuple
        
        Returns:
            Training metrics if update was performed, None otherwise
        """
        # Preserve terminal crash/completion signals. The old [-50, 10] clip
        # collapsed the -200 crash and +200 completion events into ordinary
        # ten-step shaping windows.
        reward = np.clip(reward, -500.0, 500.0)
        
        # Convert action from actual space to [-1, 1] for storage
        normalized_action = (action - self.action_bias.cpu().numpy()) / self.action_scale.cpu().numpy()
        self.buffer.push(obs, normalized_action, reward, next_obs, done)
        self.total_steps += 1
        
        # Only update after warmup and if enough samples
        if (self.total_steps >= self.config.warmup_steps and 
            len(self.buffer) >= self.config.batch_size and
            self.total_steps % self.config.update_every == 0):
            
            metrics = {}
            for _ in range(self.config.gradient_steps):
                batch = self.buffer.sample(self.config.batch_size)
                update_metrics = self.update(batch)
                metrics = update_metrics  # Return last update metrics
            
            return metrics
        
        return None
    
    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'safety_head_state_dict': self.safety_head.state_dict(),
            'policy_state_dict': self.policy.state_dict(),
            'q_network_state_dict': self.q_network.state_dict(),
            'q_target_state_dict': self.q_target.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'bc_optimizer': self.bc_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict(),
            'log_alpha': self.log_alpha if self.auto_tune_alpha else None,
            'alpha_optimizer': self.alpha_optimizer.state_dict() if self.auto_tune_alpha else None,
            'config': self.config,
            'total_steps': self.total_steps,
            'env_steps': self.env_steps,
            'update_count': self.update_count,
        }, path)
        print(f"Saved SAC model to {path}")
    
    def load(self, path: str):
        """Load model checkpoint."""
        # weights_only=False needed for loading SACConfig dataclass
        checkpoint = torch.load(path, map_location=self.device, weights_only=False)
        
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        if 'safety_head_state_dict' in checkpoint:
            self.safety_head.load_state_dict(checkpoint['safety_head_state_dict'])
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.q_target.load_state_dict(checkpoint['q_target_state_dict'])
        self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])
        if 'bc_optimizer' in checkpoint:
            self.bc_optimizer.load_state_dict(checkpoint['bc_optimizer'])
        self.q_optimizer.load_state_dict(checkpoint['q_optimizer'])
        
        if self.auto_tune_alpha and checkpoint['log_alpha'] is not None:
            self.log_alpha.data = checkpoint['log_alpha'].data
            self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer'])
            self.alpha = self.log_alpha.exp().item()
        
        self.total_steps = checkpoint['total_steps']
        self.env_steps = checkpoint.get('env_steps', self.total_steps)
        self.update_count = checkpoint.get('update_count', 0)
        print(f"Loaded SAC model from {path} (trained {self.total_steps} steps, {self.env_steps} env steps)")


class SACController:
    """Wrapper for using trained SAC policy as a controller."""
    
    def __init__(self, model_path: str, obs_dim: int, action_dim: int,
                 action_scale: np.ndarray, action_bias: np.ndarray,
                 device: str = 'cpu'):
        """Load trained SAC model for inference.
        
        Args:
            model_path: Path to saved SAC checkpoint
            obs_dim: Observation dimension
            action_dim: Action dimension
            action_scale: Action scaling factors
            action_bias: Action bias values
            device: 'cpu' or 'cuda'
        """
        self.device = torch.device(device)
        
        # Load checkpoint (weights_only=False needed for SACConfig dataclass)
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        config = checkpoint['config']
        
        # Detect if model was trained with LayerNorm by checking state_dict keys
        # Without LayerNorm: net.0 (Linear), net.1 (ReLU-no weights), net.2 (Linear)
        # With LayerNorm: net.0 (Linear), net.1 (LayerNorm-has weights), net.2 (ReLU), net.3 (Linear)
        # So if "feature_net.net.1.weight" exists, LayerNorm was used
        policy_state = checkpoint['policy_state_dict']
        use_layer_norm = 'feature_net.net.1.weight' in policy_state
        
        # Create policy network matching saved architecture
        self.policy = GaussianPolicy(
            obs_dim, action_dim, config.hidden_dim, config.num_hidden_layers,
            use_layer_norm=use_layer_norm
        ).to(self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.policy.eval()
        
        # Action scaling
        self.action_scale = torch.FloatTensor(action_scale).to(self.device)
        self.action_bias = torch.FloatTensor(action_bias).to(self.device)
        
        print(f"Loaded SAC policy from {model_path} (LayerNorm: {use_layer_norm})")
    
    def get_action(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        """Get action from policy.
        
        Args:
            obs: Observation array
            deterministic: If True, use mean action
        
        Returns:
            action: Action array in actual action space
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            action = self.policy.get_action(obs_tensor, deterministic)
            actual_action = action * self.action_scale + self.action_bias
            return actual_action.cpu().numpy()[0]
    
    def predict(self, obs: np.ndarray) -> np.ndarray:
        """Alias for get_action with deterministic=True."""
        return self.get_action(obs, deterministic=True)


def instrument_critic(agent: SAC, batch_size: int = 512) -> Optional[Dict]:
    """Measure critic hazard-sensitivity from the replay buffer.

    The replay buffer is the critic's OWN training distribution -- the exact
    data that, per v24-v26 instrumentation, washes out the hazard-gradient.
    Returning values:
      - corr_hazard_q: Pearson corr between hazard (1 - min local sensor
        reading over obs[7:12]) and min(Q1,Q2). HEALTHY = strongly negative
        (near hazard -> low value). Drift toward ~0 is the collapse marker.
      - q_contrast: mean(Q | high-hazard) - mean(Q | low-hazard) via a median
        split; a median-split contrast that is robust where corr is degenerate.
      - corr_hazard_speed: corr between hazard and obs[2] (speed) on the same
        mix; healthy policies brake near hazard (negative), exploitative ones
        flatten toward ~0.
      - q_min/q_max/q_range/q_absmean: value-head magnitude (blowup detector).
    Returns None until the buffer holds >= batch_size valid samples.
    """
    if len(agent.buffer) < min(256, batch_size):
        return None
    batch = agent.buffer.sample(batch_size)
    obs = batch['obs'].to(agent.device)
    actions = batch['actions'].to(agent.device)
    with torch.no_grad():
        q1, q2 = agent.q_network(obs, actions)
        q = torch.min(q1, q2).squeeze(-1)
    q_np = q.cpu().numpy()
    obs_np = obs.cpu().numpy()

    sensors = obs_np[:, 7:12]
    hazard = 1.0 - sensors.min(axis=1)  # 1 => at contact, 0 => clear
    speed = obs_np[:, 2]

    def _pearson(a, b):
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        if a.std() < 1e-9 or b.std() < 1e-9 or len(a) < 8:
            return None
        return float(np.corrcoef(a, b)[0, 1])

    med = float(np.median(hazard))
    hi = hazard > med
    lo = ~hi
    q_contrast = None
    if hi.sum() > 8 and lo.sum() > 8:
        q_contrast = float(q_np[hi].mean() - q_np[lo].mean())

    return {
        'corr_hazard_q': _pearson(hazard, q_np),
        'corr_hazard_speed': _pearson(hazard, speed),
        'q_contrast': q_contrast,
        'q_min': float(q_np.min()),
        'q_max': float(q_np.max()),
        'q_range': float(q_np.max() - q_np.min()),
        'q_absmean': float(np.abs(q_np).mean()),
        'n': int(len(q_np)),
    }


def train_sac(env, config: SACConfig, total_timesteps: int,
              eval_interval: int = 10000, save_interval: int = 50000,
              save_path: str = 'models/sac_model.pt',
              device: str = 'cpu',
              domain_randomization: bool = False,
              traffic_range: tuple = (4, 12),
              obstacles_range: tuple = (4, 8),
              use_decision_layer: bool = False,
              demos_path: Optional[str] = None,
              demos_completed_only: bool = False,
              resume_path: Optional[str] = None,
              early_stop_patience: int = 3,
              resume_max_timesteps: int = 150_000,
              eval_seed: int = 42,
              eval_num_episodes: int = 4,
              eval_n_seeds: int = 3,
              instrument_every: Optional[int] = None,
              instrument_out: Optional[str] = None,
              preset_agent: Optional["SAC"] = None) -> SAC:
    """Train SAC agent on environment.
    
    Args:
        env: Gymnasium environment
        config: SAC hyperparameters
        total_timesteps: Total training steps
        eval_interval: Evaluate every N steps
        save_interval: Save checkpoint every N steps
        save_path: Path to save final model
        device: 'cpu' or 'cuda'
        domain_randomization: Enable random obstacles/traffic per episode
        traffic_range: (min, max) traffic vehicles if randomizing
        obstacles_range: (min, max) obstacles if randomizing
        use_decision_layer: Enable decision layer for strategic decisions
        demos_path: Path to npz with BC demos (keys: observations, actions, episode_ids, episode_completed)
        demos_completed_only: Keep only demos from completed episodes
        resume_path: Load a checkpoint and continue training for total_timesteps MORE env steps
        early_stop_patience: Stop after this many evals without improvement.
        resume_max_timesteps: Maximum continuation budget when resuming.
        eval_seed: First seed for fixed-seed evaluation.
    
    Returns:
        Trained SAC agent
    """
    # Import decision layer if needed
    decision_layer = None
    if use_decision_layer:
        from autonomous_car.controllers.decision_layer import DecisionLayer
        decision_layer = DecisionLayer(num_lanes=env.num_lanes, verbose=False)
        print("Decision Layer: ENABLED")
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # Compute action scaling: action = scale * tanh_output + bias
    action_high = env.action_space.high
    action_low = env.action_space.low
    action_scale = (action_high - action_low) / 2
    action_bias = (action_high + action_low) / 2
    
    # Create SAC agent
    # Create SAC agent. preset_agent lets a caller inject a warm-started agent
    # (BC policy loaded + demos registered) so the loop does not rebuild one.
    agent = preset_agent or SAC(obs_dim, action_dim, config, action_scale,
                                action_bias, device)
    
    # Register BC demos (always, fresh or resume — demos are not serialized in the ckpt)
    if demos_path:
        demos = np.load(demos_path)
        action_scale_np = (env.action_space.high - env.action_space.low) / 2
        action_bias_np = (env.action_space.high + env.action_space.low) / 2
        ep_completed = demos['episode_completed'] if 'episode_completed' in demos else None
        ep_ids = demos['episode_ids'] if 'episode_ids' in demos else None
        agent.set_demos(
            demos['observations'], demos['actions'],
            action_scale=action_scale_np, action_bias=action_bias_np,
            completed_only=demos_completed_only,
            episode_ids=ep_ids, episode_completed=ep_completed,
        )
    
    # Resume from checkpoint if requested (V18 curriculum / phase-2 warm start)
    if resume_path:
        agent.load(resume_path)
        if total_timesteps > resume_max_timesteps:
            print(f"[RESUME] Capping continuation from {total_timesteps:,} "
                  f"to {resume_max_timesteps:,} env steps")
            total_timesteps = resume_max_timesteps
    
    print(f"\n{'='*60}")
    print("SAC TRAINING")
    print(f"{'='*60}")
    print(f"Observation dim: {obs_dim}")
    print(f"Action dim: {action_dim}")
    print(f"Action range: [{action_low}, {action_high}]")
    print(f"Total timesteps: {total_timesteps:,}")
    print(f"Warmup steps: {config.warmup_steps:,}")
    print(f"Device: {device}")
    print(f"BC lambda schedule: start={config.bc_coef_start} -> target={config.bc_coef}" 
          f" over {config.bc_anneal_steps:,} env steps"
          + ("" if config.bc_anneal_steps else " (FIXED)"))
    if domain_randomization:
        print(f"Domain Randomization: ENABLED")
        print(f"  Traffic range: {traffic_range}")
        print(f"  Obstacles range: {obstacles_range}")
    else:
        print(f"Domain Randomization: DISABLED")
    if demos_path:
        print(f"BC demos: {demos_path} (completed_only={demos_completed_only})")
    if resume_path:
        print(f"Resumed from: {resume_path} (env_steps={agent.env_steps:,})")
    print(f"{'='*60}\n")
    
    # Training loop
    obs, _ = env.reset()
    
    # Initialize decision layer if used
    if decision_layer is not None:
        decision_layer.reset(start_lane=env.current_lane)
        # Get decision and set targets before first observation is used
        dl_input = env.get_decision_layer_input()
        target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
        env.set_decision_targets(target_lane, desired_speed)
        obs = env._get_obs()  # Re-get observation with decision targets
    
    episode_reward = 0
    episode_length = 0
    episode_count = 0
    best_eval_reward = float('-inf')
    best_eval_score = (float('-inf'), float('-inf'), float('-inf'), float('-inf'))
    no_improve_evals = 0
    # A model may only be crowned "_best" once it has been trained meaningfully.
    # Warmup-end evals carry an untrained actor; saving one as best made every
    # later eval a false "no improvement" and preserved the warmup policy.
    best_eligible_steps = 2 * max(config.warmup_steps, 1)

    # A resumed run must beat its source checkpoint, not merely beat -inf.
    # Preserve the source as the best output until a continuation improves it.
    if resume_path:
        baseline_metrics = evaluate_sac_metrics(agent, env, num_episodes=5, seed=eval_seed)
        best_eval_reward = baseline_metrics['mean_reward']
        best_eval_score = (
            baseline_metrics['completion_rate'],
            -baseline_metrics['collision_rate'],
            baseline_metrics.get('mean_min_hazard', float('-inf')),
            best_eval_reward,
        )
        agent.save(save_path.replace('.pt', '_best.pt'))
        print(f"[EVAL] Resume baseline: {baseline_metrics}")
        obs, _ = env.reset()
    
    rewards_history = []

    # Action repetition (frame-skip)
    action_repeat = getattr(env, 'action_repeat', 1)
    env_step = agent.env_steps  # Resume continues from loaded counter; fresh starts at 0
    total_steps_final = env_step + total_timesteps  # total_timesteps MORE env steps from resume point
    last_eval_step = env_step
    last_save_step = env_step
    last_instrument_step = env_step
    eval_n_seeds = eval_n_seeds  # multi-seed eval set size (stability / honest checkpointing)
    
    while env_step < total_steps_final:
        # Decision layer: update strategic targets each env step
        if decision_layer is not None:
            dl_input = env.get_decision_layer_input()
            target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
            env.set_decision_targets(target_lane, desired_speed)
        
        # Select action (SAC decides target lane offset + target speed)
        action = agent.select_action(obs, deterministic=False)
        
        # Action repetition: hold target constant for multiple env steps
        total_reward = 0.0
        steps_this_action = 0
        for _ in range(action_repeat):
            next_obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            env_step += 1
            steps_this_action += 1
            if terminated or truncated:
                break
        
        # Training step (one gradient update per decision)
        metrics = agent.train_step(obs, action, total_reward, next_obs, terminated)
        agent.env_steps = env_step  # Sync env-transition counter for BC curriculum / resume
        agent.sync_schedule(env_step)  # alpha anneal + LR decay (stability)
        
        episode_reward += total_reward
        episode_length += steps_this_action
        
        if terminated or truncated:
            rewards_history.append(episode_reward)
            episode_count += 1
            
            if episode_count % 10 == 0:
                avg_reward = np.mean(rewards_history[-100:]) if rewards_history else 0
                print(f"Step {env_step:,} | Episode {episode_count} | "
                      f"Reward: {episode_reward:.1f} | Avg100: {avg_reward:.1f} | "
                      f"Alpha: {agent.alpha:.3f}")
            
            # Reset with domain randomization if enabled
            if domain_randomization:
                import random
                options = {
                    'num_obstacles': random.randint(*obstacles_range),
                    'num_traffic_vehicles': random.randint(*traffic_range)
                }
                obs, _ = env.reset(options=options)
            else:
                obs, _ = env.reset()
            
            # Reset decision layer for new episode
            if decision_layer is not None:
                decision_layer.reset(start_lane=env.current_lane)
                dl_input = env.get_decision_layer_input()
                target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
                env.set_decision_targets(target_lane, desired_speed)
                obs = env._get_obs()
            
            episode_reward = 0
            episode_length = 0
        else:
            obs = next_obs
        
        # Save checkpoints (using >= to handle early episode termination)
        if env_step - last_save_step >= save_interval:
            last_save_step = env_step
            checkpoint_path = save_path.replace('.pt', f'_step{env_step}.pt')
            agent.save(checkpoint_path)
            # Margin-selection backstop: every step-checkpoint is eval'd on a
            # fixed multi-seed set and the SAFEST is tracked via a dedicated
            # "_best_margin.pt", so a good mid-training peak survives even if
            # the ongoing eval never crowns it or continued training degrades.
            try:
                _mg = aggregate_eval_metrics(
                    evaluate_sac_metrics(agent, env, num_episodes=eval_num_episodes, seed=eval_seed + i)
                    for i in range(eval_n_seeds)
                )
                _mg_score = (
                    _mg['completion_rate'],
                    -_mg['collision_rate'],
                    _mg['mean_min_hazard'],
                    _mg['mean_reward'],
                )
                if _mg_score > best_eval_score:
                    agent.save(save_path.replace('.pt', '_best_margin.pt'))
                    print(f"[SAVE] Multi-seed margin best updated at step {env_step:,} "
                          f"(margin={_mg['mean_min_hazard']:.2f})\n")
            except Exception as _e:
                print(f"[SAVE] margin-eval skipped at step {env_step:,}: {_e}")

        # Critic-hazard instrumentation (stability sweep): sample the replay
        # buffer and record the collapse signatures. Cheap (one forward pass)."""
        if instrument_every and env_step - last_instrument_step >= instrument_every \
                and len(agent.buffer) >= 256:
            last_instrument_step = env_step
            rec = instrument_critic(agent, batch_size=512)
            if rec is not None:
                rec = dict(rec)
                rec['type'] = 'critic'
                rec['step'] = env_step
                print(f"[INSTR] {rec}")
                if instrument_out:
                    with open(instrument_out, 'a') as _f:
                        _f.write(json.dumps(rec) + '\n')
        
        # Evaluation (using >= to handle early episode termination)
        if env_step - last_eval_step >= eval_interval:
            last_eval_step = env_step
            agent.env_steps = env_step
            # Multi-seed eval: average over a small fixed seed set so a lucky
            # single seed cannot crown a risky checkpoint, and the safety
            # margin (closest approach) is aggregated so earlier-braking
            # policies can actually be seen as better.
            eval_metrics = aggregate_eval_metrics(
                evaluate_sac_metrics(agent, env, num_episodes=eval_num_episodes, seed=eval_seed + i)
                for i in range(eval_n_seeds)
            )
            eval_reward = eval_metrics['mean_reward']
            eval_score = (
                eval_metrics['completion_rate'],
                -eval_metrics['collision_rate'],
                eval_metrics['mean_min_hazard'],
                eval_reward,
            )
            # Expose the latest eval to callers (sweep worker records it).
            agent.last_metrics = {
                'step': env_step, 'eval_seed': eval_seed,
                'n_episodes': eval_num_episodes * eval_n_seeds,
                **eval_metrics,
            }
            if instrument_out:
                with open(instrument_out, 'a') as _f:
                    _f.write(json.dumps(
                        {'type': 'eval', 'step': env_step, **eval_metrics}) + '\n')
            print(f"\n[EVAL] Step {env_step:,} | {eval_metrics}")

            if eval_score > best_eval_score:
                best_eval_score = eval_score
                best_eval_reward = eval_reward
                no_improve_evals = 0
                if env_step >= best_eligible_steps:
                    agent.save(save_path.replace('.pt', '_best.pt'))
                    print(f"[EVAL] New best model saved! (reward={eval_reward:.1f})\n")
                else:
                    print(f"[EVAL] Best-scoring candidate (reward={eval_reward:.1f}), "
                          f"skipping save before {best_eligible_steps} steps "
                          f"(warmup-maturity gate)\n")
            else:
                no_improve_evals += 1
                print(f"[EVAL] No improvement: {no_improve_evals}/"
                      f"{early_stop_patience} evals")
                if no_improve_evals >= early_stop_patience:
                    print("[EARLY STOP] Evaluation has stopped improving; "
                          "keeping the best checkpoint.")
                    break
    
    # Final instrumentation record (ensures the tail of the run is always logged)
    if instrument_every:
        rec = instrument_critic(agent, batch_size=512)
        if rec is not None:
            rec = dict(rec)
            rec['type'] = 'critic'
            rec['step'] = env_step
            if instrument_out:
                with open(instrument_out, 'a') as _f:
                    _f.write(json.dumps(rec) + '\n')

    # Save final model
    agent.save(save_path)
    print(f"\nTraining complete! Final model saved to {save_path}")
    agent.final_metrics = getattr(agent, 'last_metrics', None)
    return agent


def train_bc(env, config: SACConfig, demos_path: str, steps: int,
             save_path: str, device: str = 'cpu', batch_size: int = 256) -> SAC:
    """Train only the policy on demonstrations for action reconstruction."""
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    action_scale = (env.action_space.high - env.action_space.low) / 2
    action_bias = (env.action_space.high + env.action_space.low) / 2
    agent = SAC(obs_dim, action_dim, config, action_scale, action_bias, device)

    demos = np.load(demos_path)
    ep_completed = demos['episode_completed'] if 'episode_completed' in demos else None
    ep_ids = demos['episode_ids'] if 'episode_ids' in demos else None
    agent.set_demos(
        demos['observations'], demos['actions'],
        action_scale=action_scale, action_bias=action_bias,
        completed_only=ep_completed is not None,
        episode_ids=ep_ids, episode_completed=ep_completed,
    )

    losses = []
    for step in range(1, steps + 1):
        n = min(batch_size, len(agent.demo_obs))
        idx = torch.randint(0, len(agent.demo_obs), (n,))
        demo_obs_t = torch.FloatTensor(agent.demo_obs[idx]).to(agent.device)
        demo_act_t = torch.FloatTensor(agent.demo_actions[idx]).to(agent.device)
        loss = agent._bc_loss(demo_obs_t, demo_act_t)
        agent.bc_optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy.parameters(), agent.grad_clip)
        agent.bc_optimizer.step()
        losses.append(loss.item())
        if step % max(1, steps // 10) == 0:
            print(f"[BC] Step {step:,}/{steps:,} | loss={np.mean(losses[-100:]):.4f}")

    agent.total_steps = steps
    agent.save(save_path)
    print(f"[BC] Pure behavior-cloning policy saved to {save_path}")
    return agent


def evaluate_sac_metrics(agent: SAC, env, num_episodes: int = 10,
                         seed: int = 42) -> Dict[str, float]:
    """Evaluate fixed-seed reward, completion, collision, and lane changes."""
    rewards = []
    completed = 0
    collisions = 0
    lane_changes = 0
    min_hazards = []
    action_repeat = getattr(env, 'action_repeat', 1)

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0
        previous_lane = env.current_lane
        episode_lane_changes = 0
        episode_min_hazard = float('inf')
        done = False
        while not done:
            action = agent.select_action(obs, deterministic=True)
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, info = env.step(action)
                episode_reward += reward
                haz = info.get('hazard_distance', float('inf'))
                if haz is not None and haz < episode_min_hazard:
                    episode_min_hazard = float(haz)
                if env.current_lane != previous_lane:
                    episode_lane_changes += 1
                    previous_lane = env.current_lane
                if terminated or truncated:
                    break
            done = terminated or truncated
        rewards.append(episode_reward)
        completed += int(bool(info.get('completed_lap', False)))
        collisions += int(bool(info.get('obstacle_hit', False) or info.get('traffic_hit', False)))
        lane_changes += episode_lane_changes
        min_hazards.append(episode_min_hazard)

    return {
        'mean_reward': float(np.mean(rewards)),
        'completion_rate': completed / num_episodes,
        'collision_rate': collisions / num_episodes,
        'mean_lane_changes': lane_changes / num_episodes,
        # RISK MARGIN: average closest approach to any hazard. Higher = safer
        # (stopped/braked earlier, never rode the contact edge). This is the
        # signal the reward-only tiebreak could not register.
        'mean_min_hazard': float(np.mean(min_hazards)) if min_hazards else float('inf'),
    }


def aggregate_eval_metrics(metrics_iter) -> Dict[str, float]:
    """Average a set of per-seed eval dicts into one stable multi-seed metric."""
    ms = list(metrics_iter)
    if not ms:
        return {
            'mean_reward': float('-inf'),
            'completion_rate': 0.0,
            'collision_rate': 1.0,
            'mean_lane_changes': 0.0,
            'mean_min_hazard': float('-inf'),
        }
    eps = max(len(ms), 1)
    return {
        'mean_reward': float(np.mean([m['mean_reward'] for m in ms])),
        'completion_rate': float(np.mean([m['completion_rate'] for m in ms])),
        'collision_rate': float(np.mean([m['collision_rate'] for m in ms])),
        'mean_lane_changes': float(np.mean([m['mean_lane_changes'] for m in ms])),
        # lowest margin among the seed set = worst-case, keep it conservative
        'mean_min_hazard': float(min(m['mean_min_hazard'] for m in ms)),
    }


def evaluate_sac(agent: SAC, env, num_episodes: int = 10) -> float:
    """Evaluate SAC agent over multiple episodes.
    
    Args:
        agent: SAC agent
        env: Gymnasium environment
        num_episodes: Number of evaluation episodes
    
    Returns:
        Mean episode reward
    """
    # Check if we need decision layer logic during evaluation
    decision_layer = None
    if getattr(env, 'use_decision_layer', False):
        from autonomous_car.controllers.decision_layer import DecisionLayer
        decision_layer = DecisionLayer(num_lanes=env.num_lanes, verbose=False)

    total_rewards = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        
        # Initialize decision layer
        if decision_layer is not None:
            decision_layer.reset(start_lane=env.current_lane)
            dl_input = env.get_decision_layer_input()
            target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
            env.set_decision_targets(target_lane, desired_speed)
            
        episode_reward = 0
        done = False
        
        while not done:
            # Update strategic targets each step
            if decision_layer is not None:
                dl_input = env.get_decision_layer_input()
                target_lane, desired_speed, _ = decision_layer.decide(**dl_input, dt=env.dt)
                env.set_decision_targets(target_lane, desired_speed)

            action = agent.select_action(obs, deterministic=True)
            action_repeat = getattr(env, 'action_repeat', 1)
            for _ in range(action_repeat):
                obs, reward, terminated, truncated, _ = env.step(action)
                episode_reward += reward
                if terminated or truncated:
                    break
            done = terminated or truncated
        
        total_rewards.append(episode_reward)
    
    return np.mean(total_rewards)
