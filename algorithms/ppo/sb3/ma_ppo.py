import logging
from typing import Optional

import numpy as np
import torch
from box import Box
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import get_schedule_fn, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from torch import nn

from networks.perm_eq_late_fusion import LateFusionNet

# Import masked buffer class
from algorithms.ppo.sb3.masked_buffer import MaskedRolloutBuffer

logging.getLogger(__name__)

class MultiAgentPPO(PPO):
    """Adapted Proximal Policy Optimization algorithm (PPO) that is compatible with multi-agent environments."""

    def __init__(
        self,
        *args,
        env_config: Optional[Box] = None, 
        mlp_class: nn.Module = LateFusionNet,
        mlp_config: Optional[Box] = None,
        **kwargs,
    ):
        self.env_config = Box(env_config) if env_config is not None else None
        self.mlp_class = mlp_class
        self.mlp_config = Box(mlp_config) if mlp_config is not None else None
        super().__init__(*args, **kwargs)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: MaskedRolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """Adapted collect_rollouts function."""

        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        # Sample new weights for the state dependent exploration
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.policy.reset_noise(env.num_envs)

            with torch.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # EDIT_1: Mask out invalid observations (NaN dimensions and/or dead agents)
                # Create dummy actions, values and log_probs (NaN)
                actions = torch.full(fill_value=np.nan, size=(self.n_envs,)).to(self.device)
                log_probs = torch.full(fill_value=np.nan, size=(self.n_envs,), dtype=torch.float32).to(self.device)
                values = (
                    torch.full(fill_value=np.nan, size=(self.n_envs,), dtype=torch.float32)
                    .unsqueeze(dim=1)
                    .to(self.device)
                )

                # Get indices of alive agent ids
                alive_agent_idx = [
                    idx for idx, agent_id in enumerate(env.agent_ids) if agent_id not in env.dead_agent_ids
                ]
                obs_tensor_alive = obs_tensor[alive_agent_idx, :]

                # Predict actions, vals and log_probs given obs
                actions_tmp, values_tmp, log_prob_tmp = self.policy(obs_tensor_alive)
                (
                    actions[alive_agent_idx],
                    values[alive_agent_idx],
                    log_probs[alive_agent_idx],
                ) = (
                    actions_tmp.float(),
                    values_tmp.float(),
                    log_prob_tmp.float(),
                )

            actions = actions.cpu().numpy()

            # Rescale and perform action
            clipped_actions = actions

            if isinstance(self.action_space, spaces.Box):
                if self.policy.squash_output:
                    # Unscale the actions to match env bounds
                    # if they were previously squashed (scaled in [-1, 1])
                    clipped_actions = self.policy.unscale_action(clipped_actions)
                else:
                    # Otherwise, clip the actions to avoid out of bound error
                    # as we are sampling from an unbounded Gaussian distribution
                    clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)

            new_obs, rewards, dones, infos = env.step(clipped_actions)

            # EDIT_2: Increment the global step by the number of valid samples in rollout step
            samples_in_timestep = env.num_envs - np.isnan(dones).sum()
            self.num_timesteps += samples_in_timestep

            # Give access to local variables
            callback.update_locals(locals())
            if callback.on_step() is False:
                return False

            self._update_info_buffer(infos)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with torch.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with torch.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())

        callback.on_rollout_end()

        # EDIT_3: Reset buffer after each rollout
        env.total_agents_in_rollout = 0
        env.num_agents_collided = 0
        env.num_agents_off_road = 0
        env.num_agents_goal_achieved = 0
        env.n_episodes = 0
        env.episode_lengths = []
        env.agents_in_scene = []

        return True

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed)

        # Change buffer to our own masked version
        buffer_cls = MaskedRolloutBuffer

        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            device=self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )

        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            env_config=self.env_config,
            mlp_class=self.mlp_class,
            mlp_config=self.mlp_config,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device)

        # Initialize schedules for policy/value clipping
        self.clip_range = get_schedule_fn(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = get_schedule_fn(self.clip_range_vf)