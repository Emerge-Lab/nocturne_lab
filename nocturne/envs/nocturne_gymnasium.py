"""Gymnasium vectorizable environment wrapper for Nocturne."""
import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, TypeVar

import gym
import gymnasium
import numpy as np

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config

logging.basicConfig(level=logging.INFO)

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class NocturneGymnasium(gymnasium.Env):
    """Nocturne environment wrapper for compatible with SB3.
    """

    def __init__(self, config, num_agents, psr=False):
        self.env = BaseEnv(config)

        # Make action and observation spaces compatible with SB3 (requires gymnasium)
        self.action_space = gymnasium.spaces.MultiDiscrete([self.env.config.max_num_vehicles, self.env.action_space.n])
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
        self.num_agents = num_agents  # The maximum number of agents allowed in the environmen
        self.psr = psr # Whether to use PSR or not

        self.psr_dict = self.init_scene_dict() if psr else None # Initialize dict to keep track of the average reward obtained in each scene
        self.n_episodes = 0
        self.episode_lengths = []
        self.rewards = []  # Log reward per step
        self.dead_agent_ids = []  # Log dead agents per step
        self.num_agents_collided = 0  # Keep track of how many agents collided
        self.total_agents_in_rollout = 0 # Log total number of agents in rollout
        self.num_agents_goal_achieved = 0 # Keep track of how many agents reached their goal
        self.agents_in_scene = []
        self.filename = None # If provided, always use the same file 

    def step(self, actions):
        """Take a step in the environment, convert dicts to np arrays.

        Args
        ----
            action (Dict): Dictionary with a single action for the controlled vehicle.

        Returns
        -------
            observation, reward, terminated, truncated, info (np.ndarray, float, bool, bool, dict)
        """
        next_obs_dict, rewards_dict, dones_dict, info_dict = self.env.step(
            action_dict=actions
        )

        return (
            next_obs_dict,
            rewards_dict,
            dones_dict,
            False,
            info_dict,
        )

    def reset(self, seed=None):
        """Reset environment and return initial observations."""
        obs_dict = self.env.reset()
                
        # Reset Nocturne env
        obs_dict = self.env.reset(self.filename, self.psr_dict)

        # Reset storage
        self.agent_ids = []
        self.rewards = []
        self.dead_agent_ids = []
        self.ep_collisions = 0
        self.ep_goal_achived = 0

        obs_all = np.full(fill_value=np.nan, shape=(self.num_envs, self.env.observation_space.shape[0]))
        for idx, agent_id in enumerate(obs_dict.keys()):
            self.agent_ids.append(agent_id)
            obs_all[idx, :] = obs_dict[agent_id]

        # Save obs in buffer
        self._save_obs(obs_all)

        logging.debug(f"RESET - agent ids: {self.agent_ids}")

        # Make dict for storing the last info set for each agent
        self.last_info_dicts = {agent_id: {} for agent_id in self.agent_ids}

        return self._obs_from_buf(), {}

    def _obs_from_buf(self) -> ObsType:
        """Get observation from buffer."""
        return np.copy(self.buf_obs)

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, action_space):
        self.env.action_space = action_space

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, observation_space):
        self.env.observation_space = observation_space

    def render(self):
        pass

    def close(self):
        pass

    @property
    def seed(self, seed=None):
        return None

    @seed.setter
    def seed(self, seed=None):
        pass

    def __getattr__(self, name):
        return getattr(self._env, name)

    def get_attr(self, attr_name: str):
        return getattr(self._env, attr_name)

    def set_attr(self, attr_name: str):
        setattr(self._env, attr_name)
