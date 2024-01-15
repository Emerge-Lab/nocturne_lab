"""Gymnasium vectorizable environment wrapper for Nocturne."""
import logging
import time
from copy import deepcopy
from typing import Any, Dict, List, TypeVar, SupportsFloat

import gymnasium
import numpy as np

from pufferlib.emulation import Postprocessor


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
        # self.action_space = gymnasium.spaces.MultiDiscrete([self.env.config.max_num_vehicles, self.env.action_space.n])
        self.num_agents = num_agents  # The maximum number of agents allowed in the environmen
        self.action_space = gymnasium.spaces.MultiDiscrete([self.env.action_space.n] * self.num_agents)
        self.observation_space = gymnasium.spaces.Box(-np.inf, np.inf, [self.num_agents, self.env.observation_space.shape[0]], np.float32)
        self.psr = psr # Whether to use PSR or not

        self.buf_obs = None  # type: ObsType
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

    def step(self, actions) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """Convert action vector to dict and call env.step()."""

        agent_actions = {
            agent_id: actions[idx] for idx, agent_id in enumerate(self.agent_ids) if agent_id not in self.dead_agent_ids
        }

        # Take a step to obtain dicts
        next_obses_dict, rew_dict, done_dict, info_dict = self.env.step(agent_actions)

        # Update dead agents based on most recent done_dict
        for agent_id, is_done in done_dict.items():
            if is_done and agent_id not in self.dead_agent_ids:
                self.dead_agent_ids.append(agent_id)
                # Store agents' last info dict
                self.last_info_dicts[agent_id] = info_dict[agent_id].copy()

        # Storage
        obs = np.full(fill_value=np.nan, shape=self.observation_space.shape)
        self.buf_dones = np.full(fill_value=np.nan, shape=(self.num_agents,))
        self.buf_rews = np.full_like(self.buf_dones, fill_value=np.nan)
        self.buf_infos = [{} for _ in range(self.num_agents)]
        
        # Override NaN placeholder for each agent that is alive
        for idx, key in enumerate(self.agent_ids):
            if key in next_obses_dict:
                self.buf_rews[idx] = rew_dict[key]
                self.buf_dones[idx] = done_dict[key] * 1
                self.buf_infos[idx] = info_dict[key]
                obs[idx, :] = next_obses_dict[key] 

        # Save step reward obtained across all agents
        self.rewards.append(sum(rew_dict.values()))
        self.agents_in_scene.append(len(self.agent_ids))

        # Store observation
        self._save_obs(obs)

        # Reset episode if ALL agents are done
        if done_dict["__all__"]:
            for agent_id in self.agent_ids:
                self.ep_collisions += self.last_info_dicts[agent_id]["collided"] * 1
                self.ep_goal_achived += self.last_info_dicts[agent_id]["goal_achieved"] * 1

            # Store the fraction of agents that collided in episode
            self.num_agents_collided += self.ep_collisions
            self.num_agents_goal_achieved += self.ep_goal_achived
            self.total_agents_in_rollout += len(self.agent_ids)

            # Save final observation where user can get it, then reset
            for idx in range(len(self.agent_ids)):
                self.buf_infos[idx]["terminal_observation"] = obs[idx]

            # Log episode stats
            ep_len = self.step_num
            self.n_episodes += 1
            self.episode_lengths.append(ep_len)

            # Store reward at scene level
            if self.psr:
                self.psr_dict[self.env.file]["count"] += 1
                self.psr_dict[self.env.file]["reward"] += (sum(rew_dict.values())) / len(self.agent_ids)
                self.psr_dict[self.env.file]["goal_rate"] += self.ep_goal_achived / len(self.agent_ids)

            # Reset
            obs = self.reset()

        return (
            self._obs_from_buf(),
            np.copy(self.buf_rews),
            self.buf_dones.all(),
            False,
            {'infos': deepcopy(self.buf_infos)},
        )

    # def step(self, actions) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
    #     """Take a step in the environment, convert dicts to np arrays.

    #     Args
    #     ----
    #         action (Dict): Dictionary with a single action for the controlled vehicle.

    #     Returns
    #     -------
    #         observation, reward, terminated, truncated, info (np.ndarray, float, bool, bool, dict)
    #     """
    #     next_obs_dict, rewards_dict, dones_dict, info_dict = self.env.step(
    #         action_dict=actions
    #     )

    #     return (
    #         next_obs_dict,
    #         rewards_dict,
    #         dones_dict,
    #         False,
    #         info_dict,
    #     )

    def reset(self, seed=None):
        """Reset environment and return initial observations."""
        # Reset Nocturne env
        obs_dict = self.env.reset(self.filename, self.psr_dict)

        # Reset storage
        self.agent_ids = []
        self.rewards = []
        self.dead_agent_ids = []
        self.ep_collisions = 0
        self.ep_goal_achived = 0

        obs_all = np.full(fill_value=-np.pi*1e7, shape=self.observation_space.shape, dtype=self.observation_space.dtype)
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

    def _save_obs(self, obs: ObsType) -> None:
        """Save observations into buffer."""
        self.buf_obs = obs

    @property
    def step_num(self) -> List[int]:
        """The episodic timestep."""
        return self.env.step_num

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

    def get_attr(self, attr_name, indices=None):
        raise NotImplementedError()

    def set_attr(self, attr_name, value, indices=None) -> None:
        raise NotImplementedError()

class CustomPostprocessor(Postprocessor):
    '''Basic postprocessor that injects returns and lengths information into infos and
    provides an option to pad to a maximum episode length. Works for single-agent and
    team-based multi-agent environments'''
    def reset(self, obs):
        self.epoch_return = 0
        self.epoch_length = 0
        self.done = False

    def reward_done_truncated_info(self, reward, done, truncated, info):
        if isinstance(reward, (list, np.ndarray)):
            reward = sum(reward)

        # Env is done
        if self.done:
            return reward, done, truncated, info

        self.epoch_length += 1
        self.epoch_return += reward

        if done.all() or truncated:
            info['return'] = self.epoch_return
            info['length'] = self.epoch_length
            self.done = True

        return reward, done, truncated, info

def make_env(env_config, num_agents):
    return NocturneGymnasium(config=env_config, num_agents=num_agents)


def nocturne_creator(env_config, num_agents):
    return pufferlib.emulation.GymnasiumPufferEnv(env_creator=make_env, env_args=(env_config,num_agents,), postprocessor_cls=CustomPostprocessor)

if __name__ == "__main__":
    MAX_AGENTS = 3
    NUM_STEPS = 400

    # Load environment variables and config
    env_config = load_config("env_config")

    # Set the number of max vehicles
    env_config.max_num_vehicles = MAX_AGENTS

    # from stable_baselines3.common.vec_env import SubprocVecEnv

    # # Make environment
    # envs = SubprocVecEnv([lambda: make_env(env_config, MAX_AGENTS) for _ in range(4)])
    env = make_env(env_config, MAX_AGENTS)
    import pufferlib.emulation
    env = pufferlib.emulation.GymnasiumPufferEnv(env, postprocessor_cls=CustomPostprocessor)
    env.reset()
    env.step(env.action_space.sample())
    import pufferlib.vectorization
    vec = pufferlib.vectorization.Multiprocessing
    envs = vec(nocturne_creator,env_args=[env_config, MAX_AGENTS], num_envs=4, envs_per_worker=2, env_pool=True)
    envs.async_reset()
    obs = envs.recv()[0]
    actions = [envs.single_action_space.sample() for _ in range(4)]
    envs.step(actions)
    envs.step(actions)

    for global_step in range(NUM_STEPS):
        # Take random action(s) -- you'd obtain this from a policy
        actions = np.array([envs.action_space.sample() for _ in range(4)])

        # Step
        obs, rew, done, info = envs.step(actions)

        # Log
        # logging.info(f"step_num: {env.step_num} (global = {global_step}) | done: {done} | rew: {rew}")

        time.sleep(0.2)
