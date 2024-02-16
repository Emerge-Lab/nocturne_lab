"""
Description: Imitation-compatible (https://imitation.readthedocs.io/)
iterator for generating expert trajectories in Waymo scenes.
"""
import json
import logging
from itertools import product
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
import pandas as pd
from gym.spaces import Discrete
from torch.utils.data import DataLoader, IterableDataset

from nocturne import Simulation
from nocturne.envs.base_env import BaseEnv
from utils.config import load_config

# Global setting
logging.basicConfig(level="DEBUG")

class TrajectoryIterator(IterableDataset):
    def __init__(self, data_path, env_config, apply_obs_correction=False, with_replacement=True, file_limit=-1):
        self.data_path = Path(data_path)
        self.config = env_config
        self.apply_obs_correction = apply_obs_correction
        self.env = BaseEnv(env_config)
        self.with_replacement = with_replacement
       
        # Select traffic scenes to sample from
        self.valid_veh_dict = json.load(open(f"{self.data_path}/valid_files.json", "r", encoding="utf-8"))
        file_paths = self.data_path.glob('*.json')
        self.file_names = sorted([os.path.basename(file) for file in file_paths])[:file_limit]

        self._set_discrete_action_space()
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, self.env.observation_space.shape, np.float32)
        self.action_space = gym.spaces.Discrete(len(self.actions_to_joint_idx))
        self.ep_norm_rewards = []

        super(TrajectoryIterator).__init__()

        logging.info(f"Using {len(self.file_names)} file(s)")
        logging.info(f"Action space: {self.action_space} D")

    def __iter__(self):
        """Return an (expert_state, expert_action) iterable."""
        return self._get_trajectories()

    def _get_trajectories(self):
        """Load scenes, preprocess and return trajectories."""

        if len(self.file_names) == None:
            logging.info("file_names is empty.")
            return None

        while True:
            # (1) Sample traffic scene
            if self.with_replacement:
                filename = np.random.choice(self.file_names)
            else:  # Every scene can only be used once
                filename = self.file_names.pop()

            # (2) Obtain discretized expert actions
            #TODO: @dc fix with only avs
            # if self.apply_obs_correction:
            #     expert_actions_df = self._discretize_expert_actions(filename)
            # else:
            expert_actions_df = None

            # (3) Obtain observations
            expert_obs, expert_acts, expert_next_obs, expert_dones = self._step_through_scene(filename)
            
            # (4) Return
            for obs, act, next_obs, done in zip(expert_obs, expert_acts, expert_next_obs, expert_dones):
                yield (obs, act, next_obs, done)

    def _discretize_expert_actions(self, filename: str):
        """Discretize human expert actions in given traffic scene."""

        # Create simulation
        sim = Simulation(f"{self.data_path}/{filename}", dict(self.config.scenario))
        scenario = sim.getScenario()

        objects_that_moved = scenario.getObjectsThatMoved()
        objects_of_interest = [
            obj
            for obj in scenario.getVehicles()
            if obj in objects_that_moved and obj.getID() not in self.valid_veh_dict[filename]
        ]

        # Setup dataframe to store actions
        actions_dict = {}
        for agent in objects_of_interest:
            actions_dict[agent.id] = np.zeros(self.config.episode_length)

        df_actions = pd.DataFrame(actions_dict)

        for timestep in range(self.config.episode_length + self.config.warmup_period):
            for veh_obj in objects_of_interest:
                # Get (continuous) expert action
                expert_action = scenario.expert_action(veh_obj, sim.step_num)

                # Check for invalid actions (None) (because no value available for taking
                # derivative) or because the vehicle is at an invalid state
                if expert_action is None:
                    continue

                expert_accel, expert_steering, _ = expert_action.numpy()

                # Map actions to nearest grid indices and joint action
                accel_grid_val, accel_grid_idx = self._find_closest_index(self.accel_grid, expert_accel)
                steering_grid_val, steering_grid_idx = self._find_closest_index(self.steering_grid, expert_steering)

                expert_action_idx = self.actions_to_joint_idx[accel_grid_val, steering_grid_val][0]

                if expert_action_idx is None:
                    logging.debug("Expert action is None!")

                # Store
                if timestep >= self.config.warmup_period:
                    df_actions.loc[timestep - self.config.warmup_period][veh_obj.getID()] = expert_action_idx

            sim.step(self.config.dt)

        return df_actions

    def _step_through_scene(self, filename: str, expert_actions_df: pd.DataFrame = None):
        """
        Step through a traffic scenario using a set of discretized expert actions
        to construct a set of corrected state-action pairs. Note: A state-action pair
        is the observation + the action chosen given that observation.
        """
        # Reset
        next_obs_dict = self.env.reset(filename)
        num_agents = len(next_obs_dict.keys())
        agent_ids = list(next_obs_dict.keys())
        dead_agent_ids = []
        veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(agent_ids)}
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}
    
        # Storage
        expert_action_arr = np.full(
            (self.config.episode_length, num_agents), 
            fill_value=np.nan,
        )

        obs_arr = np.full(
            (
                self.config.episode_length,
                num_agents,
                self.env.observation_space.shape[0],
            ),
            fill_value=np.nan,
        )
        next_obs_arr = np.full_like(obs_arr, fill_value=np.nan)
        dones_arr = np.full_like(expert_action_arr, fill_value=np.nan)
        ep_rewards = np.zeros(num_agents)

        # Step through scene
        for t_idx in range(self.config.episode_length):
        
            action_dict = {}
            
            if expert_actions_df is None:
                for veh_obj in self.env.controlled_vehicles:
                    
                    # Get (continuous) expert action
                    expert_action = self.env.scenario.expert_action(veh_obj, self.env.step_num)

                    # Discretize expert action
                    if expert_action is not None:
                        if expert_action.steering == expert_action.steering:
                            expert_accel, expert_steering, _ = expert_action.numpy()
                            
                            # Map actions to nearest grsid indices and joint action
                            acc_grid_idx = np.argmin(np.abs(self.env.accel_grid - expert_accel))
                            ste_grid_idx = np.argmin(np.abs(self.env.steering_grid - expert_steering))

                            expert_action_idx = self.env.actions_to_idx[
                                self.env.accel_grid[acc_grid_idx],
                                self.env.steering_grid[ste_grid_idx],
                            ][0]
                            action_dict[veh_obj.id] = expert_action_idx
                        
                    else: # Skip if expert action is None or nan
                        continue
            else:
                # Select action from expert grid actions dataframe
                for veh_obj in self.env.controlled_vehicles:
                    if veh_obj.id in next_obs_dict:
                        action = int(expert_actions_df[veh_obj.id].loc[self.env.step_num])
                        action_dict[veh_obj.id] = action

            # Store actions + obervations of living agents
            for veh_obj in self.env.controlled_vehicles:
                if veh_obj.id not in dead_agent_ids:
                    veh_idx = veh_id_to_idx[veh_obj.id]
                    obs_arr[t_idx, veh_idx, :] = next_obs_dict[veh_obj.id]

                    if veh_obj.id in action_dict:
                        expert_action_arr[t_idx, veh_idx] = action_dict[veh_obj.id]

            # Execute actions
            next_obs_dict, rew_dict, done_dict, info_dict = self.env.step(action_dict)

            # The i'th observation `next_obs[i]` in this array is the observation
            # after the agent has taken action `acts[i]`.
            for veh_obj in self.env.controlled_vehicles:
                veh_idx = veh_id_to_idx[veh_obj.id]
                if veh_obj.id not in dead_agent_ids:
                    next_obs_arr[t_idx, veh_idx, :] = next_obs_dict[veh_obj.id]
                    dones_arr[t_idx, veh_idx] = done_dict[veh_obj.id]

            # Update rewards
            for veh_obj in self.env.controlled_vehicles:
                if veh_obj.id in rew_dict:
                    veh_idx = veh_id_to_idx[veh_obj.id]
                    ep_rewards[veh_idx] += rew_dict[veh_obj.id]

            # Update dead agents
            for veh_id, is_done in done_dict.items():
                if is_done and veh_id not in dead_agent_ids:
                    dead_agent_ids.append(veh_id)

            if done_dict["__all__"]:  # If all agents are done or episode is done
                break
        
        # Save accumulated normalized reward
        self.ep_norm_rewards.append(sum(ep_rewards) / num_agents)

        # Some vehicles may be finished earlier than others, so we mask out the invalid samples
        # And flatten along the agent axis
        valid_samples_mask = ~np.isnan(expert_action_arr)

        expert_action_arr = expert_action_arr[valid_samples_mask]
        obs_arr = obs_arr[valid_samples_mask]
        next_obs_arr = next_obs_arr[valid_samples_mask]
        dones_arr = dones_arr[valid_samples_mask].astype(bool)

        return obs_arr, expert_action_arr, next_obs_arr, dones_arr

    def _set_discrete_action_space(self):
        """Set the discrete action space."""

        self.action_space = Discrete(self.config.accel_discretization * self.config.steering_discretization)
        self.accel_grid = np.linspace(
            -np.abs(self.config.accel_lower_bound),
            self.config.accel_upper_bound,
            self.config.accel_discretization,
        )
        self.steering_grid = np.linspace(
            -np.abs(self.config.steering_lower_bound),
            self.config.steering_upper_bound,
            self.config.steering_discretization,
        )
        self.joint_idx_to_actions = {}
        self.actions_to_joint_idx = {}
        for i, (accel, steer) in enumerate(product(self.accel_grid, self.steering_grid)):
            self.joint_idx_to_actions[i] = [accel, steer]
            self.actions_to_joint_idx[accel, steer] = [i]

    def _find_closest_index(self, action_grid, action):
        """Find the nearest value in the action grid for a given expert action."""
        indx = np.argmin(np.abs(action_grid - action))
        return action_grid[indx], indx
