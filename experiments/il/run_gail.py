"""Run Generative Adversarial Imitation Learning on Waymo Open Dataset"""
import logging
from datetime import datetime

import numpy as np
from stable_baselines3 import PPO
import torch
import wandb
from imitation.algorithms.adversarial.gail import GAIL
from imitation.data.types import Transitions
from stable_baselines3.common import policies
from torch.utils.data import DataLoader
from stable_baselines3.common.evaluation import evaluate_policy as epsb3
from evaluation.policy_evaluation import evaluate_policy
from utils.configs import load_config
from utils.imitation_learning.waymo_iterator import TrajectoryIterator
from utils.policies import load_policy
from stable_baselines3.common.vec_env import DummyVecEnv 
from imitation.util.networks import RunningNorm
from imitation.rewards.reward_nets import BasicRewardNet
from utils.string_utils import datetime_to_str
import gymnasium as gym
from nocturne.envs.base_env import BaseEnv
from nocturne.wrappers.sb3_wrappers import NocturneToSB3
import functools
from typing import List, Callable
from stable_baselines3.common.vec_env import DummyVecEnv

# Make a vectorized environment using NocturneToSB3 wrapper
def make_nocturne_vec_env(env_config) -> DummyVecEnv:
    return DummyVecEnv([lambda: NocturneToSB3(BaseEnv(env_config))])

device = "cpu"
logging.basicConfig(level=logging.INFO)

def train_gail(env_config, gail_config, num_train_files, use_av_only, train_epochs, num_eval_episodes=100):
    """Train a policy on Waymo Open Dataset using Generative Adversarial Imitation Learning.

    Args:
    - env_config: (dict) Environment configuration
    - gail_config: (dict) GAIL configuration
    - num_train_files: (int) Number of training files
    - use_av_only: (bool) Use only the AV vehicle to generate the dataset
    - num_eval_episodes: (int) Number of episodes to evaluate the policy
    """

    run = wandb.init(
        project="eval_il_policy",
        sync_tensorboard=True,
        group=f"GAIL_S{num_train_files}",
    )

    logging.info(f"Training human policy on {num_train_files} files.")

    # Settings
    gail_config.num_files = num_train_files
    env_config.use_av_only = use_av_only

    # Generate expert trajectories, that the discriminator needs to distinguish 
    # from the learner’s trajectories.
    waymo_iterator = TrajectoryIterator(
        env_config=env_config,
        data_path=env_config.data_path,
        apply_obs_correction=False,
        file_limit=gail_config.num_files,
    )

    # Rollout to get obs-act-obs-done trajectories
    rollouts = next(
        iter(
            DataLoader(
                waymo_iterator,
                batch_size=bc_config.total_samples,
                pin_memory=False,
            )
        )
    )

    # Convert to dataset of imitation "transitions"
    transitions = Transitions(
        obs=rollouts[0].to(device),
        acts=rollouts[1].to(device),
        infos=np.zeros_like(rollouts[0]),  # Dummy
        next_obs=rollouts[2],
        dones=np.array(rollouts[3]).astype(bool),
    )

    # Initialize vec env 
    env = make_nocturne_vec_env(env_config)

    # Learnner
    gail_learner = PPO(
        env=env,
        policy='MlpPolicy',
        batch_size=256,
        ent_coef=0.0,
        learning_rate=3e-4,
        gamma=0.98,
        n_epochs=500,
        seed=42,  
    )
    
    # Make discriminator network to distinguish between expert trajectories 
    # and learner trajectories
    discriminator = BasicRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    
    # Define GAIL trainer
    gail_trainer = GAIL(
        demonstrations=transitions,
        demo_batch_size=bc_config.total_samples,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=8,
        venv=env,  # environment
        gen_algo=gail_learner,  # generator policy network
        reward_net=discriminator,  # reward network (??)
        allow_variable_horizon=True,
    )
    
    learner_rewards_before_training, _ = epsb3(
        gail_learner, env, 100, return_episode_rewards=True
    )
    
    print("mean reward before training:", np.mean(learner_rewards_before_training))

    # Train
    gail_trainer.train(
        total_timesteps=10_000,
    )
    
    learner_rewards_after_training, _ = epsb3(
        gail_learner, env, 100, return_episode_rewards=True
    )
    
    print("mean reward after training:", np.mean(learner_rewards_after_training))

    # Evaluate, get scores
    df_gail = evaluate_policy(
        env_config=env_config,
        controlled_agents=1,
        data_path=env_config.data_path,
        mode="policy",
        policy=gail_learner,
        select_from_k_scenes=num_train_files,
        num_episodes=num_eval_episodes,
        use_av_only=True,
    )

    logging.info(f'--- Results: GAIL; AV ONLY ---')
    print(df_gail[["goal_rate", "off_road", "veh_veh_collision"]].mean())

    # Evaluate, get scores
    df_gail_all = evaluate_policy(
        env_config=env_config,
        controlled_agents=1,
        data_path=env_config.data_path,
        mode="policy",
        policy=gail_learner,
        select_from_k_scenes=num_train_files,
        num_episodes=num_eval_episodes,
        use_av_only=False,
    )

    logging.info(f'--- Results: GAIL; RANDOM VEHICLE ---')
    print(df_gail_all[["goal_rate", "off_road", "veh_veh_collision"]].mean())

    # Save policy (if required)
    if gail_config.save_model:
        # Save model
        datetime_ = datetime_to_str(dt=datetime.now())
        save_path = f"{gail_config.save_model_path}"
        name = f"{gail_config.model_name}_D{waymo_iterator.action_space.n}_S{num_train_files}_{datetime_}"
        gail_learner.save(
            path=f'{save_path}/{name}.pt'
        )
        logging.info("(4/4) Saved policy!")

        # Load the saved policy for evaluation
        loaded_policy = load_policy(
            data_path=save_path,
            file_name=name,
        )

        df_gail_loaded = evaluate_policy(
            env_config=env_config,
            controlled_agents=1,
            data_path=env_config.data_path,
            mode="policy",
            policy=gail_learner,
            select_from_k_scenes=num_train_files,
            num_episodes=num_eval_episodes,
            use_av_only=True,
        )

        logging.info(f'--- Results: GAIL LOADED ---')
        print(df_gail_loaded[["goal_rate", "off_road", "veh_veh_collision"]].mean())


if __name__ == "__main__":

    av_settings = [True, False]
    train_epochs = [20, 50]

    # for use_av_only, n_epochs in zip(av_settings, train_epochs):

    #     logging.info(f'---- Use AV only: {use_av_only} ----')

    # Configs
    bc_config = load_config("bc_config")
    env_config = load_config("env_config")
    env_config.max_num_vehicles = 1

    train_gail(
        num_train_files=100,
        train_epochs=20,
        use_av_only=True,
        env_config=env_config,
        gail_config=bc_config,
        num_eval_episodes=200,
    )
