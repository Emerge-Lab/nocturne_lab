"""Run behavioral cloning on Waymo Open Dataset"""
import glob
import logging
import os
from datetime import datetime

import numpy as np
import torch
from imitation.algorithms import bc
from imitation.data.types import Transitions
from stable_baselines3.common import policies
from torch.utils.data import DataLoader
from evaluation.policy_evaluation import evaluate_policy
import wandb
from utils.config import load_config
from utils.imitation_learning.waymo_iterator import TrajectoryIterator
from utils.policies import load_policy
from utils.string_utils import datetime_to_str

class CustomFeedForwardPolicy(policies.ActorCriticPolicy):
    """A feed forward policy network with a number of hidden units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=bc_config.net_arch)


# Device TODO: Add support for CUDA
device = "cpu"
logging.basicConfig(level=logging.INFO)

def train_bc(env_config, bc_config, num_train_files, use_av_only, train_epochs, num_eval_episodes=100):
    """Train a policy on Waymo Open Dataset using behavioral cloning.
    
    Args:
    - env_config: (dict) Environment configuration
    - bc_config: (dict) Behavioral cloning configuration
    - num_train_files: (int) Number of training files
    - use_av_only: (bool) Use only the AV vehicle to generate the dataset
    - num_eval_episodes: (int) Number of episodes to evaluate the policy
    """
    
    run = wandb.init( 
        project="eval_il_policy",
        sync_tensorboard=True,
        group=f"BC_S{num_train_files}",
    )

    logging.info(f"Training human policy on {num_train_files} files.")

    # Settings 
    bc_config.num_files = num_train_files
    env_config.use_av_only = use_av_only

    logging.info("(1/4) Create iterator...")
    
    waymo_iterator = TrajectoryIterator(
        env_config=env_config,
        data_path=env_config.data_path,
        apply_obs_correction=False,
        file_limit=bc_config.num_files,
    )

    logging.info("(2/4) Generating dataset from traffic scenes...")

    # Rollout to get obs-act-obs-done trajectories
    rollouts = next(
        iter(
            DataLoader(
                waymo_iterator,
                batch_size=bc_config.total_samples,
                pin_memory=True,
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

    # Make custom policy
    policy = CustomFeedForwardPolicy(
        observation_space=waymo_iterator.observation_space,
        action_space=waymo_iterator.action_space,
        lr_schedule=lambda _: torch.finfo(torch.float32).max,
    )

    # Define trainer
    rng = np.random.default_rng()
    bc_trainer = bc.BC(
        policy=policy,
        observation_space=waymo_iterator.observation_space,
        action_space=waymo_iterator.action_space,
        demonstrations=transitions,
        rng=rng,
        device=torch.device("cpu"),
    )
    
    logging.info("(3/4) Training...")

    # Train
    bc_trainer.train(
        n_epochs=train_epochs,
        log_interval=15_000,
    )

    logging.info("(4/4) Evaluate policy...")
    
    # Evaluate, get scores
    df_bc = evaluate_policy(
        env_config=env_config,
        controlled_agents=1,
        data_path=env_config.data_path,
        mode="policy",
        policy=bc_trainer.policy,
        select_from_k_scenes=num_train_files,
        num_episodes=num_eval_episodes,
        use_av_only=True,
    )
    
    logging.info(f'--- Results: BC; AV ONLY ---')
    print(df_bc[["goal_rate", "off_road", "veh_veh_collision"]].mean())
    
    # Evaluate, get scores
    df_bc_all = evaluate_policy(
        env_config=env_config,
        controlled_agents=1,
        data_path=env_config.data_path,
        mode="policy",
        policy=bc_trainer.policy,
        select_from_k_scenes=num_train_files,
        num_episodes=num_eval_episodes,
        use_av_only=False,
    )
    
    logging.info(f'--- Results: BC; RANDOM VEHICLE ---')
    print(df_bc_all[["goal_rate", "off_road", "veh_veh_collision"]].mean())
    
    # Save policy
    if bc_config.save_model:
        # Save model
        datetime_ = datetime_to_str(dt=datetime.now())
        save_path = f"{bc_config.save_model_path}" 
        name = f"{bc_config.model_name}_D{waymo_iterator.action_space.n}_S{num_train_files}_{datetime_}"
        bc_trainer.policy.save(
            path=f'{save_path}/{name}.pt'
        )
        logging.info("(4/4) Saved policy!")
    
        # BEHAVIORAL CLONING
        human_policy = load_policy(
            data_path=save_path,
            file_name=name, 
        )
        
        df_bc_loaded = evaluate_policy(
            env_config=env_config,
            controlled_agents=1,
            data_path=env_config.data_path,
            mode="policy",
            policy=human_policy,
            select_from_k_scenes=num_train_files,
            num_episodes=num_eval_episodes,
            use_av_only=True,
        )
        
        logging.info(f'--- Results: BEHAVIORAL CLONING LOADED ---')
        print(df_bc_loaded[["goal_rate", "off_road", "veh_veh_collision"]].mean())
        

if __name__ == "__main__":
    
    av_settings = [True, False]
    train_epochs = [20, 50]
    
    for use_av_only, n_epochs in zip(av_settings, train_epochs):
        
        logging.info(f'---- Use AV only: {use_av_only} ----')
        
        # Configs
        bc_config = load_config("bc_config")
        env_config = load_config("env_config")
        
        train_bc(
            num_train_files=100,
            train_epochs=n_epochs,
            use_av_only=use_av_only,
            env_config=env_config,
            bc_config=bc_config,
            num_eval_episodes=1000, 
        )
        
        
    
    