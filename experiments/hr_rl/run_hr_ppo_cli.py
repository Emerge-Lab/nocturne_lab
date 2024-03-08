"""Train HR-PPO agent with CLI arguments."""
import logging
import os
from datetime import datetime
from random import randint
from time import sleep
from typing import Callable
import numpy as np
import torch
import typer
from box import Box
from stable_baselines3.common.policies import ActorCriticPolicy
import wandb

# Permutation equivariant network
from networks.perm_eq_late_fusion import LateFusionNet, LateFusionPolicy

# Multi-agent as vectorized environment
from nocturne.envs.vec_env_ma import MultiAgentAsVecEnv
from utils.config import load_config
from utils.random_utils import init_seed
from utils.render import make_video

# Custom callback
from utils.sb3.callbacks import CustomMultiAgentCallback

# Custom PPO class that supports multi-agent control
from utils.sb3.reg_ppo import RegularizedPPO
from utils.string_utils import datetime_to_str

os.environ["WANDB__SERVICE_WAIT"] = "300"

logging.basicConfig(level=logging.INFO)

# Default settings
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")

LAYERS_DICT = {
    "tiny": [64],
    "small": [128, 64],
    "medium": [256, 128],
    "large": [256, 128, 64],
}

TIME_STEPS_DICT = {
    'A': [10, 20, 40, 60],
    'B': [10, 20, 30, 40, 50, 60, 70],
}

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """

    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func

def run_hr_ppo(
    sweep_name: str = exp_config.group,
    ent_coef: float = 0.001,
    vf_coef: float = 0.5,
    mini_batch_size: int = 256,
    seed: int = 42,
    lr: float = 3e-4,
    arch_road_objects: str = "large",
    arch_road_graph: str = "medium",
    arch_shared_net: str = "small",
    activation_fn: str = "tanh",
    dropout: float = 0.0,
    randomize_goals: int = 0,
    rand_goals_timesteps: str = 'A',
    total_timesteps: int = 60_000_000,
    num_files: int = 10,
    reg_weight: float = 0.0,
    reg_weight_decay: str = "None",
    num_controlled_veh: int = 1,
    pretrained_model: str = "None",
) -> None:
    """Train RL agent using PPO with CLI arguments."""
    
    # ==== Overwrite default settings ==== #
    # Environment
    env_config.num_files = num_files   
    
    # Randomize goals
    if randomize_goals == 1:
        env_config.target_positions.randomize_goals = True
        env_config.target_positions.time_steps_list = TIME_STEPS_DICT[rand_goals_timesteps]
    elif randomize_goals == 0:
        env_config.target_positions.randomize_goals = False

    # Set the number of vehicles to control per scene
    # If set to 1 we're doing single-agent RL (during training)
    env_config.max_num_vehicles = num_controlled_veh
    if env_config.max_num_vehicles == 1:
        exp_config.ppo.n_steps = int(4096 * 5)

    # Experiment
    exp_config.seed = seed
    exp_config.ppo.vf_coef = vf_coef
    exp_config.ppo.ent_coef = ent_coef
    exp_config.ppo.learning_rate = lr
    exp_config.ppo.batch_size = mini_batch_size
    exp_config.learn.total_timesteps = total_timesteps
    exp_config.reg_weight = reg_weight
    if reg_weight_decay != "None":
        exp_config.reg_weight_decay_schedule = reg_weight_decay
    
    # Model architecture
    exp_config.model_config.arch_ro = arch_road_objects
    exp_config.model_config.arch_rg = arch_road_graph
    exp_config.model_config.arch_shared = arch_shared_net
    exp_config.model_config.act_func = activation_fn

    # Define model architecture
    model_config = Box(
        {
            "arch_road_objects": LAYERS_DICT[arch_road_objects],
            "arch_road_graph": LAYERS_DICT[arch_road_graph],
            "arch_shared_net": LAYERS_DICT[arch_shared_net],
            "act_func": activation_fn,
            "dropout": dropout,
        }
    )
    
    # ==== Overwrite default settings ==== #
    # Ensure reproducability
    init_seed(env_config, exp_config, exp_config.seed)

    # Make environment
    env = MultiAgentAsVecEnv(
        config=env_config,
        num_envs=env_config.max_num_vehicles,
    )
    
    # Add scene to config
    exp_config.scene = env.filename

    if exp_config.track_wandb:
        run = wandb.init(
            project=exp_config.project,
            group=sweep_name,
            config={**exp_config, **env_config, **model_config},
            **exp_config.wandb,
        )

    # Set device
    exp_config.ppo.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Logging
    logging.info(f"Created env. Max # agents = {env_config.max_num_vehicles}.")
    logging.info(f"Learning in {len(env.env.files)} scene(s). Device = {exp_config.ppo.device}.")

    logging.info(f"--- obs_space: {env.observation_space.shape[0]} ---")
    logging.info(f"--- act_space: {env.action_space.n} ---")
    logging.info(f"--- randomized goals: {env_config.target_positions.randomize_goals} ---")

    # Initialize custom callback
    custom_callback = CustomMultiAgentCallback(
        env_config=env_config,
        exp_config=exp_config,
        video_config=video_config,
        wandb_run=run,
    )

    # Make scene init video to check expert actions
    if exp_config.track_wandb:
        for model in exp_config.wandb_init_videos:
            make_video(
                env_config=env_config,
                exp_config=exp_config,
                video_config=video_config,
                filenames=[env.filename],
                model=model,
                n_steps=None,
            )

    human_policy = None
    # Load human reference policy if regularization is used
    if exp_config.reg_weight > 0.0:
        saved_variables = torch.load(exp_config.human_policy_path, map_location=exp_config.ppo.device)
        human_policy = ActorCriticPolicy(**saved_variables["data"])
        human_policy.load_state_dict(saved_variables["state_dict"])
        human_policy.to(exp_config.ppo.device)

    # Proceed training from a pretrained model
    if pretrained_model != 'None':
        model = RegularizedPPO.load(pretrained_model)
        model.set_env(env)
        logging.info(f"Loaded pretrained model: {pretrained_model} --- continue training")
        
    # Set up PPO
    else:  
        model = RegularizedPPO(
            learning_rate=linear_schedule(exp_config.ppo.learning_rate),
            reg_policy=human_policy,
            reg_weight=exp_config.reg_weight,  # Regularization weight; lambda
            reg_weight_decay_schedule=exp_config.reg_weight_decay_schedule, # "None" / "linear" / exponential
            env=env,
            n_steps=exp_config.ppo.n_steps,
            policy=LateFusionPolicy,
            batch_size=exp_config.ppo.batch_size,
            ent_coef=exp_config.ppo.ent_coef,
            vf_coef=exp_config.ppo.vf_coef,
            seed=exp_config.seed,  # Seed for the pseudo random generators
            verbose=exp_config.verbose,
            tensorboard_log=f"runs/{run.id}" if run is not None else None,
            device=exp_config.ppo.device,
            env_config=env_config,
            mlp_class=LateFusionNet,
            mlp_config=model_config,
        )        
        
    # Log number of trainable parameters
    policy_params = filter(lambda p: p.requires_grad, model.policy.parameters())
    params = sum(np.prod(p.size()) for p in policy_params)
    exp_config.n_policy_params = params
    logging.info(f"Policy | trainable params: {params:,} \n")

    # Learn
    model.learn(
        **exp_config.learn,
        callback=custom_callback,
    )
    if exp_config.track_wandb:
        run.finish()


if __name__ == "__main__":
    # Run
    typer.run(run_hr_ppo)
