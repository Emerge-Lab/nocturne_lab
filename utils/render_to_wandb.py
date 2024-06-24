# Dependencies
import pandas as pd
import warnings
from nocturne.envs.base_env import BaseEnv
from pyvirtualdisplay import Display
import pickle
import imageio
import wandb
import pandas as pd
import seaborn as sns
import logging
import numpy as np
import matplotlib.pyplot as plt
import torch
from utils.configs import load_config
from utils.sb3.reg_ppo import RegularizedPPO
from utils.policies import load_policy

warnings.filterwarnings("ignore")
plt.set_loglevel('WARNING')


if __name__ == '__main__':

    # Trained policies
    model_config = load_config('models_main_paper')
    # Environment settings
    env_config = load_config('env_config')
    env_config.data_path = 'data/train_no_tl'

    # IL policy
    human_policy = load_policy(
        data_path=f'{model_config.bc_models_dir}',
        file_name=f'{model_config["used_human_policy"][0].name}',
    )

    # RL policies
    best_ppo = model_config.best_overall_models[0].name
    best_hr_ppo = model_config.best_overall_models[2].name

    ppo_policy = RegularizedPPO.load(
        f'{model_config.hr_ppo_models_dir_self_play}/{best_ppo}'
    )

    hr_ppo_policy = RegularizedPPO.load(
        f'{model_config.hr_ppo_models_dir_self_play}/{best_hr_ppo}'
    )

    logging.info(f'\n \n Using PPO policy: {best_ppo} and HR-PPO policy: {best_hr_ppo}\n')
    
    # Settings
    NUM_INTERSECTING_PATHS = 3
    GOAL_RATE = 1
    MAX_TOTAL_VEHICLES = 8

    # initialize wandb
    run = wandb.init(
        group=f'{NUM_INTERSECTING_PATHS}_int_paths',
        project="render",
        reinit=True,
        config={
            "env_config": env_config,
            "model_config": model_config,
        },
    )
    

    # Load df    
    df = pd.read_csv('evaluation/results/df_paper_agg_performance_03_07_14_02_200train_scenes_10_000_test_final.csv')
    df_scenes = df[['scene_id', 'num_total_vehs', 'veh_int_paths', 'tot_int_paths', 'goal_rate', 'off_road', 'veh_veh_collision', 'Dataset']]
        
    df_scenes = df_scenes[df_scenes['Dataset'] == 'Train']
    df_scenes = df_scenes[(df_scenes['veh_int_paths'] == NUM_INTERSECTING_PATHS) & (df_scenes['goal_rate'] == GOAL_RATE & (df_scenes['num_total_vehs'] < MAX_TOTAL_VEHICLES))]

    NUM_SAMPLES = 12
    # Single or multi-agent
    env_config.max_num_vehicles = 1
    env_config.use_av_only = True

    env = BaseEnv(env_config)

    for i in range(NUM_SAMPLES):
        
        random_scene = df_scenes.sample(n=1)
        scene_name = str(random_scene.scene_id.values[0]) 
        
        for policy_type in ["PPO", "HR-PPO"]:
            
            print(f'Scene {i}: {scene_name} -- Policy: {policy_type}')
            
            if policy_type == "PPO":
                policy = ppo_policy
            elif policy_type == "HR-PPO":
                policy = hr_ppo_policy
            
            # Reset environment
            obs_dict = env.reset(filename=scene_name)
            frames = []

            for time_step in range(env_config.episode_length):
                
                # Get action
                action_dict = {}
                for agent_id in obs_dict:
                    # Get observation
                    obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)

                    # Get action
                    action, _ = policy.predict(obs, deterministic=False)
                    action_dict[agent_id] = int(action)
                
                if time_step % 2 == 0:
                    with Display(backend="xvfb") as disp:
                        # Render the scene
                        render_scene = env.scenario.getImage(
                            img_width=1600,
                            img_height=1600,
                            draw_target_positions=True,
                            padding=50.0,
                            sources=[env.controlled_vehicles[0]],
                            view_width=100,
                            view_height=100,
                            rotate_with_source=False,
                        )
                        # Append to frames
                        frames.append(render_scene.T)
                        
    
                # Step
                obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)
                
                if done_dict['__all__'] or time_step == env_config.episode_length - 1:
                    with Display(backend="xvfb") as disp:
                        # Log trajectory
                        last_frame = env.scenario.getImage(
                                img_width=1600,
                                img_height=1600,
                                draw_target_positions=True,
                                padding=50.0,
                                sources=[env.controlled_vehicles[0]],
                                view_width=100,
                                view_height=100,
                                rotate_with_source=False,
                            )
                
                    # Log image and movie
                    trajectory = wandb.Image(np.array(last_frame), caption=f"{policy}")
                    
                    # Movie
                    movie_frames = np.array(frames, dtype=np.uint8)
                    video_key = f"{scene_name}_Policy_{policy_type}" 
                    
                    wandb.log({f"figures/{scene_name}_Policy_{policy_type}_trajectory": trajectory})
                    wandb.log(
                        {
                            f"videos/{video_key}": wandb.Video(movie_frames, fps=3, caption=f"Scene #{scene_name}_Policy_{policy_type}_int_paths_{NUM_INTERSECTING_PATHS}") ,
                        },
                    )
                
                    break 
                
    run.finish()