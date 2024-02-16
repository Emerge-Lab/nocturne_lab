# Dependencies
import logging
import pandas as pd
import pickle
from utils.policies import load_policy
from evaluation.policy_evaluation import evaluate_policy
from utils.config import load_config
from utils.sb3.reg_ppo import RegularizedPPO
import pickle
import numpy as np

# Load intersection info
with open('evaluation/scene_info/valid_all_01_23_11_18.pkl', 'rb') as handle:
    valid_scene_to_paths_dict = pickle.load(handle)

"""Evaluation settings."""
NUM_EVAL_EPISODES = 10_000 # Number of episodes to evaluate on
DETERMINISTIC = True 
SELECT_FROM = 10_000 # Number of test scenes
MAX_CONTROLLED_AGENTS = 200 # All agents are controlled
METRICS = ['goal_rate', 'off_road', 'veh_veh_collision']

TRAIN_DATA_PATH = 'data_full/train'
TEST_DATA_PATH = 'data_full/valid'

DATASETS = {'Test': TEST_DATA_PATH}
H_COLUMNS = ['Agent', 'Train agent', 'Dataset', 'Eval mode']
METRICS = ['Off-Road Rate (\%)', 'Collision Rate (\%)', 'Goal Rate (\%)']

if __name__ == "__main__":
    
    # Load config files
    env_config = load_config("env_config")
    exp_config = load_config("exp_config")
    video_config = load_config("video_config")
    models_config = load_config("model_config") # Trained models
    
    # Evaluate HR-PPO policies
    df_hr_ppo = pd.DataFrame()

    for model_config in models_config.small_data_hr_ppo:
        
        print(f'model_name: {model_config.name} \n')
        
        ppo_trained = RegularizedPPO.load(
        f'{models_config.small_data_hr_ppo_models_dir}/{model_config.name}')
        
        # Evaluate policy
        df_res_hr_ppo = evaluate_policy(
            env_config=env_config,
            controlled_agents=1,
            data_path=TEST_DATA_PATH,
            mode='policy',
            policy=ppo_trained,
            select_from_k_scenes=SELECT_FROM,
            num_episodes=NUM_EVAL_EPISODES,
            scene_path_mapping=valid_scene_to_paths_dict,
        )
        
        # Add identifiers
        df_res_hr_ppo['model_name'] = np.repeat(model_config.name, len(df_res_hr_ppo))
        df_res_hr_ppo['data_size'] = np.repeat(model_config.samples, len(df_res_hr_ppo))
        df_res_hr_ppo['Agent'] = model_config.agent

        # Store
        df_hr_ppo = pd.concat([df_hr_ppo, df_res_hr_ppo], ignore_index=True)
                
    # # Evaluate BC policy
    # df_bc = pd.DataFrame()
    # for bc_config in models_config.used_human_policy:
                       
    #     logging.info(f'Evaluate model {bc_config.name} on {dataset} data using {eval_mode} mode\n')
        
    #     # Load policy    
    #     human_policy = load_policy(
    #         data_path=f"{models_config.bc_models_dir}",
    #         file_name=bc_config.name, 
    #     )
    #     # Set evaluation settings
    #     num_controlled_agents = EVAL_MODES[eval_mode]
    #     eval_dataset = DATASETS[dataset]
    #     scene_to_paths_dict = INTERSECTION_DICTS[dataset]
            
    #     # Evaluate policy
    #     df_res_bc = evaluate_policy(
    #         env_config=env_config,
    #         scene_path_mapping=scene_to_paths_dict,
    #         controlled_agents=num_controlled_agents,
    #         data_path=eval_dataset,
    #         mode='policy',
    #         policy=human_policy,
    #         select_from_k_scenes=SELECT_FROM,
    #         num_episodes=NUM_EVAL_EPISODES,
    #     )
        
    #     # Add identifiers
    #     df_res_bc['Agent'] = bc_config.agent
    #     df_res_bc['Train agent'] = bc_config.train_agent
    #     df_res_bc['Dataset'] = dataset
    #     df_res_bc['Eval mode'] = eval_mode
    #     df_res_bc['Reg. weight'] = None
        
    #     # Store
    #     df_bc = pd.concat([df_bc, df_res_bc], ignore_index=True)
    
    
    # Concatenate results and store to csv
    #df_all = pd.concat([df_ppo_all, df_bc], ignore_index=True)
    df_hr_ppo.to_csv(f'evaluation/paper/df_small_data_0131.csv', index=False)
    print(f'Saved!')