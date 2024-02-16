"""Evaluate policies on different intersections and store results in a dataframe."""

# Dependencies
import logging
import pandas as pd
import pickle
from utils.policies import load_policy
from policy_evaluation import evaluate_policy
from utils.config import load_config
from utils.sb3.reg_ppo import RegularizedPPO
import pickle
from datetime import datetime
from utils.string_utils import datetime_to_str

def gen_and_save_res_df(
    env_config, 
    model_config, 
    num_scenes_to_select_from,
    num_eval_episodes,
    intersection_dicts=None,
    eval_modes={'Self-play': 200, 'Log-replay': 1}, 
    data_sets={'Train': 'data/train_no_tl', 'Test': 'data/valid_no_tl'},
    save_path='evaluation/results',
    deterministic=True,
    ):
    
    """Generate and save a dataframe with the performance of the policies.
    Args:
    - env_config: environment configuration
    - model_config: model configuration
    - num_scenes_to_select_from: number of scenes to select from
    - num_eval_episodes: number of episodes to evaluate
    - intersection_dicts: dictionary with the intersection to scene mapping
    - eval_modes: dictionary with the evaluation modes and the number of controlled agents
    - data_sets: dictionary with the datasets to evaluate
    - save_path: path to save the dataframe
    - deterministic: whether to use deterministic policies (stochastic if False)
    """
    
    # Evaluate HR-PPO policies
    df_ppo_all = pd.DataFrame()
    BASE_PATH = models_config.hr_ppo_models_dir_self_play
    for model_config in models_config.best_overall_models:
        for eval_mode in eval_modes:
            for dataset in data_sets:
                
                logging.info(f'Evaluate model {model_config.name} on {dataset} data using {eval_mode} mode\n')
                if intersection_dicts is not None:
                    logging.info(f'Using intersection dict.\n')
                
                # Load policy
                policy = RegularizedPPO.load(
                    f'{BASE_PATH}/{model_config.name}'
                )
                
                # Set evaluation settings
                num_controlled_agents = eval_modes[eval_mode]
                eval_dataset = data_sets[dataset]
                scene_to_paths_dict = intersection_dicts[dataset] if intersection_dicts is not None else None
                
                if num_controlled_agents >= 50:
                    eval_episodes = num_scenes_to_select_from
                else:
                    eval_episodes = num_eval_episodes
                    
                # Evaluate policy
                df_res = evaluate_policy(
                    env_config=env_config,
                    scene_path_mapping=scene_to_paths_dict,
                    controlled_agents=num_controlled_agents,
                    data_path=eval_dataset,
                    mode='policy',
                    policy=policy,
                    select_from_k_scenes=num_scenes_to_select_from,
                    num_episodes=eval_episodes,
                    deterministic=deterministic,
                )
                
                # Add identifiers
                df_res['Agent'] = model_config.agent
                df_res['Train agent'] = model_config.train_agent
                df_res['Dataset'] = dataset
                df_res['Eval mode'] = eval_mode
                df_res['Reg. weight'] = model_config.reg_weight
                 
                # Store
                df_ppo_all = pd.concat([df_ppo_all, df_res], ignore_index=True)
                

    # Evaluate BC policy
    df_bc = pd.DataFrame()
    for bc_config in models_config.used_human_policy:
        for eval_mode in eval_modes:
            for dataset in data_sets:
                
                logging.info(f'Evaluate model {bc_config.name} on {dataset} data using {eval_mode} mode\n')
                
                # Load policy    
                human_policy = load_policy(
                    data_path=f"{models_config.bc_models_dir}",
                    file_name=bc_config.name, 
                )
                # Set evaluation settings
                num_controlled_agents = eval_modes[eval_mode]
                eval_dataset = data_sets[dataset]
                scene_to_paths_dict = intersection_dicts[dataset] if intersection_dicts is not None else None
                   
                if num_controlled_agents >= 50:
                    eval_episodes = num_scenes_to_select_from
                else:
                    eval_episodes = num_eval_episodes
                    
                    
                # Evaluate policy
                df_res_bc = evaluate_policy(
                    env_config=env_config,
                    scene_path_mapping=scene_to_paths_dict,
                    controlled_agents=num_controlled_agents,
                    data_path=eval_dataset,
                    mode='policy',
                    policy=human_policy,
                    select_from_k_scenes=num_scenes_to_select_from,
                    num_episodes=eval_episodes,
                    deterministic=deterministic,
                )
                
                # Add identifiers
                df_res_bc['Agent'] = bc_config.agent
                df_res_bc['Train agent'] = bc_config.train_agent
                df_res_bc['Dataset'] = dataset
                df_res_bc['Eval mode'] = eval_mode
                df_res_bc['Reg. weight'] = None
                
                # Store
                df_bc = pd.concat([df_bc, df_res_bc], ignore_index=True)

    # Concatenate results and store to csv
    datetime_ = datetime_to_str(dt=datetime.now())
    df_all = pd.concat([df_ppo_all, df_bc], ignore_index=True)
    df_all.to_csv(f'{save_path}/df_agg_performance_{num_scenes_to_select_from}_{datetime_}.csv', index=False)
    
    logging.info(f'Saved at {save_path} stamped with {datetime_} \n')

if __name__ == "__main__":
    
    # Load intersection info
    with open('evaluation/scene_info/info_dict_train_no_tl', 'rb') as handle:
        train_scene_to_paths_dict = pickle.load(handle)
    
    # Load configs
    env_config = load_config("env_config")
    models_config = load_config("model_config") # Trained models
    
    # Generate dataframe
    gen_and_save_res_df(
        num_scenes_to_select_from=100,
        num_eval_episodes=1000,
        env_config=env_config,
        intersection_dicts={'Train': train_scene_to_paths_dict},
        model_config=models_config,   
        data_sets={'Train': 'data/train_no_tl'},
    )