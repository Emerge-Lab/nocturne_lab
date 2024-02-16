# Dependencies
import glob
import pandas as pd
import os
from utils.eval import EvaluatePolicy
from utils.policies import load_policy
from utils.config import load_config
from utils.sb3.reg_ppo import RegularizedPPO
import logging

# Load config files
env_config = load_config("env_config")
exp_config = load_config("exp_config")
video_config = load_config("video_config")
models_config = load_config("model_config") # Trained models

# Evaluation settings
EVAL_MODE = 'Self-play'
DATASET = 'data/train_no_tl'
NUM_EVAL_EPISODES = 100 # Number of episodes to evaluate on
DETERMINISTIC = True 
SELECT_FROM = 100 # Number of test scenes
MAX_CONTROLLED_AGENTS = 200 # All agents are controlled

# Dataset
env_config.data_path = DATASET
test_file_paths = glob.glob(f"{env_config.data_path}" + "/tfrecord*")
test_eval_files = sorted([os.path.basename(file) for file in test_file_paths])[:SELECT_FROM]

if __name__ == '__main__':  

    # Evaluate PPO models
    df_ppo_all = pd.DataFrame()

    BASE_PATH = models_config.hr_ppo_models_dir_self_play

    for model_config in models_config.best_overall_models:
        logging.info(f'Evaluate model {model_config.name} on {DATASET} data using {EVAL_MODE} mode\n')
        
        # Load policy
        policy = RegularizedPPO.load(
            f'{BASE_PATH}/{model_config.name}'
        )
        
        # Evaluate policy
        evaluator = EvaluatePolicy(
            env_config=env_config,
            policy=policy,
            eval_files=test_eval_files,
            deterministic=DETERMINISTIC,
            reg_coef=model_config.reg_weight,
        )
        
        df_res = evaluator._get_scores()
                
        # Add identifiers
        df_res['Agent'] = model_config.agent
        df_res['Train agent'] = model_config.train_agent
        df_res['Dataset'] = DATASET
        df_res['Eval mode'] = EVAL_MODE
        
        # Store
        df_ppo_all = pd.concat([df_ppo_all, df_res], ignore_index=True)
            

    # Evaluate BC model
    df_bc = pd.DataFrame()
    
    for bc_config in models_config.used_human_policy:   
        logging.info(f'Evaluate model {bc_config.name} on {DATASET} data using {EVAL_MODE} mode\n')
        
        # Load policy    
        human_policy = load_policy(
            data_path=f"{models_config.bc_models_dir}",
            file_name=bc_config.name, 
        )
   
        # Evaluate policy
        evaluator = EvaluatePolicy(
            env_config=env_config,
            policy=human_policy,
            eval_files=test_eval_files,
            deterministic=DETERMINISTIC,
            reg_coef=None,
        )
        
        df_res_bc = evaluator._get_scores()
        
        # Add identifiers
        df_res_bc['Agent'] = bc_config.agent
        df_res_bc['Train agent'] = bc_config.train_agent
        df_res_bc['Dataset'] = DATASET
        df_res_bc['Eval mode'] = EVAL_MODE
        
        # Store
        df_bc = pd.concat([df_bc, df_res_bc], ignore_index=True)
        
    
    # Concatenate results and store to csv
    df_all = pd.concat([df_ppo_all, df_bc], ignore_index=True)
    df_all.to_csv(f'evaluation/results/df_trade_off.csv', index=False)