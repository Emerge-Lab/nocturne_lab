"""Obtain dataframes with results for the learned policies."""
import pickle
import re

import numpy as np
import pandas as pd

from evaluation.policy_evaluation import evaluate_policy
from utils.config import load_config
from utils.sb3.reg_ppo import RegularizedPPO

# Settings
CONTROLLED_AGENTS = 100
HR_RL_BASE_PATH = "models/hr_rl/ma_s1000"
NUM_EVAL_EPISODES = 10_000  # All scenes
DATA_PATH = "data_full/valid/"
SELECT_FROM = 12_000

# Load dict with number of intersecting paths per scene
with open("evaluation/scene_info/valid_all_01_23_11_18.pkl", "rb") as handle:
    scene_to_paths_dict = pickle.load(handle)


if __name__ == "__main__":
    # Environment configurations
    env_config = load_config("env_config")
    env_config.data_path = DATA_PATH

    # Select a subset of models
    hr_rl_policy_names = [
        "policy_L0.0_S1000_I606",
        "policy_L0.005_S1000_I601",
        "policy_L0.05_S1000_I582",
        "policy_L0.2_S1000_I559",
    ]

    reg_weights = []
    pattern = r"L(\d+\.\d+)_S"
    for string in hr_rl_policy_names:
        match = re.search(pattern, string)
        if match:
            reg_weights.append(float(match.group(1)))

    df_self_play = pd.DataFrame()

    for model_name, reg_weight in zip(hr_rl_policy_names, reg_weights, strict=False):
        # Load RL policy
        policy = RegularizedPPO.load(f"{HR_RL_BASE_PATH}/{model_name}")

        # Evaluate
        df_res_sp = evaluate_policy(
            env_config=env_config,
            scene_path_mapping=scene_to_paths_dict,
            controlled_agents=CONTROLLED_AGENTS,
            data_path=DATA_PATH,
            mode="policy",
            policy=policy,
            select_from_k_scenes=SELECT_FROM,
            num_episodes=NUM_EVAL_EPISODES,
        )

        # Add identifiers
        df_res_sp["reg_weight"] = np.repeat(reg_weight, len(df_res_sp))
        df_res_sp["model_name"] = np.repeat(model_name, len(df_res_sp))

        # Store
        df_self_play = pd.concat([df_self_play, df_res_sp], ignore_index=True)

        # Save
        df_self_play.to_csv(
            f"evaluation/results/df_compatible_hr_ppo_self_play_test_0124_{NUM_EVAL_EPISODES}.csv", index=False
        )
