"""Evaluate a policy on a set of scenes."""
import glob
import logging
import os

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config
from utils.policies import load_policy

logging.basicConfig(level=logging.DEBUG)

def evaluate_policy(
    env_config,
    mode,
    controlled_agents,
    data_path,
    select_from_k_scenes=100,
    num_episodes=100,
    scene_path_mapping=None,
    policy=None,
    deterministic=True,
    traffic_files=None,
    use_av_only=False,
):
    """Evaluate a policy on a set of scenes.

    Args:
    ----
        env_config: Environment configuration.
        mode: Mode of evaluation. Can be one of the following:
            - policy: Evaluate a policy.
            - expert_replay: Replay expert actions.
            - cont_expert_act_replay: Replay continuous expert actions.
            - disc_expert_act_replay: Replay discretized expert actions.
        controlled_agents: Number of agents to control.
        data_path: Path to data.
        select_from_k_scenes: Number of scenes to select from.
        num_episodes: Number of episodes to run; how many times to reset the environment to a new scene.
        scene_path_mapping (dict, optional): Mapping from scene to dict with the number of intersecting
            paths of that scene.
        policy (optional): Policy to evaluate.
        deterministic (optional): Whether to use a deterministic policy.

    Raises:
    ------
        ValueError: If scene_path_mapping is provided, if scene is not found in scene_path_mapping.

    Returns:
    -------
        df: performance per scene and vehicle id.
    """
    # Configurations
    env_config.max_num_vehicles = controlled_agents
    env_config.data_path = data_path
    env_config.use_av_only = use_av_only

    # Set which files to use
    env_config.num_files = select_from_k_scenes

    # Make env
    env = BaseEnv(env_config)

    # Storage
    df = pd.DataFrame()

    # Run
    for _ in tqdm(range(num_episodes)):
        
        if traffic_files is not None:
            # Reset to a new scene
            obs_dict = env.reset(
                filename=np.random.choice(traffic_files),
            )

        else:
            obs_dict = env.reset()

        agent_ids = [veh.id for veh in env.controlled_vehicles]
        dead_agent_ids = []
        veh_id_to_idx = {veh_id: idx for idx, veh_id in enumerate(agent_ids)}
        last_info_dicts = {agent_id: {} for agent_id in agent_ids}
        
        # Storage
        goal_achieved = np.zeros(len(agent_ids))
        off_road = np.zeros(len(agent_ids))
        veh_veh_coll = np.zeros(len(agent_ids))

        for _ in range(env_config.episode_length):

            # Get actions
            action_dict = {}
            
            if mode == "expert_replay":
                # Set all controlled vehicles to be experts
                for veh in env.controlled_vehicles:
                    veh.expert_control = True
                    
            if mode == "random":
                # Take random actions
                action_dict = {
                    veh.id: env.action_space.sample() 
                    for veh in env.controlled_vehicles
                    if veh.id not in dead_agent_ids
                }

            if mode == "cont_expert_act_replay":  
                # Use continuous expert actions
                for veh_obj in env.controlled_vehicles:
                    # Get (continuous) expert action
                    expert_action = env.scenario.expert_action(veh_obj, env.step_num)
                    action_dict[veh_obj.id] = expert_action
                    
                    # if expert_action is None or expert_action.steering != expert_action.steering:
                    #     print(f"None at {env.step_num} for veh {veh_obj.id} in {env.file} \n")
                        
            if mode == "disc_expert_act_replay":  
                
                # Use discretized expert actions
                for veh_obj in env.controlled_vehicles:
                    # Get (continuous) expert action
                    expert_action = env.scenario.expert_action(veh_obj, env.step_num)

                    # Discretize expert action
                    # if expert_action is None or expert_action.steering != expert_action.steering:
                    #     logging.info(f"None at {env.step_num} for veh {veh_obj.id} in {env.file} \n")

                    if expert_action is not None:
                        expert_accel, expert_steering, _ = expert_action.numpy()

                        # Map actions to nearest grsid indices and joint action
                        acc_grid_idx = np.argmin(np.abs(env.accel_grid - expert_accel))
                        ste_grid_idx = np.argmin(np.abs(env.steering_grid - expert_steering))

                        expert_action_idx = env.actions_to_idx[
                            env.accel_grid[acc_grid_idx],
                            env.steering_grid[ste_grid_idx],
                        ][0]

                        action_dict[veh_obj.id] = expert_action_idx

            if mode == "policy" and policy is not None:
                for agent_id in obs_dict:
                    # Get observation
                    obs = torch.from_numpy(obs_dict[agent_id]).unsqueeze(dim=0)

                    # Get action
                    action, _ = policy.predict(obs, deterministic=deterministic)
                    action_dict[agent_id] = int(action[0])

            elif mode == "policy" and policy is None:
                raise ValueError("Policy is not given. Please provide a policy.")

            # Take a step
            obs_dict, rew_dict, done_dict, info_dict = env.step(action_dict)

            for agent_id, is_done in done_dict.items():
                if is_done and agent_id not in dead_agent_ids:
                    dead_agent_ids.append(agent_id)
                    # Store agents' last info dict
                    last_info_dicts[agent_id] = info_dict[agent_id].copy()

            if done_dict["__all__"]:  # If all agents are done
                for agent_id in agent_ids:
                    agent_idx = veh_id_to_idx[agent_id]
                    veh_veh_coll[agent_idx] += last_info_dicts[agent_id]["veh_veh_collision"] * 1
                    off_road[agent_idx] += last_info_dicts[agent_id]["veh_edge_collision"] * 1
                    goal_achieved[agent_idx] += last_info_dicts[agent_id]["goal_achieved"] * 1

                # Get scene info
                if scene_path_mapping is not None:
                    if str(env.file) in scene_path_mapping.keys():
                        control_veh_int_paths = np.zeros(len(agent_ids))
                        control_veh_step_diff = np.full(len(agent_ids), fill_value=np.nan)

                        # Obtain the number of intersecting paths for every vehicle
                        for agent_id in agent_ids:
                            if agent_id in scene_path_mapping[str(env.file)]["veh_id"]:
                                agent_idx = veh_id_to_idx[agent_id]
                                control_veh_idx = scene_path_mapping[str(env.file)]["veh_id"].index(agent_id)
                                total_int_paths_in_scene = (
                                    sum(scene_path_mapping[str(env.file)]["intersecting_paths"]) / 2
                                )
                                control_veh_int_path = scene_path_mapping[str(env.file)]["intersecting_paths"][
                                    control_veh_idx
                                ]
                                control_veh_min_step = scene_path_mapping[str(env.file)]["min_step_diff"][control_veh_idx]
                                total_vehs_in_scene = len(scene_path_mapping[str(env.file)]["veh_id"])

                                control_veh_int_paths[agent_idx] = control_veh_int_path
                                control_veh_step_diff[agent_idx] = control_veh_min_step

                    else:
                        control_veh_int_paths = np.zeros(len(agent_ids))
                        total_int_paths_in_scene = 0
                        logging.info(f"Scene {env.file} not found in scene_path_mapping")

                    df_scene_i = pd.DataFrame(
                        {
                            "scene_id": env.file,
                            "veh_id": agent_ids,
                            "veh_int_paths": control_veh_int_paths,
                            "min_step_diff": control_veh_step_diff,
                            "tot_int_paths": total_int_paths_in_scene,
                            "num_total_vehs": total_vehs_in_scene,
                            "goal_rate": goal_achieved,
                            "off_road": off_road,
                            "veh_veh_collision": veh_veh_coll,
                        },
                        index=list(range(len(agent_ids))),
                    )
            
                else:  # If we don't have any scene-specific info
                    df_scene_i = pd.DataFrame(
                        {
                            "scene_id": env.file,
                            "veh_id": agent_ids,
                            "num_total_vehs": len(agent_ids),
                            "goal_rate": goal_achieved,
                            "off_road": off_road,
                            "veh_veh_collision": veh_veh_coll,
                        },
                        index=list(range(len(agent_ids))),
                    )

                # Append to df
                df = pd.concat([df, df_scene_i], ignore_index=True)

                break  # Proceed to next scene

    return df


if __name__ == "__main__":
    
    # Configurations
    CONTROLLED_AGENTS = 1
    SELECT_FROM_K_SCENES = 2000
    NUM_EPISODES = 500
    USE_AV_ONLY = False

    # Load environment config
    env_config = load_config("env_config")

    # Set data path to NEW scenes (with is_av flag)
    env_config.data_path = "data/train_no_tl"

    # EXPERT-TELEPORT
    df_expert_replay = evaluate_policy(
        env_config=env_config,
        controlled_agents=CONTROLLED_AGENTS,
        data_path=env_config.data_path,
        mode="expert_replay",
        select_from_k_scenes=SELECT_FROM_K_SCENES,
        num_episodes=NUM_EPISODES,
        use_av_only=USE_AV_ONLY,
    )
    
    logging.info(f'--- Results: EXPERT-TELEPORT ---')
    print(df_expert_replay[["goal_rate", "off_road", "veh_veh_collision"]].mean())
    
    # # RANDOM
    # df_random = evaluate_policy(
    #     env_config=env_config,
    #     controlled_agents=CONTROLLED_AGENTS,
    #     data_path=env_config.data_path,
    #     mode="random",
    #     select_from_k_scenes=SELECT_FROM_K_SCENES,
    #     num_episodes=NUM_EPISODES,
    #     use_av_only=USE_AV_ONLY,
    # )
    
    # logging.info(f'--- Results: RANDOM ACTIONS ---')
    # print(df_random[["goal_rate", "off_road", "veh_veh_collision"]].mean())
    
    # EXPERT-ACTIONS
    df_expert_replay_actions = evaluate_policy(
        env_config=env_config,
        controlled_agents=CONTROLLED_AGENTS,
        data_path=env_config.data_path,
        mode="cont_expert_act_replay",
        select_from_k_scenes=SELECT_FROM_K_SCENES,
        num_episodes=NUM_EPISODES,
        use_av_only=USE_AV_ONLY,
    )
    
    logging.info(f'--- Results: EXPERT-TRAJECTORY ACTIONS ---')
    print(df_expert_replay_actions[["goal_rate", "off_road", "veh_veh_collision"]].mean())
    
    # Discretized EXPERT-ACTIONS
    df_expert_replay_actions_disc = evaluate_policy(
        env_config=env_config,
        controlled_agents=CONTROLLED_AGENTS,
        data_path=env_config.data_path,
        mode="disc_expert_act_replay",
        select_from_k_scenes=SELECT_FROM_K_SCENES,
        num_episodes=NUM_EPISODES,
        use_av_only=USE_AV_ONLY,
    )
    
    logging.info(f'--- Results: EXPERT-TRAJECTORY DISCRETIZED ACTIONS ste: {env_config.steering_discretization} | acc: {env_config.accel_discretization} ---')
    print(df_expert_replay_actions_disc[["goal_rate", "off_road", "veh_veh_collision"]].mean())

    # BEHAVIORAL CLONING
    # human_policy = load_policy(
    #     data_path="models/il/",
    #     file_name="human_policy_D403_S500_02_08_21_30", 
    # )
    
    # df_bc = evaluate_policy(
    #     env_config=env_config,
    #     controlled_agents=1,
    #     data_path=env_config.data_path,
    #     mode="policy",
    #     policy=human_policy,
    #     select_from_k_scenes=SELECT_FROM_K_SCENES,
    #     num_episodes=NUM_EPISODES,
    #     use_av_only=True,
    # )
    
    # logging.info(f'--- Results: BEHAVIORAL CLONING ---')
    # print(df_bc[["goal_rate", "off_road", "veh_veh_collision"]].mean())

