import glob
import logging
import os
import pickle
from itertools import combinations
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from shapely.geometry import LineString, Point, MultiPoint
from tqdm import tqdm

from nocturne.envs.base_env import BaseEnv
from utils.config import load_config
from utils.string_utils import datetime_to_str


def calculate_squared_distances(path, point):
    return np.sum((path - point)**2, axis=1)

def find_closest_point_index(path, point):
    distances = calculate_squared_distances(path, point)
    return np.argmin(distances)

def initialize_simulation(env, filename=None):
    try:
        return env.reset(filename), True
    except ValueError:
        return (np.zeros((1, 90, 2)), {}), False

def execute_step(env, mode):
    action_dict = {}
    if mode == "expert":
        for vehicle in env.controlled_vehicles:
            vehicle.expert_control = True
    else:
        for vehicle in env.controlled_vehicles:
            vehicle.expert_control = False
    return env.step(action_dict)

def step_through_scene(env, mode, filename=None, num_steps=90):
    obs_dict, simulation_valid = initialize_simulation(env, filename)
    if not simulation_valid:
        return obs_dict  # Returns the empty array and dict if initialization failed

    num_agents = len(env.controlled_vehicles)
    agent_positions, agent_speed = prepare_agent_metrics(num_agents, num_steps)
    goal_achieved, veh_edge_collision, veh_veh_collision = np.zeros(num_agents), np.zeros(num_agents), np.zeros(num_agents)

    agent_ids, agent_id_to_idx_dict, last_info_dicts, dead_agent_ids = prepare_agent_dicts(env)

    for timestep in range(num_steps):
        if mode == "expert":
            record_agent_positions_and_speeds(env, agent_positions, agent_speed, timestep, dead_agent_ids, agent_id_to_idx_dict)

        obs_dict, rew_dict, done_dict, info_dict = execute_step(env, mode)
        update_dead_agents(done_dict, dead_agent_ids, last_info_dicts, info_dict)

        if done_dict["__all__"]:
            break

    update_final_metrics(agent_ids, agent_id_to_idx_dict, last_info_dicts, goal_achieved, veh_edge_collision, veh_veh_collision)
    return agent_positions, agent_id_to_idx_dict

def prepare_agent_metrics(num_agents, num_steps):
    agent_positions = np.full((num_agents, num_steps, 2), np.nan)
    agent_speed = np.full((num_agents, num_steps), np.nan)
    return agent_positions, agent_speed

def prepare_agent_dicts(env):
    agent_ids = np.sort([vehicle.id for vehicle in env.controlled_vehicles])
    agent_id_to_idx_dict = {id: index for index, id in enumerate(agent_ids)}
    last_info_dicts = {id: {} for id in agent_ids}
    dead_agent_ids = []
    return agent_ids, agent_id_to_idx_dict, last_info_dicts, dead_agent_ids

def record_agent_positions_and_speeds(env, agent_positions, agent_speed, timestep, dead_agent_ids, agent_id_to_idx_dict):
    for vehicle in env.controlled_vehicles:
        if vehicle.id not in dead_agent_ids:
            idx = agent_id_to_idx_dict[vehicle.id]
            agent_positions[idx, timestep] = np.array([vehicle.position.x, vehicle.position.y])
            agent_speed[idx, timestep] = vehicle.speed

def update_dead_agents(done_dict, dead_agent_ids, last_info_dicts, info_dict):
    for agent_id, is_done in done_dict.items():
        if is_done and agent_id not in dead_agent_ids and agent_id != "__all__":
            dead_agent_ids.append(agent_id)
            last_info_dicts[agent_id] = info_dict[agent_id].copy()

def update_final_metrics(agent_ids, agent_id_to_idx_dict, last_info_dicts, goal_achieved, veh_edge_collision, veh_veh_collision):
    for agent_id in agent_ids:
        idx = agent_id_to_idx_dict[agent_id]
        goal_achieved[idx] += last_info_dicts[agent_id].get("goal_achieved", 0)
        veh_edge_collision[idx] += last_info_dicts[agent_id].get("veh_edge_collision", 0)
        veh_veh_collision[idx] += last_info_dicts[agent_id].get("veh_veh_collision", 0)

def plot_lines(line1, line2, title="Line Plot"):
    fig, ax = plt.subplots(figsize=(3, 3))
    ax.set_title(title)
    ax.plot(*line1.xy, label="Line 1")
    ax.plot(*line2.xy, label="Line 2")
    ax.legend()
    plt.show()


def process_vehicle_combinations(expert_trajectories, vehicle_id_dict, allowed_time_window):
    
    veh_id_to_intersecting_paths_dict = {veh_id: 0 for veh_id in vehicle_id_dict}
    veh_id_to_time_diff = {veh_id: 500 for veh_id in vehicle_id_dict}
    
    for veh_i, veh_j in combinations(vehicle_id_dict, 2):
        
        # Get paths
        path_veh_i, path_veh_j = expert_trajectories[vehicle_id_dict[veh_i], :, :], expert_trajectories[vehicle_id_dict[veh_j], :, :]
        nonnan_ids = ~np.logical_or(np.isnan(path_veh_i), np.isnan(path_veh_j)).any(axis=1)   
        path_veh_i = path_veh_i[nonnan_ids]
        path_veh_j = path_veh_j[nonnan_ids]
        
        if nonnan_ids.sum() > 1:
            line1, line2 = LineString(path_veh_i), LineString(path_veh_j)
            
            if line1.intersects(line2):
                intersection = line1.intersection(line2)
                points = [intersection] if isinstance(intersection, Point) else list(intersection.geoms) if isinstance(intersection, MultiPoint) else []
                
                # Add the minimum temporal distance between the two paths
                step_dists = []
                for point in points:
                    intersection_point = np.array([point.x, point.y])
                    closest_index_i = find_closest_point_index(path_veh_i, intersection_point)
                    closest_index_j = find_closest_point_index(path_veh_j, intersection_point)
                    step_dists.append(abs(closest_index_i - closest_index_j))
                    
                # Increment the number of intersecting paths for both vehicles
                if min(step_dists) < allowed_time_window:
                    veh_id_to_intersecting_paths_dict[veh_i] += 1
                    veh_id_to_intersecting_paths_dict[veh_j] += 1 
                
                if step_dists:
                    min_step_dist = min(step_dists)
                    # Always store the minimum time difference
                    if veh_id_to_time_diff[veh_i] != 500 or veh_id_to_time_diff[veh_j] != 500:
                        veh_id_to_time_diff[veh_i] = min(veh_id_to_time_diff[veh_i], min_step_dist)
                        veh_id_to_time_diff[veh_j] = min(veh_id_to_time_diff[veh_j], min_step_dist)
                    else:
                        veh_id_to_time_diff[veh_i] = min_step_dist
                        veh_id_to_time_diff[veh_j] = min_step_dist
                        
    return veh_id_to_intersecting_paths_dict, veh_id_to_time_diff

def get_intersecting_path_dict(env, traffic_scenes, allowed_time_window=50, save_dict=True, filename="intersecting_paths.pkl"):
    """Main function to obtain the number of intersecting paths per scene and agent id."""
    scene_intersecting_paths_dict = {}
    for traffic_scene in tqdm(traffic_scenes):
        expert_trajectories, vehicle_id_dict = step_through_scene(env, mode="expert", filename=traffic_scene)
        if not vehicle_id_dict:
            continue
        
        veh_id_to_intersecting_paths_dict, veh_id_to_time_diff = process_vehicle_combinations(
            expert_trajectories, 
            vehicle_id_dict, 
            allowed_time_window,
        )
        
        # Map large numbers to np.nan 
        t_diff = list(veh_id_to_time_diff.values())
        t_diff_nan = [np.nan if x == 500 else x for x in t_diff]
        
        scene_info = {
            "veh_id": list(veh_id_to_intersecting_paths_dict.keys()),
            "intersecting_paths": list(veh_id_to_intersecting_paths_dict.values()),
            "min_step_diff": t_diff_nan,
            "total_intersecting_paths": sum(veh_id_to_intersecting_paths_dict.values())
        }
        scene_intersecting_paths_dict[traffic_scene] = scene_info

    if save_dict:
        with open(filename, 'wb') as f:
            pickle.dump(scene_intersecting_paths_dict, f)
    
    return scene_intersecting_paths_dict

if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger().setLevel(logging.INFO)
    
    # Load config
    env_config = load_config("env_config")
    env_config.max_num_vehicles = 500
    
    # Set data path for which we want to obtain the number of intersecting paths
    env_config.data_path = "data/train_no_tl/"
    
    # Scenes on which to evaluate the models
    # Make sure file order is fixed so that we evalu on the same f iles used for training
    file_paths = glob.glob(f"{env_config.data_path}" + "/tfrecord*")
    files = sorted([os.path.basename(file) for file in file_paths])
    
    logging.info(f'num_scenes: {len(files)}')
    
    # Make env
    env = BaseEnv(env_config)
    
    # Create dictionary with number of intersecting paths per scene and agent id
    int_dict = get_intersecting_path_dict(
        env=env, 
        traffic_scenes=files, 
        save_dict=True,
        filename=f'info_dict_train_no_tl'
    )
    