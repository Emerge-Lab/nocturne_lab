
sweep_name_values=( hr_rl_little_data )
human_policy_names_values=( human_policy_D99_S10_UNFILTERED_01_29_15_57.pt human_policy_D99_S57_UNFILTERED_01_29_15_31.pt human_policy_D99_S283_UNFILTERED_01_29_16_38.pt human_policy_D99_S13_FILTERED_01_29_14_46.pt human_policy_D99_S104_FILTERED_01_29_14_02.pt human_policy_D99_S436_FILTERED_01_29_15_15.pt )
seed_values=( 42 8 )
reg_weight_values=( 0.02 0.025 )
total_timesteps_values=( 50000000 )
num_controlled_veh_values=( 20 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
human_policy_names=${human_policy_names_values[$(( trial % ${#human_policy_names_values[@]} ))]}
trial=$(( trial / ${#human_policy_names_values[@]} ))
seed=${seed_values[$(( trial % ${#seed_values[@]} ))]}
trial=$(( trial / ${#seed_values[@]} ))
reg_weight=${reg_weight_values[$(( trial % ${#reg_weight_values[@]} ))]}
trial=$(( trial / ${#reg_weight_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}
trial=$(( trial / ${#total_timesteps_values[@]} ))
num_controlled_veh=${num_controlled_veh_values[$(( trial % ${#num_controlled_veh_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate
python experiments/hr_rl/run_hr_ppo_cli.py --sweep-name=${sweep_name} --human-policy-name=${human_policy_names} --seed=${seed} --reg-weight=${reg_weight} --total-timesteps=${total_timesteps} --num-controlled-veh=${num_controlled_veh}
