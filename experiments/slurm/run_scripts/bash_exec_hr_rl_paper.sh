
sweep_name_values=( hr_rl_paper )
pretrained_model_values=( policy_L0.01_S1000_I2250 policy_L0.025_S1000_I2250 policy_L0.05_S1000_I2250 )
human_policy_name_values=( human_policy_D99_S1000_01_29_11_53.pt )
ent_coef_values=( 0.001 0.005 )
total_timesteps_values=( 30000000 )
num_controlled_veh_values=( 20 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
pretrained_model=${pretrained_model_values[$(( trial % ${#pretrained_model_values[@]} ))]}
trial=$(( trial / ${#pretrained_model_values[@]} ))
human_policy_name=${human_policy_name_values[$(( trial % ${#human_policy_name_values[@]} ))]}
trial=$(( trial / ${#human_policy_name_values[@]} ))
ent_coef=${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}
trial=$(( trial / ${#ent_coef_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}
trial=$(( trial / ${#total_timesteps_values[@]} ))
num_controlled_veh=${num_controlled_veh_values[$(( trial % ${#num_controlled_veh_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate
python experiments/hr_rl/run_hr_ppo_cli.py --sweep-name=${sweep_name} --pretrained-model=${pretrained_model} --human-policy-name=${human_policy_name} --ent-coef=${ent_coef} --total-timesteps=${total_timesteps} --num-controlled-veh=${num_controlled_veh}
