
sweep_name_values=( randomized_goals )
lr_values=( 0.0003 )
ent_coef_values=( 0.001 0.002 )
reg_weight_values=( 0.0 0.005 0.01 0.025 0.05 0.2 )
total_timesteps_values=( 60000000 )
num_controlled_veh_values=( 1 50 )

trial=${SLURM_ARRAY_TASK_ID}
sweep_name=${sweep_name_values[$(( trial % ${#sweep_name_values[@]} ))]}
trial=$(( trial / ${#sweep_name_values[@]} ))
lr=${lr_values[$(( trial % ${#lr_values[@]} ))]}
trial=$(( trial / ${#lr_values[@]} ))
ent_coef=${ent_coef_values[$(( trial % ${#ent_coef_values[@]} ))]}
trial=$(( trial / ${#ent_coef_values[@]} ))
reg_weight=${reg_weight_values[$(( trial % ${#reg_weight_values[@]} ))]}
trial=$(( trial / ${#reg_weight_values[@]} ))
total_timesteps=${total_timesteps_values[$(( trial % ${#total_timesteps_values[@]} ))]}
trial=$(( trial / ${#total_timesteps_values[@]} ))
num_controlled_veh=${num_controlled_veh_values[$(( trial % ${#num_controlled_veh_values[@]} ))]}

source /scratch/dc4971/nocturne_lab/.venv/bin/activate
python experiments/hr_rl/run_hr_ppo_cli.py --sweep-name=${sweep_name} --lr=${lr} --ent-coef=${ent_coef} --reg-weight=${reg_weight} --total-timesteps=${total_timesteps} --num-controlled-veh=${num_controlled_veh}
