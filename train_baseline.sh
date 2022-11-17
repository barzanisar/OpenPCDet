# # Train on FOV3000

########################################## Baselines (scratch training) ########################################## 
## Adam one cycle
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18860 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn_snow_wet_coupled_barza tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml --tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed
## TODO (Low priority) need code changes
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn_fog_sim tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_fog_sim.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed

## AdamW
# sbatch --time=02:00:00  --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-clear-60-FOV3000-adamW tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_adamw.yaml --tcp_port 18860 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=03:00:00  --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn_snow_wet_coupled-adamW tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza_adamw.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=03:00:00  --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-all-60-FOV3000-adamW tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_adamw.yaml --tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed
## TODO (Low priority) need code changes
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn_fog_sim-adamW tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_fog_sim_adamw.yaml --tcp_port 18863 --ckpt_save_interval 1 --fix_random_seed


## Adam one cycle (on 5%, 10% etc of all weather data)
#sbatch --time=05:00:00  --gres=gpu:v100:4 --array=1-6%1 --job-name=pointrcnn-all-splits-5-snowsim tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_5_snowsim.yaml -tcp_port 18451 --ckpt_save_interval 20 --fix_random_seed
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=pointrcnn-all-splits-5 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_5.yaml -tcp_port 18461 --ckpt_save_interval 20 --fix_random_seed

## TODO (low priority)
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-10 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_10.yaml -tcp_port 18863 --ckpt_save_interval 10 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-20 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_20.yaml -tcp_port 18864 --ckpt_save_interval 5 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-30 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_30.yaml -tcp_port 18865 --ckpt_save_interval 3 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-40 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_40.yaml -tcp_port 18866 --ckpt_save_interval 2 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-50 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_50.yaml -tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed


# sbatch --time=05:00:00  --gres=gpu:v100:4 --array=1-3%1 --job-name=pointrcnn-all-60-1_snowsim tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_1_snowsim.yaml -tcp_port 18465 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=05:00:00  --gres=gpu:v100:4 --array=1-2%1 --job-name=pointrcnn-all-60-5_snowsim tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_5_snowsim.yaml -tcp_port 18470 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-all-60-1 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_1.yaml -tcp_port 18491 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-all-60-1-adamO-0p003-0p001-0p4 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_1_adamO_0p003_0p001_0p4.yaml -tcp_port 18490 --ckpt_save_interval 1 --fix_random_seed

# sbatch --time=05:00:00  --gres=gpu:v100:4 --array=1-2%1 --job-name=pointrcnn-all-60-5 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_5.yaml -tcp_port 18481 --ckpt_save_interval 1 --fix_random_seed

#sbatch --time=02:00:00  --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-all-60-1-adamW tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_1_adamw.yaml -tcp_port 18490 --ckpt_save_interval 1 --fix_random_seed
#sbatch --time=02:00:00  --gres=gpu:t4:4 --array=1-6%1 --job-name=pointrcnn-all-60-1-snowsim-adamW tools/scripts/compute_canada_train_eval_dense_snow_sim.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_1_snowsim_adamw.yaml -tcp_port 18490 --ckpt_save_interval 1 --fix_random_seed



## TODO (High priority)
## Adam one cycle (on 60% clear and 5%, 10% etc of adverse weather data)
#sbatch --time=05:00:00 --array=1-3%1 --job-name=pointrcnn-all-60-10 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_10.yaml -tcp_port 18863 --ckpt_save_interval 1 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-20 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_20.yaml -tcp_port 18864 --ckpt_save_interval 1 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-30 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_30.yaml -tcp_port 18865 --ckpt_save_interval 1 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-40 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_40.yaml -tcp_port 18866 --ckpt_save_interval 1 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-50 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60_50.yaml -tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed