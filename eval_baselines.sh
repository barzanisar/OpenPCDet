####################################### Baselines (Scratch training all data) #######################################

####################################### AdamOnecycle #######################################

# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-snow_wet_coupled_barza-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml  --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-snow_wet_coupled_barza-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml  --tcp_port 18891 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-snow_wet_coupled_barza-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml  --tcp_port 18892 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-snow_wet_coupled_barza-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml  --tcp_port 18893 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointrcnn-snow_wet_coupled_barza-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml  --tcp_port 18894 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

#Test with dror
# sbatch --time=04:00:00 --array=1-1%1 --job-name=dror-pointrcnn-clear-60-FOV3000-test-clear tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18860 --fix_random_seed --test_only --eval_tag test_clear_FOV3000_dror --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=dror-pointrcnn-clear-60-FOV3000-test-snow tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18861 --fix_random_seed --test_only --eval_tag test_snow_FOV3000_dror --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=dror-pointrcnn-clear-60-FOV3000-test-light-fog tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18862 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000_dror --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=dror-pointrcnn-clear-60-FOV3000-test-dense-fog tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18863 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000_dror --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=dror-pointrcnn-clear-60-FOV3000-test-all tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18864 --fix_random_seed --test_only --eval_tag test_all_FOV3000_dror --test_info_pkl dense_infos_test_all_FOV3000_25.pkl
