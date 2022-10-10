#######################################  Depth Contrast + weather aug + pretrained on dense and kitti datasets ####################################### 

####################################### Finetune SSL models (Splits: Finetuning on small subset of labelled data from each weather split) #######################################

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

####################################### Finetune SSL models (Finetune on 60% clear and 5%, 10% etc adverse) #######################################

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_adamw_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18835 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_adamw_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18836 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_adamw_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18837 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_adamw_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18838 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_adamw_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18839 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18835 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18836 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18837 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18838 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18839 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p02_60epochs_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02_60epochs.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18785 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p02_60epochs_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02_60epochs.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18786 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p02_60epochs_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02_60epochs.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18787 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p02_60epochs_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02_60epochs.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18788 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_lr_0p02_60epochs_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02_60epochs.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18789 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p02_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18785 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p02_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18786 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p02_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18787 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p02_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18788 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_lr_0p02_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p02.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18789 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p015_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p015.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_lr_0p015_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p015.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p015_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p015.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_lr_0p015_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p015.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_lr_0p015_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_lr0p015.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_wd_0p001_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_wd_0p001.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_wd_0p001_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_wd_0p001.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_wd_0p001_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_wd_0p001.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_wd_0p001_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_wd_0p001.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_wd_0p001_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_wd_0p001.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18399 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl




# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_5_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_10_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_20_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_30_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_40_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_50_dc_1in2_cube_up_red_dense_kitti_ep330-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl
