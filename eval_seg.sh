#######################################  Seg Contrast + pretrained on dense and kitti datasets ####################################### 

####################################### Finetune SSL models (Splits: Finetuning on small subset of labelled data from each weather split) #######################################

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_5_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_5_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_5_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_5_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_5_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_10_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_10_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_10_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_10_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_10_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_20_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_20_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_20_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_20_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_20_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_30_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_30_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_30_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_30_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_30_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_40_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_40_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_40_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_40_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_40_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_50_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_splits_50_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_50_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_splits_50_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_splits_50_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

####################################### Finetune SSL models (Finetune on 60% clear and 5%, 10% etc adverse) #######################################

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_adamw_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18295 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_adamw_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18296 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_adamw_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18297 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_adamw_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18298 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_adamw_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1_adamw.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18299 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl


# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18295 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_60_1_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18296 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18297 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_60_1_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18298 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_60_1_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_1.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18299 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_5_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_5_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_5_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_5_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_5_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_5.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_10_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_10_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_10_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_10_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_10_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_10.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_20_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_20_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_20_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_20_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_20_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_20.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_30_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_30_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_30_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18897 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_30_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18898 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_30_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_30.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18899 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_40_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18995 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_40_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18996 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_40_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18997 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_40_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18998 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_40_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_40.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 18999 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_50_seg_dense_kitti_ep200-test-clear  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16795 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-all_50_seg_dense_kitti_ep200-test-snow  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16796 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_50_seg_dense_kitti_ep200-test-light-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16797 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=finetune-all_50_seg_dense_kitti_ep200-test-dense-fog  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16798 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-3%1 --job-name=finetune-all_50_seg_dense_kitti_ep200-test-all  tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60_50.yaml  --extra_tag seg_dense_kitti_ep200 --tcp_port 16799 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl
