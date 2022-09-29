# # Train on FOV3000

########################################## Baselines (scratch training) ########################################## 
# # Adam one cycle
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18860 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn_snow_wet_coupled_barza tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml --tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed

# # AdamW-cosine
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000_adamw_cosine_0p001 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_adamw_cosine_0p001.yaml -tcp_port 18864 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn_snow_wet_coupled_barza_adamw_cosine_0p001 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_snow_wet_coupled_barza_adamw_cosine_0p001.yaml -tcp_port 18865 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000_adamw_cosine_0p001 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_adamw_cosine_0p001.yaml -tcp_port 18866 --ckpt_save_interval 1 --fix_random_seed
 
# # Adam one cycle (on subset)
#sbatch --time=15:00:00 --array=1-2%1 --job-name=pointrcnn-all-splits-60-1 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_1.yaml -tcp_port 18862 --ckpt_save_interval 100 --fix_random_seed
#sbatch --time=15:00:00 --array=1-2%1 --job-name=pointrcnn-all-splits-60-5 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_5.yaml -tcp_port 18861 --ckpt_save_interval 20 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-60-10 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_10.yaml -tcp_port 18863 --ckpt_save_interval 10 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-60-20 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_20.yaml -tcp_port 18864 --ckpt_save_interval 5 --fix_random_seed

## TODO
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-60-30 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_30.yaml -tcp_port 18865 --ckpt_save_interval 3 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-60-40 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_40.yaml -tcp_port 18866 --ckpt_save_interval 2 --fix_random_seed
#sbatch --time=15:00:00 --array=1-1%1 --job-name=pointrcnn-all-splits-60-50 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_splits_60_50.yaml -tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed

########################################## Finetune SSL models ########################################## 

# Finetune (on subset)
# sbatch --time=15:00:00 --array=1-2%1 --job-name=finetune-all_splits_1_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_1_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18845 --ckpt_save_interval 100 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
# sbatch --time=15:00:00 --array=1-2%1 --job-name=finetune-all_splits_5_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_5_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18855 --ckpt_save_interval 20 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
# sbatch --time=15:00:00 --array=1-1%1 --job-name=finetune-all_splits_10_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_10_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18865 --ckpt_save_interval 10 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
# sbatch --time=15:00:00 --array=1-1%1 --job-name=finetune-all_splits_20_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_20_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18875 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar

## TODO
#sbatch --time=15:00:00 --array=1-1%1 --job-name=finetune-all_splits_30_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_30_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18885 --ckpt_save_interval 3 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
#sbatch --time=15:00:00 --array=1-1%1 --job-name=finetune-all_splits_40_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_40_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18895 --ckpt_save_interval 2 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
#sbatch --time=15:00:00 --array=1-1%1 --job-name=finetune-all_splits_50_dc_1in2_cube_up_red_dense_kitti_ep330 tools/scripts/compute_canada_train_eval_dense.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_splits_60_50_adamw_cosine_0p001_0p0001.yaml --extra_tag dc_1in2_cube_up_red_dense_kitti_ep330 --tcp_port 18955 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_1in2_cube_up_b14_lr0p15_red_dense_kitti_ep330.pth.tar
