# Train on FOV3000 (Done)
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60-FOV3000 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18880 --ckpt_save_interval 5 --fix_random_seed
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-all-60-FOV3000 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18882 --ckpt_save_interval 5 --fix_random_seed

#Train on clear FOV3000 with weather sim
#sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18880 --ckpt_save_interval 5 --fix_random_seed
#Test-Train on clear FOV3000 with weather sim
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-weather-sim-1in10-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60_weather.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl




# Train on 360 deg (Done)
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4  --tcp_port 18881 --ckpt_save_interval 5 --fix_random_seed
# sbatch --time=6:00:00 --array=1-1%1 --job-name=pointrcnn-all-60 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --tcp_port 18883 --ckpt_save_interval 5 --fix_random_seed

# Finetune on FOV3000 (Done)
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --extra_tag dc_FOV3000_ep180 --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18884 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_FOV3000_checkpoint-ep180.pth.tar
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18885 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_checkpoint-ep180.pth.tar
#?????
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in2_wet_fog1in2_cube_upsampleF_FOV3000_ep180 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --extra_tag dc_snow1in2_wet_fog1in2_cube_upsampleF_FOV3000_ep180 --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 18886 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_snow1in2_wet_fog1in2_cube_upsampleF_FOV3000_checkpoint-ep180.pth.tar
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000_ep180 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --extra_tag dc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000_ep180 --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4  --tcp_port 28886 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/dc_snow1in2_wet_fog1in2_cubeF_upsampleF_FOV3000_checkpoint-ep180.pth.tar

# Finetune on 360deg
# sbatch --time=6:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml --extra_tag dc_360deg_ep150 --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --tcp_port 28990 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_60/dc_360deg_checkpoint-ep150.pth.tar
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --tcp_port 18891 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_60/dc_snow1in10_wet_fog1in10_cubeF_upsampleF_checkpoint-ep150.pth.tar
### sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in2_wet_fog1in2_cube_upsampleF_360deg_ep150 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml --extra_tag dc_snow1in2_wet_fog1in2_cube_upsampleF_360deg_ep150  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --tcp_port 18892 --ckpt_save_interval 5 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_60/dc_snow1in2_wet_fog1in2_cube_upsampleF_checkpoint-ep150.pth.tar

# # TEST Finetune on FOV3000
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_FOV3000_ep180 --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_FOV3000_ep180 --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_FOV3000_ep180 --tcp_port 18888 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_FOV3000_ep180 --tcp_port 18889 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_FOV3000_ep180-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_FOV3000_ep180 --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --tcp_port 28886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --tcp_port 28887 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --tcp_port 28888 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --tcp_port 28889 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-FOV3000-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_FOV3000_ep180 --tcp_port 28890 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# # TEST Finetune on 360 deg
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_360deg_ep150 --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_360deg_ep150 --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_360deg_ep150 --tcp_port 18888 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_360deg_ep150 --tcp_port 18889 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_360deg_ep150-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_360deg_ep150 --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --tcp_port 28886 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --tcp_port 28887 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --tcp_port 28888 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --tcp_port 28889 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-all-60_dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag dc_snow1in10_wet_fog1in10_cubeF_upsampleF_360deg_ep150 --tcp_port 28890 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl


#TEST FOV3000 (DONE)
# #Test pointrcnn.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# # Test pointrcnn_train_all.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18888 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18889 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# #TEST 360deg 
# #Test pointrcnn.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl

# # # Test pointrcnn_train_all.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --tcp_port 18892 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --tcp_port 18893 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --tcp_port 18894 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --tcp_port 18895 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_60.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --tcp_port 18896 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl
