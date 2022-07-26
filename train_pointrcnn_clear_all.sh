# Train on FOV3000
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60-FOV3000 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18880 --ckpt_save_interval 5 --fix_random_seed

# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-all-60-FOV3000 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --train_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18882 --ckpt_save_interval 5 --fix_random_seed

# Train on 360 deg
# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --extra_tag train_clear_60 --tcp_port 18881 --ckpt_save_interval 5 --fix_random_seed

# sbatch --time=24:00:00 --array=1-2%1 --job-name=pointrcnn-all-60 --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --train_batch_size 4 --extra_tag train_all_60 --tcp_port 18883 --ckpt_save_interval 5 --fix_random_seed


#TEST FOV3000
#Test pointrcnn.yaml on different weather test sets
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-FOV3000-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_clear_60_FOV3000 --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# Test pointrcnn_train_all.yaml on different weather test sets
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_snow_FOV3000 --test_info_pkl dense_infos_test_snow_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18888 --fix_random_seed --test_only --eval_tag test_light_fog_FOV3000 --test_info_pkl dense_infos_test_light_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18889 --fix_random_seed --test_only --eval_tag test_dense_fog_FOV3000 --test_info_pkl dense_infos_test_dense_fog_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-FOV3000-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/FOV3000_Infos --test_batch_size 4 --extra_tag train_all_60_FOV3000 --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_all_FOV3000 --test_info_pkl dense_infos_test_all_FOV3000_25.pkl

# #TEST 360deg 
# #Test pointrcnn.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-360deg-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-360deg-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-360deg-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-360deg-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-clear-60-360deg-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_clear_60 --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl

# # Test pointrcnn_train_all.yaml on different weather test sets
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-360deg-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_all_60 --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear --test_info_pkl dense_infos_test_clear_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-360deg-test-snow --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_all_60 --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_snow --test_info_pkl dense_infos_test_snow_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-360deg-test-light-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_all_60 --tcp_port 18888 --fix_random_seed --test_only --eval_tag test_light_fog --test_info_pkl dense_infos_test_light_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-360deg-test-dense-fog --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_all_60 --tcp_port 18889 --fix_random_seed --test_only --eval_tag test_dense_fog --test_info_pkl dense_infos_test_dense_fog_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-360deg-test-all --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all.yaml  --data_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense --infos_dir /home/nisarbar/projects/rrg-swasland/Datasets/Dense/Infos --test_batch_size 4 --extra_tag train_all_60 --tcp_port 18890 --fix_random_seed --test_only --eval_tag test_all --test_info_pkl dense_infos_test_all_25.pkl
