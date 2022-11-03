sbatch --time=3:00:00 --array=1-5%1 --job-name=pointrcnn-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18810 --ckpt_save_interval 1 --fix_random_seed
sbatch --time=3:00:00 --array=1-5%1 --job-name=pointrcnn-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml --tcp_port 18820 --ckpt_save_interval 1 --fix_random_seed
sbatch --time=3:00:00 --array=1-5%1 --job-name=pointrcnn-adverse-60-adverse-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60.yaml --tcp_port 18830 --ckpt_save_interval 1 --fix_random_seed
sbatch --time=3:00:00 --array=1-5%1 --job-name=pointrcnn--adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18840 --ckpt_save_interval 1 --fix_random_seed
