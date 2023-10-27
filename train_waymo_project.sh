#sbatch --time=3:00:00 --array=1-1%1 --job-name=pvrcnn-waymo tools/scripts/compute_canada_train_eval_waymo_project.sh --cfg_file tools/cfgs/waymo_models/pv_rcnn.yaml --tcp_port 18810

sbatch --time=1:00:00 --array=1-1%1 --job-name=create_database-waymo tools/scripts/create_waymo_infos.sh
