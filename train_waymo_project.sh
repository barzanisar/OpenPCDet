#sbatch --time=03:00:00 --array=1-1%1 --job-name=create_database-waymo tools/scripts/create_waymo_infos.sh

sbatch --time=00:30:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
sbatch --time=00:30:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18890
