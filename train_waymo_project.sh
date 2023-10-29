#sbatch --time=03:00:00 --array=1-1%1 --job-name=create_database-waymo tools/scripts/create_waymo_infos.sh

# sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
# sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18820

# sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830
# sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=centerpoint-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_clus_1percent.yaml --tcp_port 18840


sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18820

sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830
sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-4%1 --job-name=centerpoint-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_clus_1percent.yaml --tcp_port 18840

# sbatch --time=00:15:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
# sbatch --time=00:15:00 --nodes=1 --ntasks=1 --array=1-1%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18890
