################## create waymo database 1% ##################
sbatch --time=03:00:00 --array=1-1%1 --job-name=create_database-waymo tools/scripts/create_waymo_infos.sh

sbatch --time=10:00:00 --array=1-2%1 --job-name=pointrcnn-minkunet tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet.yaml --tcp_port 18910 
sbatch --time=10:00:00 --array=1-2%1 --job-name=pointrcnn-minkunet tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_finetune.yaml --tcp_port 18911 --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep49.pth.tar