################## create waymo database 1% ##################
#sbatch --time=03:00:00 --array=1-1%1 --job-name=create_database-waymo tools/scripts/create_waymo_infos.sh

################## Scratch train on narval ##############################
# sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
# sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830

# Test (not needed, check results from graham)
# sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=test-pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810 --test_only 
# sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=test-pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830 --test_only

#### In progress #########
#sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18820 --workers 2
#sbatch --time=03:00:00 --nodes=2 --ntasks=2 --array=1-2%1 --job-name=centerpoint-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_clus_1percent.yaml --tcp_port 18840 --workers 2

################## Scratch train on graham ##############################
# sbatch --time=04:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810
# sbatch --time=04:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830

# Test (In progress)
#sbatch --time=04:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=test-pointrcnn-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_1percent.yaml --tcp_port 18810 --test_only 
#sbatch --time=04:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=test-pointrcnn-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_1percent.yaml --tcp_port 18830 --test_only

############ Not needed ###############
# sbatch --time=03:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-3%1 --job-name=centerpoint-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_1percent.yaml --tcp_port 18820 --workers 2 --batch_size 4
# sbatch --time=03:00:00 --gres=gpu:t4:4 --nodes=2 --ntasks=2 --array=1-3%1 --job-name=centerpoint-clus-waymo tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/centerpoint_clus_1percent.yaml --tcp_port 18840 --workers 2 --batch_size 4


######################## Finetune on narval (In progress) ###########################
sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_finetune_100ep_lr0p003.yaml --tcp_port 18910 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar
sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_finetune_200ep_lr0p003.yaml --tcp_port 18920 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar
sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_clus_finetune_200ep_lr0p01.yaml --tcp_port 18930 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar

sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_finetune_100ep_lr0p003.yaml --tcp_port 18940 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar
sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_finetune_200ep_lr0p003.yaml --tcp_port 18950 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar
sbatch --time=04:00:00 --nodes=2 --ntasks=2 --array=1-1%1 --job-name=finetune-pointrcnn tools/scripts/submit_ddp_compute_canada_waymo_multinode.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_finetune_200ep_lr0p01.yaml --tcp_port 18960 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_pretrain_ep99.pth.tar




# prerain centerpoint waymo -> reduce batchsize and try on 3 nodes?-> on 2 nodes working
