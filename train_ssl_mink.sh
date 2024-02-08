################## create waymo database 1% ##################
sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_100.yaml
sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_20.yaml
sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_100.yaml
sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_20.yaml

sbatch --time=01:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_100_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_100.yaml
# sbatch --time=02:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_20_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_20.yaml

#################### 80 epochs #########################################
## scratch
sbatch --time=10:00:00 --array=1-2%1 --job-name=det-scratch-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18910 --extra_tag scratch 

## finetune seg 
sbatch --time=10:00:00 --array=1-2%1 --job-name=det-seg-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18911 --extra_tag segcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-2%1 --job-name=det-seg_plus_det0p5-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18912 --extra_tag segcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-2%1 --job-name=det-dc-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18913 --extra_tag depthcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-2%1 --job-name=det-dc_det0p5-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18914 --extra_tag depthcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar

sbatch --time=10:00:00 --array=1-1%1 --job-name=det-scratch-80ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_sampled_100.yaml --tcp_port 18911 --extra_tag scratch 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg-80ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_sampled_100.yaml --tcp_port 18911 --extra_tag segcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg_plus_det0p5-80ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_sampled_100.yaml --tcp_port 18912 --extra_tag segcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc-80ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_sampled_100.yaml --tcp_port 18913 --extra_tag depthcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc_det0p5-80ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_sampled_100.yaml --tcp_port 18914 --extra_tag depthcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar

# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-scratch-80ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18915 --extra_tag scratch_try2
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg-80ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18916 --extra_tag segcontrast_try2 --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg_plus_det0p5-80ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18917 --extra_tag segcontrast_plus_dethead_try2 --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc-80ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18918 --extra_tag depthcontrast_try2 --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc_det0p5-80ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18919 --extra_tag depthcontrast_plus_dethead_try2 --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar


#################### 200 epochs #########################################
## scratch
sbatch --time=10:00:00 --array=1-3%1 --job-name=det-scratch-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18710 --extra_tag scratch

## finetune seg 
sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18711 --extra_tag segcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg_plus_det0p5-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18712 --extra_tag segcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18713 --extra_tag depthcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc_det0p5-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18714 --extra_tag depthcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar


sbatch --time=10:00:00 --array=1-4%1 --job-name=det-scratch-200ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_sampled_100.yaml --tcp_port 18810 --extra_tag scratch 
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg-200ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_sampled_100.yaml --tcp_port 18811 --extra_tag segcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-seg_plus_det0p5-200ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_sampled_100.yaml --tcp_port 18812 --extra_tag segcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc-200ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_sampled_100.yaml --tcp_port 18813 --extra_tag depthcontrast --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar
# sbatch --time=10:00:00 --array=1-4%1 --job-name=det-dc_det0p5-200ep tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_sampled_100.yaml --tcp_port 18814 --extra_tag depthcontrast_plus_dethead --pretrained_model /OpenPCDet/checkpoints/minkunet_pretrain_ep99.pth.tar

########################## nuscenes #########################################