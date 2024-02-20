################## create waymo database 1% ##################
sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_100.yaml
sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_20.yaml
sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_100.yaml
sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_20.yaml

sbatch --time=01:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_100_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_100.yaml
# sbatch --time=02:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_20_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_20.yaml

#################### 80 epochs #########################################
## scratch
#sbatch --time=10:00:00 --array=1-5%1 --job-name=det-scratch-80ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18910 --extra_tag scratch
sbatch --time=10:00:00 --array=1-5%1 --job-name=det-scratch-80ep-scene100_try3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 19910 --extra_tag scratch_try3

#################### 200 epochs #########################################
## scratch
# sbatch --dependency=afterany:25352493 --time=10:00:00 --array=1-2%1 --job-name=det-scratch-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18710 --extra_tag scratch
# sbatch --time=10:00:00 --array=1-5%1 --job-name=det-scratch-200ep-scene100_try2 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 17710 --extra_tag scratch_try2

##### 30 epochs waymo sampled 100 i.e. 1% (for pretrained models)######
# sbatch --time=8:00:00 --array=1-5%1 --job-name=det-scratch-30ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100.yaml --tcp_port 17910 --extra_tag scratch
sbatch --time=8:00:00 --array=1-5%1 --job-name=det-scratch-30ep-scene100_try3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100.yaml --tcp_port 17911 --extra_tag scratch_try3 


## finetune seg 
sbatch --time=10:00:00 --array=1-3%1 --job-name=segdet-25ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_25ep_scene_sampled_100.yaml --tcp_port 18412 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --time=10:00:00 --array=1-3%1 --job-name=det-segdet0p8-28ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_28ep_scene_sampled_100.yaml --tcp_port 18412 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --time=10:00:00 --array=1-5%1 --job-name=det-segdet0p8-70ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_70ep_scene_sampled_100.yaml --tcp_port 18413 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-75ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_75ep_scene_sampled_100.yaml --tcp_port 18413 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --time=10:00:00 --array=1-5%1 --job-name=det-segdet0p8-78ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_78ep_scene_sampled_100.yaml --tcp_port 18415 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar

# sbatch --time=10:00:00 --array=1-5%1 --job-name=det-segdet0p8-20ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_20ep_scene_sampled_100.yaml --tcp_port 18311 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --time=10:00:00 --array=1-5%1 --job-name=det-segdet0p8-60ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_60ep_scene_sampled_100.yaml --tcp_port 18313 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-30ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100.yaml --tcp_port 18914 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-80ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18915 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --dependency=afterany:25351147  --time=10:00:00 --array=1-2%1 --job-name=det-segdet0p8-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18913 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar


########################## waymo sampled 20 i.e. 5%  #########################################
### 30, 80, 200 epochs
########################## nuscenes sampled 100 i.e. 1%  #########################################
### 30, 80, 200 epochs
########################## nuscenes sampled 20 i.e. 5%  #########################################
### 30, 80, 200 epochs