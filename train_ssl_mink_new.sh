################## create waymo database 1% ##################
# sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_100.yaml
# sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_sampled_20.yaml
sbatch --time=05:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_100 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_100.yaml
sbatch --time=06:00:00 --array=1-1%1 --job-name=create_database-waymo_scene_sampled_20 tools/scripts/create_waymo_infos.sh --cfg_file tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_20.yaml
tools/scripts/create_waymo_infos_turing.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_100.yaml > ./output/log/create_waymo_gtdb_scene100_$(date +%Y-%m-%d_%H:%M).out 2>&1
tools/scripts/create_waymo_infos_turing.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/waymo_dataset_scene_sampled_20.yaml > ./output/log/create_waymo_gtdb_scene20_$(date +%Y-%m-%d_%H:%M).out 2>&1

sbatch --time=02:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_100_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_100.yaml
# sbatch --time=02:00:00 --array=1-1%1 --job-name=create_nuscenes_infos_1sweep_20_sampled tools/scripts/create_nuscenes_infos.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_20.yaml
tools/scripts/create_nuscenes_infos_turing.sh --cfg_file /OpenPCDet/tools/cfgs/dataset_configs/nuscenes_dataset_1sweeps_sampled_100.yaml > ./output/log/create_nuscenes_val_infos_$(date +%Y-%m-%d_%H:%M).out 2>&1

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
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-30ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100.yaml --tcp_port 18914 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-75ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_75ep_scene_sampled_100.yaml --tcp_port 18413 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-5%1 --job-name=segdet-80ep-scene100-t3 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_80ep_scene_sampled_100.yaml --tcp_port 18915 --extra_tag segcontrast_plus_dethead_0p8w_try3 --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
# sbatch --time=10:00:00 --array=1-2%1 --job-name=det-segdet0p8-200ep-scene100 tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_200ep_scene_sampled_100.yaml --tcp_port 18913 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar


### Ignore bn
sbatch --time=10:00:00 --array=1-3%1 --job-name=segdet-25ep-ignore_mv tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_25ep_scene_sampled_100_run_mv_ignored.yaml --tcp_port 17412 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-3%1 --job-name=segdet-30ep-ignore_mv tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100_run_mv_ignored.yaml --tcp_port 17413 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-3%1 --job-name=segdet-25ep-ignore_bn tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_25ep_scene_sampled_100_bn_ignored.yaml --tcp_port 17414 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar
sbatch --time=10:00:00 --array=1-3%1 --job-name=segdet-30ep-ignore_bn tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100_bn_ignored.yaml --tcp_port 17415 --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar



########################## waymo sampled 20 i.e. 5%  #########################################
### 30, 80, 200 epochs
########################## nuscenes sampled 100 i.e. 1%  #########################################
### 30, 80, 200 epochs
sbatch --time=8:00:00 --array=1-5%1 --job-name=30ep-nus-1perc tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/nuscenes_models/pointrcnn_minkunet_30ep_1sweeps_sampled_100.yaml --tcp_port 16911 --extra_tag scratch 
sbatch --time=8:00:00 --array=1-5%1 --job-name=t2_30ep-nus-1perc tools/scripts/submit_ddp_$CLUSTER_NAME.sh --cfg_file tools/cfgs/nuscenes_models/pointrcnn_minkunet_30ep_1sweeps_sampled_100.yaml --tcp_port 16912 --extra_tag scratch_t2
tools/scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 0,1 --tcp_port 19835 --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100_run_mv_ignored.yaml --extra_tag segcontrast_plus_dethead_0p8w --pretrained_model /OpenPCDet/checkpoints/seg_plus_det0p8_ep49.pth.tar > ./output/log/segdet_30ep_waymo1perc_mv_ignored_$(date +%Y-%m-%d_%H:%M).out 2>&1
tools/scripts/submit_ddp_turing.sh --num_gpus 2 --cuda_visible_devices 2,3 --tcp_port 19836 --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_30ep_scene_sampled_100.yaml --extra_tag scratch > ./output/log/scratch_30ep_waymo1perc_$(date +%Y-%m-%d_%H:%M).out 2>&1

########################## nuscenes sampled 20 i.e. 5%  #########################################
### 30, 80, 200 epochs