
#!/bin/bash

TRY=1
KITTI_EPOCH_1=5
KITTI_EPOCH_2=80
WAYMO_EPOCH_1=5
WAYMO_EPOCH_2=80

# Get command line arguments
while :; do
    case $1 in
    -t|--try)       # Takes an option argument; ensure it has been specified.
        if [ "$2" ]; then
            TRY=$2
            shift
        else
            die 'ERROR: "--num_epochs_1" requires a non-empty option argument.'
        fi
        ;;
    -?*)
        printf 'WARN: Unknown option (ignored): %s\n' "$1" >&2
        ;;
    *)               # Default case: No more options, so break out of the loop.
        break
    esac

    shift
done

#Kitti
#TODO fix kitti eval division by zero in iou
#5 percent
sbatch --time=8:00:00 --array=1-1%1 --job-name=kitti_scratch_"$KITTI_EPOCH_2"ep_5percent_$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet_0p05_$TRY.yaml --mode scratch --tcp_port 16920 --extra_tag scratch --num_epochs_1 $KITTI_EPOCH_2 --test_start_epoch 55 --wandb_run_name scratch_minkunet_pointrcnn --wandb_group ep"$KITTI_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-1%1 --job-name=kitti_seg_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_5percent_$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet_0p05_$TRY.yaml --tcp_port 16921 --extra_tag segcontrast --pretrained_model_1 /OpenPCDet/checkpoints/seg_ep199.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet_0p05_$TRY/segcontrast_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$KITTI_EPOCH_1.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 55 --wandb_run_name segcontrast_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-1%1 --job-name=kitti_seglidarplusdet_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_5percent_$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet_0p05_$TRY.yaml --tcp_port 16922 --extra_tag segcontrast_lidarplusdet --pretrained_model_1 /OpenPCDet/checkpoints/seg_lidar_plus_det_ep199_t2.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet_0p05_$TRY/segcontrast_lidarplusdet_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$KITTI_EPOCH_1.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 55 --wandb_run_name segcontrast_lidarplusdet_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-1%1 --job-name=kitti_segonlydet_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_5percent_$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet_0p05_$TRY.yaml --tcp_port 16924 --extra_tag segcontrast_onlydet --pretrained_model_1 /OpenPCDet/checkpoints/seg_only_det_ep199_t2.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet_0p05_$TRY/segcontrast_onlydet_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$KITTI_EPOCH_1.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 55 --wandb_run_name segcontrast_det_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY

# sbatch --time=8:00:00 --array=1-1%1 --job-name=kitti_seglidarplusdet_lnbt_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_5percent_$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet_0p05_$TRY.yaml --tcp_port 16923 --extra_tag segcontrast_lidarplusdet_lnbt --pretrained_model_1 /OpenPCDet/checkpoints/seg_lidar_plus_det_ep199_t2.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet_0p05_$TRY/segcontrast_lidarplusdet_lnbt_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$KITTI_EPOCH_1.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 55 --wandb_run_name segcontrast_lidarplusdet_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try"$TRY"_lnbt --load_num_batches_tracked


# #100 percent
# sbatch --time=8:00:00 --array=1-2%1 --job-name=kitti_scratch_"$KITTI_EPOCH_2"ep_100percent_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet.yaml --mode scratch --tcp_port 16925 --extra_tag scratch_t$TRY --num_epochs_1 $KITTI_EPOCH_2 --test_start_epoch 0 --wandb_run_name scratch_minkunet_pointrcnn --wandb_group ep"$KITTI_EPOCH_2"_try$TRY

# sbatch --time=8:00:00 --array=1-2%1 --job-name=kitti_seg_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_100percent_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet.yaml --tcp_port 16926 --extra_tag segcontrast_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_ep199.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet/segcontrast_t"$TRY"_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_5.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 0 --wandb_run_name segcontrast_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY

# sbatch --time=8:00:00 --array=1-2%1 --job-name=kitti_seglidarplusdet_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_100percent_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet.yaml --tcp_port 16927 --extra_tag segcontrast_lidarplusdet_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_lidar_plus_det_ep199_t2.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet/segcontrast_lidarplusdet_t"$TRY"_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_5.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 0 --wandb_run_name segcontrast_lidarplusdet_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY

# sbatch --time=8:00:00 --array=1-2%1 --job-name=kitti_segonlydet_"$KITTI_EPOCH_1"ep_"$KITTI_EPOCH_2"ep_100percent_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minkunet.yaml --tcp_port 16928 --extra_tag segcontrast_onlydet_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_only_det_ep199_t2.pth.tar --num_epochs_1 $KITTI_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/kitti_models/pointrcnn_minkunet/segcontrast_onlydet_t"$TRY"_ep"$KITTI_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_5.pth --num_epochs_2 $KITTI_EPOCH_2 --test_start_epoch 0 --wandb_run_name segcontrast_det_10perc_waymo_minkunet --wandb_group ep"$KITTI_EPOCH_1"_ep"$KITTI_EPOCH_2"_try$TRY


#Waymo
#1 percent
# which is better without or with batches tracked? and run more tries of that
sbatch --time=8:00:00 --array=1-3%1 --job-name=waymo_scratch_"$WAYMO_EPOCH_2"ep_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class.yaml --mode scratch --tcp_port 16929 --extra_tag scratch_t$TRY --num_epochs_1 $WAYMO_EPOCH_2 --test_start_epoch 0 --test_sample_interval 20 --eval_tag val_5perc --wandb_run_name scratch_minkunet_pointrcnn --wandb_group ep"$WAYMO_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-3%1 --job-name=waymo_seg_"$WAYMO_EPOCH_1"ep_"$WAYMO_EPOCH_2"ep_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class.yaml --tcp_port 16930 --extra_tag segcontrast_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_ep199.pth.tar --num_epochs_1 $WAYMO_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class/segcontrast_t"$TRY"_ep"$WAYMO_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$WAYMO_EPOCH_1.pth --num_epochs_2 $WAYMO_EPOCH_2 --test_start_epoch 0 --test_sample_interval 20 --eval_tag val_5perc --wandb_run_name segcontrast_10perc_waymo_minkunet --wandb_group ep"$WAYMO_EPOCH_1"_ep"$WAYMO_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-3%1 --job-name=waymo_seglidarplusdet_"$WAYMO_EPOCH_1"ep_"$WAYMO_EPOCH_2"ep_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class.yaml --tcp_port 16931 --extra_tag segcontrast_lidarplusdet_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_lidar_plus_det_ep199_t2.pth.tar --num_epochs_1 $WAYMO_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class/segcontrast_lidarplusdet_t"$TRY"_ep"$WAYMO_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$WAYMO_EPOCH_1.pth --num_epochs_2 $WAYMO_EPOCH_2 --test_start_epoch 0 --test_sample_interval 20 --eval_tag val_5perc --wandb_run_name segcontrast_lidarplusdet_10perc_waymo_minkunet --wandb_group ep"$WAYMO_EPOCH_1"_ep"$WAYMO_EPOCH_2"_try$TRY

sbatch --time=8:00:00 --array=1-3%1 --job-name=waymo_segonlydet_"$WAYMO_EPOCH_1"ep_"$WAYMO_EPOCH_2"ep_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class.yaml --tcp_port 16932 --extra_tag segcontrast_onlydet_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_only_det_ep199_t2.pth.tar --num_epochs_1 $WAYMO_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class/segcontrast_onlydet_t"$TRY"_ep"$WAYMO_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$WAYMO_EPOCH_1.pth --num_epochs_2 $WAYMO_EPOCH_2 --test_start_epoch 0 --test_sample_interval 20 --eval_tag val_5perc --wandb_run_name segcontrast_det_10perc_waymo_minkunet --wandb_group ep"$WAYMO_EPOCH_1"_ep"$WAYMO_EPOCH_2"_try$TRY 

# sbatch --time=8:00:00 --array=1-3%1 --job-name=waymo_seglidarplusdet_lnbt_"$WAYMO_EPOCH_1"ep_"$WAYMO_EPOCH_2"ep_try$TRY tools/scripts/submit_ddp_${CLUSTER_NAME}_frozen_waymo.sh --cfg_file tools/cfgs/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class.yaml --tcp_port 16933 --extra_tag segcontrast_lidarplusdet_lnbt_t$TRY --pretrained_model_1 /OpenPCDet/checkpoints/seg_lidar_plus_det_ep199_t2.pth.tar --num_epochs_1 $WAYMO_EPOCH_1 --pretrained_model_2 /OpenPCDet/output/waymo_models/pointrcnn_minkunet_scene_sampled_100_all_class/segcontrast_lidarplusdet_lnbt_t"$TRY"_ep"$WAYMO_EPOCH_1"_frozen_bb/ckpt/checkpoint_epoch_$WAYMO_EPOCH_1.pth --num_epochs_2 $WAYMO_EPOCH_2 --test_start_epoch 0 --test_sample_interval 20 --eval_tag val_5perc --wandb_run_name segcontrast_lidarplusdet_10perc_waymo_minkunet --wandb_group ep"$WAYMO_EPOCH_1"_ep"$WAYMO_EPOCH_2"_try"$TRY"_lnbt --load_num_batches_tracked

