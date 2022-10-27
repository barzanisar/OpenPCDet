# Train_5_0
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=kitti-pointrcnn_iou_train_5 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5.yaml --tcp_port 18461 --ckpt_save_interval 1 --fix_random_seed
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-15%1 --job-name=kitti-pointrcnn_iou_train_5_highepochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_highepochs.yaml --tcp_port 18465 --ckpt_save_interval 20 --fix_random_seed

# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18821 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-15%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_highep_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_highepochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18825 --ckpt_save_interval 20 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar


#Test
sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_highepochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_highepochs.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val

#TODO
sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_highep_seg_semkitti_syncbn_ep130-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_highepochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val
