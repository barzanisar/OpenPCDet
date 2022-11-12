# Train Pointrcnn Kitti 100%
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=kitti-pointrcnn tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn.yaml --tcp_port 18461 --ckpt_save_interval 1 --fix_random_seed

# Test Pointrcnn Kitti 100%
#sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val

# Train Pointrcnn Kitti 5%
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=kitti-pointrcnn_train_5_1_200epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1_200epochs.yaml --tcp_port 18465 --ckpt_save_interval 2 --fix_random_seed
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=kitti-pointrcnn_train_5_2_200epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_2_200epochs.yaml --tcp_port 18475 --ckpt_save_interval 2 --fix_random_seed
sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=kitti-pointrcnn_train_5_200epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_200epochs.yaml --tcp_port 18485 --ckpt_save_interval 2 --fix_random_seed

# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=kitti-pointrcnn_train_5_500epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_500epochs.yaml --tcp_port 18475 --ckpt_save_interval 6 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=kitti-pointrcnn_train_5_1000epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1000epochs.yaml --tcp_port 18485 --ckpt_save_interval 12 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-20%1 --job-name=kitti-pointrcnn_train_5_1600epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1600epochs.yaml --tcp_port 18495 --ckpt_save_interval 20 --fix_random_seed

# # TODO: Test Pointrcnn Kitti 5%
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_1_200epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1_200epochs.yaml --tcp_port 18781 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_2_200epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_2_200epochs.yaml --tcp_port 18791 --fix_random_seed --test_only --eval_tag kitti_infos_val

# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_200epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_200epochs.yaml --tcp_port 18791 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_500epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_500epochs.yaml --tcp_port 18792 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_1000epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1000epochs.yaml --tcp_port 18793 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_train_5_1600epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_1600epochs.yaml --tcp_port 18794 --fix_random_seed --test_only --eval_tag kitti_infos_val



# # Train_5_0
# # sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=kitti-pointrcnn_iou_train_5 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5.yaml --tcp_port 18461 --ckpt_save_interval 1 --fix_random_seed
# # sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-15%1 --job-name=kitti-pointrcnn_iou_train_5_highepochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_highepochs.yaml --tcp_port 18465 --ckpt_save_interval 20 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=kitti-pointrcnn_iou_train_5_200epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_200epochs.yaml --tcp_port 18465 --ckpt_save_interval 2 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=kitti-pointrcnn_iou_train_5_500epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_500epochs.yaml --tcp_port 18475 --ckpt_save_interval 6 --fix_random_seed
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=kitti-pointrcnn_iou_train_5_1000epochs tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_1000epochs.yaml --tcp_port 18485 --ckpt_save_interval 12 --fix_random_seed

# # sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18821 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar
# #sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-15%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_highep_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_highepochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18825 --ckpt_save_interval 20 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-2%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_200ep_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_200epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18825 --ckpt_save_interval 2 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_500ep_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_500epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18835 --ckpt_save_interval 6 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar
# sbatch --time=02:00:00 --gres=gpu:t4:4 --array=1-10%1 --job-name=finetune-kitti-pointrcnn_iou_train_5_1000ep_seg_semkitti_syncbn_ep130 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_1000epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18845 --ckpt_save_interval 12 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/seg_semkitti_syncbn_ep130.pth.tar


# # #Test
# # #sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_highepochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_highepochs.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val
# # sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_200epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_200epochs.yaml --tcp_port 18755 --fix_random_seed --test_only --eval_tag kitti_infos_val
# # sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_500epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_500epochs.yaml --tcp_port 18765 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_1000epochs-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_train_5_1000epochs.yaml --tcp_port 18775 --fix_random_seed --test_only --eval_tag kitti_infos_val

# # #TODO
# # #sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_highep_seg_semkitti_syncbn_ep130-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_highepochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_200ep_seg_semkitti_syncbn_ep130-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_200epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18715 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_500ep_seg_semkitti_syncbn_ep130-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_500epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18725 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=02:00:00 --array=1-4%1 --job-name=kitti-pointrcnn_iou_train_5_1000ep_seg_semkitti_syncbn_ep130-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_iou_finetune_train_5_1000epochs.yaml --extra_tag seg_semKitti_syncbn_ep130 --tcp_port 18735 --fix_random_seed --test_only --eval_tag kitti_infos_val
