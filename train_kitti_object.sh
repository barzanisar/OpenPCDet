
############### Supervised pretraining: predict gt boxes and fg/bg, Finetune on 5% with class labels ###################################
# Train Pointrcnn Kitti 95% split 0, only predict gt boxes (and fg/bg)
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=kitti-pointrcnn-95_0_sup_object tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_object.yaml --tcp_port 18461 --fix_random_seed

# Test Pointrcnn Kitti 95% split 0, only predict gt boxes to find best pretrained ckpt
# sbatch --time=04:00:00 --array=1-1%1 --job-name=kitti-pointrcnn-95_0_sup_object-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_object.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val

# Finetune model:"Pointrcnn Kitti 95% split 0, only predict gt boxes" on 5% kitti split 0 with classes
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p01.yaml --extra_tag supervised_object --tcp_port 18821 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p001.yaml --extra_tag supervised_object --tcp_port 18822 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18823 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-100ep-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_100epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18824 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth

# Val-finetuned model
sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p01.yaml --extra_tag supervised_object --tcp_port 18821 --fix_random_seed --test_only --eval_tag kitti_infos_val
sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p001.yaml --extra_tag supervised_object --tcp_port 18822 --fix_random_seed --test_only --eval_tag kitti_infos_val
sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18823 --fix_random_seed --test_only --eval_tag kitti_infos_val
sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-100ep-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_100epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18824 --fix_random_seed --test_only --eval_tag kitti_infos_val

