
############### Supervised pretraining: predict gt boxes and fg/bg, Finetune on 5% with class labels ###################################
# Train Pointrcnn Kitti 95% split 0, only predict gt boxes (and fg/bg)
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=kitti-pointrcnn-95_0_sup_object tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_object.yaml --tcp_port 18461 --fix_random_seed

# Validate Pointrcnn Kitti 95% split 0, only predict gt boxes to find best pretrained ckpt
# sbatch --time=04:00:00 --array=1-1%1 --job-name=kitti-pointrcnn-95_0_sup_object-val  tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_object.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val

# Finetune model:"Pointrcnn Kitti 95% split 0, only predict gt boxes" on 5% kitti split 0 with classes
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p01.yaml --extra_tag supervised_object --tcp_port 18821 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p001.yaml --extra_tag supervised_object --tcp_port 18822 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18823 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_sup_object-5_0-100ep-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_100epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18824 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/kitti-pointrcnn-95_0_sup_object_ep78.pth

# Val-finetuned model
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p01.yaml --extra_tag supervised_object --tcp_port 18821 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p001.yaml --extra_tag supervised_object --tcp_port 18822 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18823 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-3%1 --job-name=val-finetune-kitti-95_0_sup_object-5_0-100ep-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_100epochs_object_lr0p003.yaml --extra_tag supervised_object --tcp_port 18824 --fix_random_seed --test_only --eval_tag kitti_infos_val

############### Self-Supervised pretraining: predict approx pca target boxes and fg/bg, Finetune on 5% with class labels ###################################
# Train Pointrcnn Kitti 95% split 0, only predict pca boxes (and fg/bg)
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=kitti-pointrcnn-95_0_pca_object tools/scripts/compute_canada_train_eval_kitti_pca.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_pca_object.yaml --tcp_port 18461 --fix_random_seed
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=kitti-pointrcnn-95_0_close_object tools/scripts/compute_canada_train_eval_kitti_close.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_close_object.yaml --tcp_port 18462 --fix_random_seed
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-12%1 --job-name=kitti-pointrcnn-95_0_minmax_object tools/scripts/compute_canada_train_eval_kitti_minmax.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minmax_object.yaml --tcp_port 18463 --fix_random_seed  --extra_tag startfromckpt76 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_minmax_object_checkpoint_epoch_76.pth
# sbatch --time=03:00:00 --gres=gpu:t4:4 --array=1-12%1 --job-name=kitti-pointrcnn-95_0_minmax_object tools/scripts/compute_canada_train_eval_kitti_minmax.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minmax_object.yaml --tcp_port 18463 --fix_random_seed

# Validate Pointrcnn Kitti 95% split 0, only predict pca boxes to find best pretrained ckpt
# sbatch --time=04:00:00 --array=1-1%1 --job-name=kitti-pointrcnn-95_0_pca_object-val  tools/scripts/compute_canada_train_eval_kitti_pca.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_pca_object.yaml --tcp_port 18795 --fix_random_seed --test_only --eval_tag kitti_infos_val_pca_object
# sbatch --time=04:00:00 --array=1-1%1 --job-name=kitti-pointrcnn-95_0_close_object-val  tools/scripts/compute_canada_train_eval_kitti_close.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_close_object.yaml --tcp_port 18796 --fix_random_seed --test_only --eval_tag kitti_infos_val_close_object
# sbatch --time=04:00:00 --array=1-4%1 --job-name=kitti-pointrcnn-95_0_minmax_object-val  tools/scripts/compute_canada_train_eval_kitti_minmax.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minmax_object.yaml --tcp_port 18797 --fix_random_seed --test_only --extra_tag startfromckpt76 --eval_tag kitti_infos_val_minmax_object
# sbatch --time=04:00:00 --array=1-4%1 --job-name=kitti-pointrcnn-95_0_minmax_object-val  tools/scripts/compute_canada_train_eval_kitti_minmax.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_minmax_object.yaml --tcp_port 18797 --fix_random_seed --test_only

#TODO: 
# Get the best checkpt from above
# Finetune model:"Pointrcnn Kitti 95% split 0, only predict pca boxes" on 5% kitti split 0 with classes
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_minmax-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p001.yaml --tcp_port 18822 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_minmax_object_checkpoint_epoch_191.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_minmax-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p003.yaml --tcp_port 18823 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_minmax_object_checkpoint_epoch_191.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_minmax-5_0-lr0p005 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p005.yaml --tcp_port 18824 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_minmax_object_checkpoint_epoch_191.pth
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-kitti-95_0_minmax-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p01.yaml --tcp_port 18825 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_minmax_object_checkpoint_epoch_191.pth

# Val-finetuned model
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=val-finetune-kitti-95_0_minmax-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p001.yaml --tcp_port 18822 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=val-finetune-kitti-95_0_minmax-5_0-lr0p003 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p003.yaml --tcp_port 18823 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=val-finetune-kitti-95_0_minmax-5_0-lr0p005 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p005.yaml --tcp_port 18824 --fix_random_seed --test_only --eval_tag kitti_infos_val
# sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=val-finetune-kitti-95_0_minmax-5_0-lr0p01 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_finetune_train_5_0_200epochs_minmax_lr0p01.yaml --tcp_port 18825 --fix_random_seed --test_only --eval_tag kitti_infos_val

# Self-training
#sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=kitti-pointrcnn-5_0_train_scratch tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_0_200epochs.yaml --tcp_port 18461 --fix_random_seed
sbatch --time=04:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=val-finetune-kitti-95_0_minmax-5_0-lr0p001 tools/scripts/compute_canada_train_eval_kitti.sh --cfg_file tools/cfgs/kitti_models/pointrcnn_train_5_0_200epochs.yaml --tcp_port 18822 --fix_random_seed --test_only --eval_tag kitti_infos_val

