# # Train baselines
# sbatch --time=3:00:00 --array=1-3%1 --job-name=pointrcnn-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18810 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-8%1 --job-name=pointrcnn-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml --tcp_port 18820 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn-adverse-60-adverse-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60.yaml --tcp_port 18830 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn--adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18840 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_40epochs.yaml --tcp_port 18820 --ckpt_save_interval 1 --fix_random_seed

# Test baselines
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_40epochs.yaml  --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointrcnn-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_40epochs.yaml  --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-clear-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml  --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-adverse-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60.yaml  --tcp_port 18884 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-adverse-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60.yaml  --tcp_port 18885 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-adverse-60-allgtbdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointrcnn-adverse-60-allgtbdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# Finetune
###### Naive
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-adverse-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60.yaml --tcp_port 18856 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18857 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth

###### Low LR
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-adverse-gtdb-low-lr tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr.yaml --tcp_port 18858 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-all-gtdb-low-lr tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr.yaml --tcp_port 18859 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18860 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=finetune-adverse-adverse-lowlr-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr_0p003.yaml --tcp_port 18861 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth


###### EWC
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-adverse-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_ewc.yaml --tcp_port 18858 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-all-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18859 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth

# ###### Replay #Choose best lr and set it in cfg file before running this
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18863 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18864 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18865 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-1epoch-interval tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18866 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
#sbatch  --time=1:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-10epoch-interval tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_10ep.yaml --tcp_port 18867 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth

# ###### TODO: SI #Choose best lr and set it in cfg file before running this
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-adverse-gtdb-SI tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_SI.yaml --tcp_port 18858 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-all-gtdb-SI tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_SI.yaml --tcp_port 18862 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth


# # Test finetune
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-low-lr-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-low-lr-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-low-lr-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-low-lr-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_ewc.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-ewc-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_ewc.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-lr-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-lowlr-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr_0p003.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-lowlr-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_lowlr_0p003.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-ewc-new-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_ewc_new.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-adverse-gtdb-ewc-new-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_ewc_new.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-new-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc_new.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-new-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc_new.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl
