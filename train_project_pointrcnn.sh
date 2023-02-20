# # Train baselines
# sbatch --time=3:00:00 --array=1-3%1 --job-name=pointrcnn-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_clear_FOV3000_60.yaml --tcp_port 18810 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=2:00:00 --array=1-1%1 --job-name=pointrcnn-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60.yaml --tcp_port 18820 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn-adverse-60-adverse-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60.yaml --tcp_port 18830 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn--adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18840 --ckpt_save_interval 1 --fix_random_seed
# sbatch --time=3:00:00 --array=1-4%1 --job-name=pointrcnn-all-60-40epochs tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_train_all_FOV3000_60_40epochs.yaml --tcp_port 18820 --ckpt_save_interval 1 --fix_random_seed

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
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18865 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-1epoch-interval tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18866 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-1epoch-720 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18866 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-AGEM tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18867 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-AGEM-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18869 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18868 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-GSS tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18270 --ckpt_save_interval 1 --fix_random_seed --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-EMA tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18222 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth

# # ############# Hyper param sensivity 
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-d10 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d10.yaml --tcp_port 18771 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-d20 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d20.yaml --tcp_port 18772 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-d50 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d50.yaml --tcp_port 18773 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-ep5 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep5.yaml --tcp_port 18774 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-ep20 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep20.yaml --tcp_port 18775 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-ep30 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep30.yaml --tcp_port 18776 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-m1 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m1.yaml --tcp_port 18778 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-m10 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m10.yaml --tcp_port 18779 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-m20 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m20.yaml --tcp_port 18780 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth
# sbatch  --time=3:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-m50 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m50.yaml --tcp_port 18781 --pretrained_model /OpenPCDet/checkpoints/pointrcnn_train_clear_FOV3000_60_ep78.pth


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

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-replay-GSS-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18686 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=pointrcnn-finetune-replay-GSS-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18687 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-EMA-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-EMA-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d10-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d10.yaml --tcp_port 18100 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d10-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d10.yaml --tcp_port 18101 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d20-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d20.yaml --tcp_port 18102 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d20-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d20.yaml --tcp_port 18103 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d50-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d50.yaml --tcp_port 18104 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-d50-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_d50.yaml --tcp_port 18105 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep5-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep5.yaml --tcp_port 18106 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep5-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep5.yaml --tcp_port 18107 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep20-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep20.yaml --tcp_port 18108 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep20-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep20.yaml --tcp_port 18109 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep30-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep30.yaml --tcp_port 18110 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-ep30-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_ep30.yaml --tcp_port 18111 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m1-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m1.yaml --tcp_port 18112 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m1-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m1.yaml --tcp_port 18113 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m10-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m10.yaml --tcp_port 18114 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m10-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m10.yaml --tcp_port 18115 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m20-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m20.yaml --tcp_port 18116 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m20-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m20.yaml --tcp_port 18117 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m50-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m50.yaml --tcp_port 18118 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-allgtdb-replay-emir-m50-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointrcnn_finetune_all_FOV3000_60_replay_EMIR_m50.yaml --tcp_port 18119 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl
