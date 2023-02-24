##################################################### Train baselines
# sbatch --time=3:00:00 --array=1-5%1 --job-name=voxelrcnncar-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_clear_FOV3000_60.yaml --tcp_port 18610 
# sbatch --time=3:00:00 --array=1-5%1 --job-name=voxelrcnncar-adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18640 
# sbatch --time=3:00:00 --array=1-6%1 --job-name=voxelrcnncar-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_all_FOV3000_60.yaml --tcp_port 18620 

##################################################### Test baselines
# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-clear-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_clear_FOV3000_60.yaml  --tcp_port 18860 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-clear-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_clear_FOV3000_60.yaml  --tcp_port 18861 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-adverse-60-allgtbdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18866 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-adverse-60-allgtbdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18867 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_all_FOV3000_60.yaml  --tcp_port 18862 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=voxelrcnncar-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_train_all_FOV3000_60.yaml  --tcp_port 18863 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################ Finetune
##### Naive
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18857  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth

###### Low LR
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-lowlr-0p0003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p0003.yaml --tcp_port 18861  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-lowlr-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18861  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth


###### EWC
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18859  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth

# ###### Replay #Choose best lr and set it in cfg file before running this
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-fixed tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18863  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-random tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18864  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-mir-1epoch-720 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18866  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-8%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-AGEM tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18867  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-8%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-AGEM-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18869  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-GSS tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18870  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18865  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-8%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18868  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18871  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-8%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-plus-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus_0p003.yaml --tcp_port 18872  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth

#sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-EMA tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18865  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth

# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p01 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p01.yaml --tcp_port 18865  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p005 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18866  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18867  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18868  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p0003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18869  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth

# Repeat all experiments with 0.001 and if time permits repeat EMIR, plus and ema with 0.003
# 0.001 is the best lr for adverse and 0.0003 is the best for clear
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18865  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-EMA-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_EMA_0p001.yaml --tcp_port 18865  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-8%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-EMIR-plus-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus_0p001.yaml --tcp_port 18868  --pretrained_model /OpenPCDet/checkpoints/voxelrcnncar_train_clear_FOV3000_60_ep71.pth



##################### Test 
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p01-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p01.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p01-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p01.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p005-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p005-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-plus-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus_0p003.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-plus-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus_0p003.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p0003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18399 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-EMIR-0p0003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18390 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################## Test finetune
# ###### Naive
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### Low LR
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-lowlr-0p0003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p0003.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-lowlr-0p0003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p0003.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### EWC
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-all-gtdb-ewc-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ###### REPLAY
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-fixed-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-fixed-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-random-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18399 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-random-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18300 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18305 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18306 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-agem-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18307 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-agem-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18308 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-agem-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18309 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-agem-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18310 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-GSS-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18311 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-GSS-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18312 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18301 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18302 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18303 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18304 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-EMA-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18305 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=voxelrcnn-finetune-adverse-180clear-allgtdb-replay-emir-EMA-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/voxelrcnncar_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18306 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

