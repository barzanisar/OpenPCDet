##################################################### Train baselines
# sbatch --time=3:00:00 --array=1-5%1 --job-name=second-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_clear_FOV3000_60.yaml --tcp_port 18710 
# sbatch --time=3:00:00 --array=1-5%1 --job-name=second-adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18720 
# sbatch --time=3:00:00 --array=1-6%1 --job-name=second-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_all_FOV3000_60.yaml --tcp_port 18730 

##################################################### Test baselines
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-clear-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_clear_FOV3000_60.yaml  --tcp_port 18870 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-clear-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_clear_FOV3000_60.yaml  --tcp_port 18871 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-adverse-60-allgtbdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18876 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-adverse-60-allgtbdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18877 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_all_FOV3000_60.yaml  --tcp_port 18872 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_train_all_FOV3000_60.yaml  --tcp_port 18873 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################ Finetune
# ###### Naive
#sbatch  --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18657  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth

##### Low LR
sbatch  --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-all-gtdb-lowlr-0p0003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p0003.yaml --tcp_port 18661  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth


##### EWC
#sbatch  --time=02:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-all-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18659  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth

##### Replay #Choose best lr and set it in cfg file before running this
#sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-fixed tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18663  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
#sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-random tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18664  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-mir-1epoch-720 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18666  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-AGEM tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18667  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-AGEM-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18669  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
#sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-GSS tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18670  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=12:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18665  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=12:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-EMA tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_EMA.yaml --tcp_port 18668  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18668  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth

# Tune LR
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-0p0001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0001.yaml --tcp_port 18665  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-0p005 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18666  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18667  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18668  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth
# sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=second-finetune-adverse-180clear-allgtdb-replay-EMIR-0p0003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18669  --pretrained_model /OpenPCDet/checkpoints/second_train_clear_FOV3000_60_ep74.pth

# ##################### Test tune LR
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p0001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0001.yaml --tcp_port 18291 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p0001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0001.yaml --tcp_port 18292 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p005-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18293 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p005-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18294 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18295 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18296 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18297 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18298 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p0003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18299 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-EMIR-0p0003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18290 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################## Test finetune
# ###### Naive
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18193 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18194 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### Low LR
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-lowlr-0p001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18197 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-lowlr-0p001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18198 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### EWC
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18195 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=second-finetune-adverse-all-gtdb-ewc-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18196 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

###### REPLAY
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-fixed-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18891 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-fixed-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18292 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-random-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18293 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-random-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18294 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-emir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18295 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-emir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18296 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-emir-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18297 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-emir-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18298 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-mir_1ep_720-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18299 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-mir_1ep_720-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18281 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-agem-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18282 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-agem-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18283 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-agem-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18284 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-agem-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18285 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-GSS-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18286 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-1%1 --job-name=second-finetune-replay-GSS-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/second_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18287 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl
