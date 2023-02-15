##################################################### Train baselines
# sbatch --time=3:00:00 --array=1-5%1 --job-name=pointpillar-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_clear_FOV3000_60.yaml --tcp_port 18510 
# sbatch --time=3:00:00 --array=1-5%1 --job-name=pointpillar-adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18540 
# sbatch --time=3:00:00 --array=1-6%1 --job-name=pointpillar-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_all_FOV3000_60.yaml --tcp_port 18520 

##################################################### Test baselines
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-clear-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_clear_FOV3000_60.yaml  --tcp_port 18850 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-clear-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_clear_FOV3000_60.yaml  --tcp_port 18851 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-adverse-60-allgtbdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18856 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-adverse-60-allgtbdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18857 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_all_FOV3000_60.yaml  --tcp_port 18852 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_train_all_FOV3000_60.yaml  --tcp_port 18853 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################ Finetune
###### Naive
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=pointpillar-finetune-adverse-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18457  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth

# ##### Low LR
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-4%1 --job-name=pointpillar-finetune-adverse-all-gtdb-lowlr-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18461  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth


# ##### EWC
# sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=pointpillar-finetune-adverse-all-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18459  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth

###### Replay #Choose best lr and set it in cfg file before running this
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-fixed tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18463  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-random tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18464  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18465  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-mir-1epoch-720 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18466  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-AGEM tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18467  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-AGEM-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18469  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18468  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
# sbatch  --time=15:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-GSS tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18470  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth

sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-0p0001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_0p0001.yaml --tcp_port 18465  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-0p005 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_0p005.yaml --tcp_port 18466  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_0p003.yaml --tcp_port 18467  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-0p001 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_0p001.yaml --tcp_port 18468  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth
sbatch  --time=2:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-EMIR-0p0003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_0p0003.yaml --tcp_port 18469  --pretrained_model /OpenPCDet/checkpoints/pointpillar_train_clear_FOV3000_60_ep77.pth

############################################################## Test finetune
###### Naive
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### Low LR
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-lowlr-0p001-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-lowlr-0p001-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p001.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# ##### EWC
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pointpillar-finetune-adverse-all-gtdb-ewc-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

###### REPLAY
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-fixed-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18381 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-fixed-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18382 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-random-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18383 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-random-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18384 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-emir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18385 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-emir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18386 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-emir-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18387 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-emir-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18388 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

#sbatch --time=02:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18390 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-agem-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-agem-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-agem-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-agem-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-GSS-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=03:00:00 --array=1-1%1 --job-name=pointpillar-finetune-adverse-180clear-allgtdb-replay-GSS-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pointpillar_finetune_all_FOV3000_60_replay_GSS.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

