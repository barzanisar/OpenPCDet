##################################################### Train baselines
# sbatch --time=3:00:00 --array=1-5%1 --job-name=pvrcnn-clear-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_clear_FOV3000_60.yaml --tcp_port 18810 
# sbatch --time=3:00:00 --array=1-5%1 --job-name=pvrcnn-adverse-60-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18820 
# sbatch --time=3:00:00 --array=1-9%1 --job-name=pvrcnn-all-60 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_all_FOV3000_60.yaml --tcp_port 18830 

##################################################### Test baselines
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-clear-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18880 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-clear-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_clear_FOV3000_60.yaml  --tcp_port 18881 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-adverse-60-allgtbdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18886 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-adverse-60-allgtbdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_adverse_FOV3000_60_allgtdb.yaml  --tcp_port 18887 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-all-60-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_all_FOV3000_60.yaml  --tcp_port 18882 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=pvrcnn-all-60-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_train_all_FOV3000_60.yaml  --tcp_port 18883 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl


############################################################ Finetune
###### Naive
sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-all-gtdb tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18757  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth

##### Low LR
sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-5%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18761  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth


##### EWC
sbatch  --time=03:00:00 --gres=gpu:t4:4 --array=1-6%1 --job-name=finetune-adverse-all-gtdb-ewc tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18759  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth

###### Replay #Choose best lr and set it in cfg file before running this
sbatch  --time=24:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18763  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=24:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18764  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18765  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir-1epoch-720 tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18766  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-AGEM tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18767  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-AGEM-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18769  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth
sbatch  --time=30:00:00 --gres=gpu:t4:4 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-EMIR-plus tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18768  --pretrained_model /OpenPCDet/checkpoints/pvrcnn_train_clear_FOV3000_60_ep61.pth


############################################################## Test finetune
###### Naive
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

###### Low LR
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-lowlr-0p003-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_lowlr_0p003.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

###### EWC
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=02:00:00 --array=1-2%1 --job-name=finetune-adverse-all-gtdb-ewc-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_adverse_FOV3000_60_allgtdb_ewc.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

###### REPLAY
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18391 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-fixed-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_fixed.yaml --tcp_port 18392 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-random-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_random.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18393 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-emir-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_EMIR_plus.yaml --tcp_port 18394 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_MIR_1ep.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-mir_1ep_720-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_MIR_1ep_720.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18395 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM.yaml --tcp_port 18396 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl

# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-plus-test-clear  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18397 --fix_random_seed --test_only --eval_tag test_clear_FOV3000 --test_info_pkl dense_infos_test_clear_FOV3000_25.pkl
# sbatch --time=04:00:00 --array=1-1%1 --job-name=finetune-adverse-180clear-allgtdb-replay-agem-plus-test-adverse  tools/scripts/compute_canada_train_eval_project.sh --cfg_file tools/cfgs/dense_models/pvrcnn_finetune_all_FOV3000_60_replay_AGEM_plus.yaml --tcp_port 18398 --fix_random_seed --test_only --eval_tag test_adverse_FOV3000 --test_info_pkl dense_infos_test_adverse_FOV3000_25.pkl
