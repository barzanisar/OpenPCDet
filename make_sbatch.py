import glob
from pathlib import Path 

mode = 'Test'
root_path = '/home/barza/OpenPCDet'
cfg_dir_path = f'{root_path}/tools/cfgs/dense_models'
tcp_port = 18800
pretrained_model='dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000_b14_lr0p15_checkpoint-ep340.pth.tar'
extra_tag='dc_vdc_snow1in10_wet_fog1in10_cubeF_upsample_FOV3000_b14_lr0p15_ep340'
test_split = ['clear']

cfg_paths = glob.glob(f'{cfg_dir_path}/pointrcnn_finetune_train_all_FOV3000_60_50_*.yaml')

if mode == 'Train':
    sbatch_file = f'{root_path}/finetune.sh'

    with open(sbatch_file, 'w') as f:
        f.write('#Finetune-Tuning\n')

    for cfg_path in cfg_paths:
        cfg_name=cfg_path.split('/')[-1].replace('.yaml', '')
        job_name=cfg_name.split('pointrcnn_finetune_train_all_FOV3000_60_50_')[-1]
        tcp_port +=1
        sbatch_cmd = f'sbatch --time=4:00:00 --array=1-1%1 --job-name=finetune-{job_name} --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh'
        sbatch_cmd += f' --tcp_port {tcp_port} --cfg_file --cfg_file tools/cfgs/dense_models/{cfg_name}.yaml'
        sbatch_cmd += f' --train_batch_size 4 --ckpt_save_interval 1 --max_ckpt_save_num 5 --fix_random_seed'
        sbatch_cmd += f' --extra_tag {extra_tag} --pretrained_model /OpenPCDet/checkpoints/pointnet_train_all_FOV3000_60/{pretrained_model}\n' 

        with open(sbatch_file, 'a') as f:
            f.write(sbatch_cmd)

elif mode == 'Test':
    sbatch_file = f'{root_path}/finetune-test.sh'

    with open(sbatch_file, 'w') as f:
        f.write('#Test Finetune-Tuning\n')

    for weather in test_split:
        for cfg_path in cfg_paths:
            cfg_name=cfg_path.split('/')[-1].replace('.yaml', '')
            job_name=cfg_name.split('pointrcnn_finetune_train_all_FOV3000_60_50_')[-1]
            tcp_port +=1
            sbatch_cmd = f'sbatch --time=1:00:00 --array=1-1%1 --job-name=finetune-{job_name}-test-clear --mail-user=barzanisar93@gmail.com tools/scripts/compute_canada_train_eval_density_det.sh'
            sbatch_cmd += f' --tcp_port {tcp_port} --cfg_file --cfg_file tools/cfgs/dense_models/{cfg_name}.yaml'
            sbatch_cmd += f' --test_batch_size 4 --test_only --fix_random_seed'
            sbatch_cmd += f' --extra_tag {extra_tag} --eval_tag test_{weather}_FOV3000 --test_info_pkl dense_infos_test_{weather}_FOV3000_25.pkl\n' 

            with open(sbatch_file, 'a') as f:
                f.write(sbatch_cmd)