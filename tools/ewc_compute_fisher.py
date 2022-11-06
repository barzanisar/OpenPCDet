import argparse
import datetime
import glob
import os
from pathlib import Path
from test import repeat_eval_ckpt
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
from tensorboardX import SummaryWriter

from pcdet.config import cfg, cfg_from_list, cfg_from_yaml_file, log_config_to_file
from pcdet.datasets import build_dataloader
from pcdet.models import build_network, model_fn_decorator
from pcdet.utils import common_utils
from train_utils.optimization import build_optimizer, build_scheduler
from train_utils.train_utils import train_model
from pcdet.utils import wandb_utils

from checkpoint import init_model_from_weights
torch.backends.cudnn.enabled = False

def none_or_str(value):
    if value == 'None':
        return None
    return value

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    parser.add_argument('--batch_size', type=int, default=None, required=False, help='batch size for training')
    parser.add_argument('--epochs', type=int, default=None, required=False, help='number of epochs to train for')
    parser.add_argument('--workers', type=int, default=4, help='number of workers for dataloader')
    parser.add_argument('--extra_tag', type=str, default='default', help='extra tag for this experiment')
    parser.add_argument('--ckpt', type=none_or_str, default=None, help='checkpoint to start from')
    parser.add_argument('--pretrained_model', type=none_or_str, default=None, help='pretrained_model')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--sync_bn', action='store_true', default=False, help='whether to use sync bn')
    parser.add_argument('--fix_random_seed', action='store_true', default=False, help='')
    parser.add_argument('--ckpt_save_interval', type=int, default=5, help='number of training epochs')
    parser.add_argument('--save_ckpt_after_epoch', type=int, default=0, help='number of training epochs to save ckpt after')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--max_ckpt_save_num', type=int, default=20, help='max number of saved checkpoint')
    parser.add_argument('--merge_all_iters_to_one_epoch', action='store_true', default=False, help='')
    parser.add_argument('--set', dest='set_cfgs', default=None, nargs=argparse.REMAINDER,
                        help='set extra config keys if needed')

    parser.add_argument('--max_waiting_mins', type=int, default=0, help='max waiting minutes')
    parser.add_argument('--start_epoch', type=int, default=0, help='')
    parser.add_argument('--num_epochs_to_eval', type=int, default=0, help='number of checkpoints to be evaluated')
    parser.add_argument('--save_to_file', action='store_true', default=False, help='')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)
    cfg.TAG = Path(args.cfg_file).stem
    cfg.EXP_GROUP_PATH = args.cfg_file.split('/')[-2] #'/'.join(args.cfg_file.split('/')[1:-1])  # remove 'cfgs' and 'xxxx.yaml'

    if args.set_cfgs is not None:
        cfg_from_list(args.set_cfgs, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    if args.launcher == 'none':
        dist_train = False
        total_gpus = 1
    else:
        total_gpus, cfg.LOCAL_RANK = getattr(common_utils, 'init_dist_%s' % args.launcher)(
            args.tcp_port, args.local_rank, backend='nccl'
        )
        dist_train = True

    if args.batch_size is None:
        args.batch_size = cfg.OPTIMIZATION.BATCH_SIZE_PER_GPU
    else:
        assert args.batch_size % total_gpus == 0, 'Batch size should match the number of gpus'
        args.batch_size = args.batch_size // total_gpus

    args.epochs = cfg.OPTIMIZATION.NUM_EPOCHS if args.epochs is None else args.epochs

    if args.fix_random_seed:
        common_utils.set_random_seed(666 + cfg.LOCAL_RANK)

    output_dir = cfg.ROOT_DIR / 'output' / cfg.EXP_GROUP_PATH / cfg.TAG
    if args.extra_tag != 'default':
        output_dir = output_dir / args.extra_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    log_file = output_dir / ('log_compute_fisher_%s.txt' % datetime.datetime.now().strftime('%Y%m%d-%H%M%S'))
    logger = common_utils.create_logger(log_file, rank=cfg.LOCAL_RANK)

    # log to file
    logger.info('**********************Start logging**********************')
    gpu_list = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ.keys() else 'ALL'
    logger.info('CUDA_VISIBLE_DEVICES=%s' % gpu_list)

    if dist_train:
        logger.info('total_batch_size: %d' % (total_gpus * args.batch_size))
    for key, val in vars(args).items():
        logger.info('{:16} {}'.format(key, val))
    log_config_to_file(cfg, logger=logger)
    if cfg.LOCAL_RANK == 0:
        os.system('cp %s %s' % (args.cfg_file, output_dir))

    tb_log = SummaryWriter(log_dir=str(output_dir / 'tensorboard')) if cfg.LOCAL_RANK == 0 else None

    # -----------------------create dataloader & network & optimizer---------------------------

    cfg.DATA_CONFIG.DATA_SPLIT['train'] = cfg.EWC.PREVIOUS_DATA_SPLIT
    cfg.DATA_CONFIG.INFO_PATH['train'] = cfg.EWC.PREVIOUS_INFO_PATH
    cfg.DATA_CONFIG.DATA_AUGMENTOR.AUG_CONFIG_LIST[0].DB_INFO_PATH = cfg.EWC.PREVIOUS_DBINFO_PATH
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=args.batch_size,
        dist=dist_train, workers=args.workers,
        logger=logger,
        training=True,
        merge_all_iters_to_one_epoch=args.merge_all_iters_to_one_epoch,
        total_epochs=args.epochs,
        seed=666 if args.fix_random_seed else None
    )

    model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
    if args.sync_bn:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    # load checkpoint if it is possible
    if args.pretrained_model is not None:
        if cfg.get('LOAD_WHOLE_MODEL', False):
            ## For project
            logger.info('**********************Loading whole model**********************')
            model.load_params_from_file(filename=args.pretrained_model, to_cpu=dist_train, logger=logger) 
        else:
            logger.info('**********************Loading SSL backbone**********************')
            state = torch.load(args.pretrained_model)
            init_model_from_weights(model, state, freeze_bb=cfg.OPTIMIZATION.get('FREEZE_BB', False), logger=logger)

    optimizer = build_optimizer(model, cfg.OPTIMIZATION)

    model.train()  # before wrap to DistributedDataParallel to support fixed some parameters
    if dist_train:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[cfg.LOCAL_RANK % torch.cuda.device_count()])
    logger.info(model)
    
    # -----------------------start training---------------------------
    logger.info('**********************Start Computing Fisher %s/%s(%s)**********************'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))
    
    model_func=model_fn_decorator()
    total_it_each_epoch = len(train_loader)
    dataloader_iter = iter(train_loader)

    model.train()
    optimizer.zero_grad()

    # Accumulate gradients for one epoch
    for cur_it in tqdm(range(total_it_each_epoch)):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            dataloader_iter = iter(train_loader)
            batch = next(dataloader_iter)
            print('new iters')

        loss, _, _ = model_func(model, batch)

        loss.backward()        
    
    fisher_dict= {}
    optpar_dict = {}

    # gradients accumulated can be used to calculate fisher
    for name, param in model.named_parameters():
        optpar_dict[name] = param.data.clone()
        fisher_dict[name] = param.grad.data.clone().pow(2)

    output_dict = {'fisher' :fisher_dict, 'optpar': optpar_dict}

    # save fisher and opt param means 
    if cfg.LOCAL_RANK == 0:
        ewc_param_path = output_dir / 'ewc_params.pkl'
        with open(ewc_param_path, 'wb') as f:
            pickle.dump(output_dict, f)
    
    if hasattr(train_set, 'use_shared_memory') and train_set.use_shared_memory:
        train_set.clean_shared_memory()

    logger.info('**********************End Compute Fisher %s/%s(%s)**********************\n\n\n'
                % (cfg.EXP_GROUP_PATH, cfg.TAG, args.extra_tag))


if __name__ == '__main__':
    main()
