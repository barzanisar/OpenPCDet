from functools import partial

import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

from .fastai_optim import OptimWrapper
from .learning_schedules_fastai import CosineWarmupLR, OneCycle


def build_optimizer(model, optim_cfg):
    parameters = None
    if optim_cfg.get("LR_BB", None) is not None:
        print("Use different learning rates between the head and trunk.")

        param_group_head = [
            param for name, param in model.named_parameters()
            if param.requires_grad and 'backbone_3d.' not in name]
        param_group_trunk = [
            param for name, param in model.named_parameters()
            if param.requires_grad and 'backbone_3d.' in name]
        param_group_all = [
            param for key, param in model.named_parameters()
            if param.requires_grad]
        assert len(param_group_all) == (len(param_group_head) + len(param_group_trunk))

        # weight_decay = optim_cfg["WEIGHT_DECAY"]
        # weight_decay_head = optim_cfg["WEIGHT_DECAY_head"] if (optim_cfg["WEIGHT_DECAY_head"] is not None) else weight_decay
        
        parameters = [
            {"params": param_group_head, "lr": optim_cfg.LR}
            ]
        if len(param_group_trunk) > 0: # not optim_cfg.FREEZE_BB
            parameters.append({"params": param_group_trunk, "lr": optim_cfg.LR_BB})
        print(f"==> Head:  #{len(param_group_head)} params with learning rate: {optim_cfg.LR}")
        print(f"==> Trunk: #{len(param_group_trunk)} params with learning rate: {optim_cfg.LR_BB}")

    if optim_cfg.OPTIMIZER == 'adam':
        optimizer = optim.Adam(model.parameters() if parameters is None else parameters, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY, betas=(0.9, 0.99))
    elif optim_cfg.OPTIMIZER == 'adamW':
        optimizer = optim.AdamW(model.parameters() if parameters is None else parameters, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY, betas=(0.9, 0.99))
    elif optim_cfg.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            model.parameters() if parameters is None else parameters, lr=optim_cfg.LR, weight_decay=optim_cfg.WEIGHT_DECAY,
            momentum=optim_cfg.MOMENTUM
        )
    elif optim_cfg.OPTIMIZER == 'adam_onecycle':
        def children(m: nn.Module):
            return list(m.children())

        def num_children(m: nn.Module) -> int:
            return len(children(m))

        flatten_model = lambda m: sum(map(flatten_model, m.children()), []) if num_children(m) else [m]
        get_layer_groups = lambda m: [nn.Sequential(*flatten_model(m))]
        #get_layer_groups(model)[0]
        #get_layer_groups(children(model)[1])[0]
        #get_layer_groups(children(model)[2])[0]
        #len(get_layer_groups(children(model)[0])[0]) + len(get_layer_groups(children(model)[1])[0]) + len(get_layer_groups(children(model)[2])[0])

        optimizer_func = partial(optim.Adam, betas=(0.9, 0.99))
        optimizer = OptimWrapper.create(
            optimizer_func, 3e-3, get_layer_groups(model), wd=optim_cfg.WEIGHT_DECAY, true_wd=True, bn_wd=True
        )
    else:
        raise NotImplementedError

    return optimizer


def build_scheduler(optimizer, total_iters_each_epoch, total_epochs, last_epoch, optim_cfg):
    lr_warmup_scheduler = None
    total_steps = total_iters_each_epoch * total_epochs
    if optim_cfg.OPTIMIZER == 'adam_onecycle':
        lr_scheduler = OneCycle(
            optimizer, total_steps, optim_cfg.LR, list(optim_cfg.MOMS), optim_cfg.DIV_FACTOR, optim_cfg.PCT_START
        )
    else:
        if optim_cfg.get("LR_SCHEDULER"):
            if  optim_cfg["LR_SCHEDULER"] == 'steplr':
                print('Scheduler: StepLR')
                lr_scheduler = lr_sched.StepLR(
                    optimizer, int(.9 * optim_cfg["NUM_EPOCHS"]),
                )
            elif optim_cfg["LR_SCHEDULER"] == 'cosine':
                print('Scheduler: Cosine')
                lr_scheduler = lr_sched.CosineAnnealingLR(
                    optimizer, optim_cfg["NUM_EPOCHS"]
                )
            elif optim_cfg["LR_SCHEDULER"] == 'onecyle':
                print('Scheduler: Onecyle')
                if len(optimizer.param_groups) < 2:
                    max_lr = optim_cfg.LR
                else:
                    max_lr=[optim_cfg.LR, optim_cfg.LR_BB]
                lr_scheduler = lr_sched.OneCycleLR(optimizer, max_lr=max_lr, total_steps=total_steps, 
                pct_start=optim_cfg.PCT_START, anneal_strategy='cos', div_factor=optim_cfg.DIV_FACTOR, verbose=False)
        else:
            decay_steps = [x * total_iters_each_epoch for x in optim_cfg.DECAY_STEP_LIST]
            def lr_lbmd_head(cur_epoch):
                cur_decay = 1
                for decay_step in decay_steps:
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * optim_cfg.LR_DECAY
                return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR)

            def lr_lbmd_bb(cur_epoch):
                cur_decay = 1
                for decay_step in decay_steps:
                    if cur_epoch >= decay_step:
                        cur_decay = cur_decay * optim_cfg.LR_DECAY
                return max(cur_decay, optim_cfg.LR_CLIP / optim_cfg.LR_BB)

            if len(optimizer.param_groups) < 2:
                lr_lambda = lr_lbmd_head
            else:
                lr_lambda=[lr_lbmd_head, lr_lbmd_bb]

            lr_scheduler = lr_sched.LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=last_epoch)

        if optim_cfg.LR_WARMUP:
            lr_warmup_scheduler = CosineWarmupLR(
                optimizer, T_max=optim_cfg.WARMUP_EPOCH * len(total_iters_each_epoch),
                eta_min=optim_cfg.LR / optim_cfg.DIV_FACTOR
            )

    return lr_scheduler, lr_warmup_scheduler
