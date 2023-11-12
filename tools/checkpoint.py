#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import logging
import os
import tempfile
import traceback
from shutil import copy2, move

import torch

def replace_module_suffix(state_dict, suffix, replace_with=""):
    """
    Replace suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {
        (key.replace(suffix, replace_with, 1) if key.startswith(suffix) else key): val
        for (key, val) in state_dict.items()
    }
    return state_dict


def append_module_suffix(state_dict, suffix):
    """
    Append suffixes in a state_dict
    Needed when loading DDP or classy vision models
    """
    state_dict = {f"{suffix}{key}": val for (key, val) in state_dict.items()}
    return state_dict

def init_model_from_weights(
    model,
    state_dict,
    logger,
    state_dict_key_name="model",
    skip_layers=None,
    print_init_layers=True,
    freeze_bb=False,
    rank=0
):
    """
    Initialize the model from any given params file. This is particularly useful
    during the finetuning process or when we want to evaluate a model on a range
    of tasks.
    skip_layers:     string : layer names with this key are not copied
    replace_suffix: string : remove these suffixes from the layer names
    print_init_layers:   print whether layer was init or ignored
                    indicates whether the layername was copied or not
    """
    # whether it's a model from somewhere else or a model from this codebase
    if state_dict_key_name and len(state_dict_key_name) > 0:
        #state_dict = state_dict["model_state_dict"]
        assert (
            state_dict_key_name in state_dict.keys()
        ), f"Unknown state dict key: {state_dict_key_name}"
        state_dict = state_dict[state_dict_key_name]

    all_layers = model.state_dict()
    init_layers = {layername: False for layername in all_layers}

    new_state_dict = {}
    # Change  names of ssl model to match that of opdcmodel, select only those layers from ssl model that are present in opdmodel
    #For all 'backbone3d' layers in opcdet model, if same layers exists in ssl state_dict, copy them to new_state_dict with keysname same as the opdcet model's backbone3d
    for param_name in init_layers:
        if 'global_step' in param_name: #skip copying global step
            continue
        if "module.trunk.0."+param_name in state_dict:
            new_state_dict[param_name] = state_dict["module.trunk.0."+param_name] # this layer will be transfered to opdmodel
        elif "module.det_head."+param_name in state_dict:
            new_state_dict[param_name] = state_dict["module.det_head."+param_name] # this layer will be transfered to opdmodel
        else:
            logger.info(f"{param_name} not found in ssl model!")
    state_dict = new_state_dict
            
    #local_rank = int(os.environ.get("LOCAL_RANK", 0))
    not_found, not_init = [], []
    for layername in all_layers.keys():
        if (
            skip_layers and len(skip_layers) > 0 and layername.find(skip_layers) >= 0
        ) or layername.find("num_batches_tracked") >= 0:
            if print_init_layers and (rank == 0):
                not_init.append(layername)
                logger.info(f"Ignored layer:\t{layername}")
            continue
        if layername in state_dict:
            param = state_dict[layername]
            if not isinstance(param, torch.Tensor):
                param = torch.from_numpy(param)
            all_layers[layername].copy_(param)
            init_layers[layername] = True
            if print_init_layers and (rank == 0):
                logger.info(f"Init layer:\t{layername}")
        else:
            not_found.append(layername)
            if print_init_layers and (rank == 0):
                logger.info(f"Not found:\t{layername}")
    ####################### DEBUG ############################
    # _print_state_dict_shapes(model.state_dict())
    if freeze_bb:
        logger.info('Freezing Backbone!')
        for name, param in model.named_parameters():
            if 'backbone_3d' in name:
                param.requires_grad = False

    torch.cuda.empty_cache()
    return model
