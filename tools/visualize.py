import argparse
import glob
from pathlib import Path
from functools import partial

import mayavi.mlab as mlab
import numpy as np
import torch

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate, build_dataloader
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils
from visual_utils import visualize_utils as V
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils

hooked = {}


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default='cfgs/kitti_models/second.yaml',
                        help='specify the config for demo')
    parser.add_argument('--ckpt', type=str, default=None, help='specify the pretrained model')
    parser.add_argument('--testing', action='store_true', default=False, help='')
    parser.add_argument('--index', type=int, default=None, help='Choose the index')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    return args, cfg


def main():
    args, cfg = parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    # torch.manual_seed(0)
    train_set, train_loader, train_sampler = build_dataloader(
        dataset_cfg=cfg.DATA_CONFIG,
        class_names=cfg.CLASS_NAMES,
        batch_size=1,
        dist=False, workers=8,
        logger=logger,
        training=not args.testing
    )
    if args.ckpt:
        model = build_network(model_cfg=cfg.MODEL, num_class=len(cfg.CLASS_NAMES), dataset=train_set)
        model.load_params_from_file(filename=args.ckpt, logger=logger, to_cpu=True)
        model.cuda()
        model.eval()

        with torch.no_grad():
            index = args.index if args.index is not None else np.random.choice(len(train_set))
            data_dict = train_set[index]
            data_dict = train_set.collate_batch([data_dict])
            load_data_to_gpu(data_dict)
            pred_dicts, _ = model.forward(data_dict)

            # Print out batch info
            print('index: {}, frame_id: {}, num_boxes: {}'.format(index, data_dict['frame_id'], len(data_dict['gt_boxes'][0])))

            V.draw_scenes(
                points=data_dict['points'][:, 1:], ref_boxes=pred_dicts[0]['pred_boxes'],
                ref_scores=pred_dicts[0]['pred_scores'], ref_labels=pred_dicts[0]['pred_labels'],
                gt_boxes=data_dict['gt_boxes'][0],
            )
            mlab.show(stop=True)
    else:
        index = args.index if args.index is not None else np.random.choice(len(train_set))
        data_dict = train_set[index]
        data_dict = train_set.collate_batch([data_dict])

        print('index: {}, frame_id: {}'.format(index, data_dict['frame_id']))

        V.draw_scenes(points=data_dict['points'][:, 1:], gt_boxes=data_dict['gt_boxes'][0])
        mlab.show(stop=True)

    logger.info('Vis done.')


if __name__ == '__main__':
    main()
