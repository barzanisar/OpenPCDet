from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler
import torch


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_boxes, points = data_dict['gt_boxes'], data_dict['points']
        enable_flag = False
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_boxes, points, enable_flag = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_boxes, points,
            )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['random_world_flip'] = enable_flag
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_boxes, points, noise_rotation = augmentor_utils.global_rotation(
            data_dict['gt_boxes'], data_dict['points'], rot_range=rot_range
        )

        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['random_world_rotation'] = noise_rotation
        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        gt_boxes, points, noise_scale = augmentor_utils.global_scaling(
            data_dict['gt_boxes'], data_dict['points'], config['WORLD_SCALE_RANGE']
        )
        data_dict['gt_boxes'] = gt_boxes
        data_dict['points'] = points
        data_dict['random_world_scaling'] = noise_scale
        return data_dict

    def random_image_depth_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        depth_maps = data_dict["depth_maps"]
        gt_boxes = data_dict['gt_boxes']
        gt_boxes2d = data_dict["gt_boxes2d"]
        calib = data_dict["calib"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, depth_maps, gt_boxes = getattr(augmentor_utils, 'random_image_depth_flip_%s' % cur_axis)(
                images, depth_maps, gt_boxes, calib,
            )

        data_dict['images'] = images
        data_dict['depth_maps'] = depth_maps
        data_dict['gt_boxes'] = gt_boxes
        return data_dict

    def random_image_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_image_flip, config=config)
        images = data_dict["images"]
        gt_boxes2d = data_dict["gt_boxes2d"]
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['horizontal']
            images, gt_boxes2d, enable_flag = getattr(augmentor_utils, 'random_image_flip_%s' % cur_axis)(
                images, gt_boxes2d
            )

        data_dict['images'] = images
        data_dict['gt_boxes2d'] = gt_boxes2d
        data_dict['image_flip'] = enable_flag
        return data_dict

    def forward(self, data_dict):
        """
        Args:
            data_dict:
                points: (N, 3 + C_in)
                gt_boxes: optional, (N, 7) [x, y, z, dx, dy, dz, heading]
                gt_names: optional, (N), string
                ...

        Returns:
        """
        # Debug reverse transformation
        DEBUG_REVERSE_TRANSFORM = False
        if DEBUG_REVERSE_TRANSFORM:
            old_points = np.copy(data_dict['points'][:,0:3])

        transformations = []
        for cur_augmentor in self.data_augmentor_queue:
            data_dict = cur_augmentor(data_dict=data_dict)
            transformations.append(cur_augmentor.keywords['config'].NAME)
        data_dict['gt_boxes'][:, 6] = common_utils.limit_period(
            data_dict['gt_boxes'][:, 6], offset=0.5, period=2 * np.pi
        )
        if 'calib' in data_dict:
            data_dict.pop('calib')
        if 'road_plane' in data_dict:
            data_dict.pop('road_plane')
        if 'gt_boxes_mask' in data_dict:
            gt_boxes_mask = data_dict['gt_boxes_mask']
            data_dict['gt_boxes'] = data_dict['gt_boxes'][gt_boxes_mask]
            data_dict['gt_names'] = data_dict['gt_names'][gt_boxes_mask]
            if 'gt_boxes2d' in data_dict:
                data_dict['gt_boxes2d'] = data_dict['gt_boxes2d'][gt_boxes_mask]

            data_dict.pop('gt_boxes_mask')

        # Compute augmentation transformation matrix
        aug_trans_matrix = np.eye(3, 3, dtype=np.float32)
        aug_trans_matrix, _ = common_utils.check_numpy_to_torch(aug_trans_matrix)
        for trans in transformations:
            if trans == 'random_world_flip' and data_dict[trans] == True:
                # assume flip around x
                flip_trans_x = np.eye(3, 3, dtype=np.float32)
                flip_trans_x, _ = common_utils.check_numpy_to_torch(flip_trans_x)
                flip_trans_x[1, 1] = -1
                aug_trans_matrix = aug_trans_matrix @ flip_trans_x
            elif trans == 'random_world_rotation':
                rotation_angle = np.array(data_dict[trans])
                rotation_trans = common_utils.get_rotation_matrix_along_z(rotation_angle)
                aug_trans_matrix = aug_trans_matrix @ rotation_trans
            elif trans == 'random_world_scaling':
                scale_val = np.array(data_dict[trans])
                scale_val, _ = common_utils.check_numpy_to_torch(scale_val)
                aug_trans_matrix = aug_trans_matrix * scale_val

        # remove unwanted global transformation entries
        if 'random_world_flip' in data_dict:
            data_dict.pop('random_world_flip')
        if 'random_world_rotation' in data_dict:
            data_dict.pop('random_world_rotation')
        if 'random_world_scaling' in data_dict:
            data_dict.pop('random_world_scaling')

        # add global transform matrix to undo augmentation for image projection
        undo_global_transform = torch.linalg.inv(aug_trans_matrix)
        data_dict['undo_global_transform'] = undo_global_transform

        # Debug reverse transformation
        if DEBUG_REVERSE_TRANSFORM:
            new_points = data_dict['points']
            points = np.array(new_points[:, 0:3])
            points, _ = common_utils.check_numpy_to_torch(points)
            before_transform = points @ undo_global_transform
            before_transform = before_transform.detach().cpu().numpy()
            max_diff = np.abs(old_points - before_transform).max()
            print('max diff is: {}'.format(max_diff))
            assert max_diff < 0.001

        return data_dict
