# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import numpy as np
import torch
import multiprocessing
import SharedArray
import torch.distributed as dist
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from functools import partial
# from ..simulator.vis import open3d_vis_utils as V
# from ..simulator.atmos_models import LISA
# from ..simulator.simulate_rain_waymo import *


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()] #read split.txt
        self.pseudo_classes_cfg = self.dataset_cfg.get('PSEUDO_CLASSES_CFG', None)
        self.approx_boxes_path = self.root_path / (self.dataset_cfg.PROCESSED_DATA_TAG + '_clustered')

        self.infos = []
        self.include_waymo_data(self.mode) # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos

        if self.pseudo_classes_cfg is not None:
            self.add_pseudo_classes()

        self.use_shared_memory = self.dataset_cfg.get('USE_SHARED_MEMORY', False) and self.training
        if self.use_shared_memory:
            self.shared_memory_file_limit = self.dataset_cfg.get('SHARED_MEMORY_FILE_LIMIT', 0x7FFFFFFF)
            self.load_data_to_shared_memory()

    def set_split(self, split):
        super().__init__(
            dataset_cfg=self.dataset_cfg, class_names=self.class_names, training=self.training,
            root_path=self.root_path, logger=self.logger
        )
        self.split = split
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        self.infos = []
        self.include_waymo_data(self.mode)
    
    def add_pseudo_classes(self):
        pseudo_class_cnts = np.zeros(len(self.pseudo_classes_cfg.PSEUDO_CLASSES))
        mean_box_sizes = self.pseudo_classes_cfg.get('PSEUDO_MEAN_SIZES', None)

        
        if mean_box_sizes is not None and len(self.pseudo_classes_cfg.PSEUDO_CLASSES) > 1:
            mean_box_sizes = np.array(mean_box_sizes)

        for info in self.infos:
            info['annos'] = common_utils.drop_info_with_name(info['annos'], name='unknown')
            approx_boxes = info['approx_boxes_closeness_to_edge'][np.logical_not(info['approx_boxes_overlaps_gt_box'])]

            
            # assign psuedo classes according to mean sizes
            if mean_box_sizes is not None and len(self.pseudo_classes_cfg.PSEUDO_CLASSES) > 1:
                lwh = approx_boxes[:, 3:6]
                l = np.max(lwh[:,:2], axis=1)
                w = np.min(lwh[:,:2], axis=1)
                lwh[:,0] = l
                lwh[:,1] = w
                
                dist = (((mean_box_sizes.reshape(1, -1, 3) - \
                lwh.reshape(-1, 1, 3)) ** 2).sum(axis=2))  # N=boxes x M=mean sizes 
                idx_matched_mean_sizes = dist.argmin(axis=1) # N gt boxes
                approx_names = np.array(self.pseudo_classes_cfg.PSEUDO_CLASSES)[idx_matched_mean_sizes]
                unique_idx, counts = np.unique(idx_matched_mean_sizes, return_counts=True)
                pseudo_class_cnts[unique_idx] += counts
            else:
                approx_names = [self.pseudo_classes_cfg.PSEUDO_CLASSES[0]] * len(approx_boxes)
                pseudo_class_cnts[0] += len(approx_boxes)
            
            info['annos']['name'] = np.concatenate((info['annos']['name'], approx_names))
            info['annos']['gt_boxes_lidar'] = np.concatenate((info['annos']['gt_boxes_lidar'], approx_boxes[:,:7]))
            info['annos']['difficulty'] = np.concatenate((info['annos']['difficulty'], np.zeros(len(approx_boxes), dtype=np.int)))
            info['annos']['num_points_in_gt'] = np.concatenate((info['annos']['num_points_in_gt'], approx_boxes[:,16]))
        
        self.logger.info('Pseudo Class Counts:')
        for i in range(len(pseudo_class_cnts)):
            self.logger.info(f'{self.pseudo_classes_cfg.PSEUDO_CLASSES[i]}: {pseudo_class_cnts[i]}')    
        
    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            if self.pseudo_classes_cfg is not None:
                info_path = self.approx_boxes_path / sequence_name / ('approx_boxes.pkl')
            else:
                info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f) # loads 20 infos for one seq pkl i.e. 20 frames if seq pkl was formed by sampling every 10th frame
                waymo_infos.extend(infos) # each info is one frame

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos))) # total frames

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1: # not needed since our pkl file was already sampled at 10
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    def load_data_to_shared_memory(self):
        self.logger.info(f'Loading training data to shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            points = self.get_lidar(sequence_name, sample_idx)
            common_utils.sa_create(f"shm://{sa_key}", points)

        dist.barrier()
        self.logger.info('Training data has been saved to shared memory')

    def clean_shared_memory(self):
        self.logger.info(f'Clean training data from shared memory (file limit={self.shared_memory_file_limit})')

        cur_rank, num_gpus = common_utils.get_dist_info()
        all_infos = self.infos[:self.shared_memory_file_limit] \
            if self.shared_memory_file_limit < len(self.infos) else self.infos
        cur_infos = all_infos[cur_rank::num_gpus]
        for info in cur_infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            sa_key = f'{sequence_name}___{sample_idx}'
            if not os.path.exists(f"/dev/shm/{sa_key}"):
                continue

            SharedArray.delete(f"shm://{sa_key}")

        if num_gpus > 1:
            dist.barrier()
        self.logger.info('Training data has been deleted from shared memory')

    # @staticmethod
    # def check_sequence_name_with_all_version(sequence_file):
    #     if 'sim_rain' in str(sequence_file):
    #         return sequence_file
    #     if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
    #         sequence_file = Path(str(sequence_file[:-9]) + '_with_camera_labels.tfrecord')
    #     if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
    #         sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

    #     return sequence_file
        
    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if not sequence_file.exists():
            found_sequence_file = sequence_file
            for pre_text in ['training', 'validation', 'testing']:
                if not sequence_file.exists():
                    temp_sequence_file = Path(str(sequence_file).replace('segment', pre_text + '_segment'))
                    if temp_sequence_file.exists():
                        found_sequence_file = temp_sequence_file
                        break
            if not found_sequence_file.exists():
                found_sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))
            if found_sequence_file.exists():
                sequence_file = found_sequence_file
        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1, use_two_returns=True, only_extract_seg_labels=False):
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label, use_two_returns=use_two_returns, only_extract_seg_labels=only_extract_seg_labels
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        with multiprocessing.Pool(num_workers) as p:
            sequence_infos = list(tqdm(p.imap(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))

        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
        if not self.dataset_cfg.get('DISABLE_NLZ_FLAG_ON_POINTS', False):
            points_all = points_all[NLZ_flag == -1]
        if 'sim_rain' not in sequence_name:
            points_all[:, 3] = np.tanh(points_all[:, 3])
        return points_all

    def __len__(self):
        if self._merge_all_iters_to_one_epoch:
            return len(self.infos) * self.total_epochs

        return len(self.infos)

    def __getitem__(self, index):
        if self._merge_all_iters_to_one_epoch:
            index = index % len(self.infos)

        info = copy.deepcopy(self.infos[index])
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        if self.use_shared_memory and index < self.shared_memory_file_limit:
            sa_key = f'{sequence_name}___{sample_idx}'
            points = SharedArray.attach(f"shm://{sa_key}").copy()
        else:
            points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            if self.pseudo_classes_cfg is None:
                annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

            if self.training and self.dataset_cfg.get('FILTER_EMPTY_BOXES_FOR_TRAIN', False):
                mask = (annos['num_points_in_gt'] > 0)  # filter empty boxes
                annos['name'] = annos['name'][mask]
                gt_boxes_lidar = gt_boxes_lidar[mask]
                annos['num_points_in_gt'] = annos['num_points_in_gt'][mask]

            input_dict.update({
                'gt_names': annos['name'],
                'gt_boxes': gt_boxes_lidar,
                'num_points_in_gt': annos.get('num_points_in_gt', None)
            })

        data_dict = self.prepare_data(data_dict=input_dict)
        data_dict['metadata'] = info.get('metadata', info['frame_id'])
        data_dict.pop('num_points_in_gt', None)
        return data_dict

    @staticmethod
    def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
        """
        Args:
            batch_dict:
                frame_id:
            pred_dicts: list of pred_dicts
                pred_boxes: (N, 7), Tensor
                pred_scores: (N), Tensor
                pred_labels: (N), Tensor
            class_names:
            output_path:

        Returns:

        """

        def get_template_prediction(num_samples):
            ret_dict = {
                'name': np.zeros(num_samples), 'score': np.zeros(num_samples),
                'boxes_lidar': np.zeros([num_samples, 7])
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict['pred_scores'].cpu().numpy()
            pred_boxes = box_dict['pred_boxes'].cpu().numpy()
            pred_labels = box_dict['pred_labels'].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict['name'] = np.array(class_names)[pred_labels - 1]
            pred_dict['score'] = pred_scores
            pred_dict['boxes_lidar'] = pred_boxes

            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict['frame_id'] = batch_dict['frame_id'][index]
            single_pred_dict['metadata'] = batch_dict['metadata'][index]
            annos.append(single_pred_dict)

        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        if 'annos' not in self.infos[0].keys():
            return 'No ground-truth boxes for evaluation', {}

        def kitti_eval(eval_det_annos, eval_gt_annos):
            from ..kitti.kitti_object_eval_python import eval as kitti_eval
            from ..kitti import kitti_utils

            map_name_to_kitti = {
                'Vehicle': 'Car',
                'Pedestrian': 'Pedestrian',
                'Cyclist': 'Cyclist',
                'Sign': 'Sign',
                'Car': 'Car'
            }

            if self.pseudo_classes_cfg is not None:
                map_name_to_kitti.update({name: name for name in self.pseudo_classes_cfg.PSEUDO_CLASSES})
                pseudo_class_dict = {'names': self.pseudo_classes_cfg.PSEUDO_CLASSES ,'min_ious': self.pseudo_classes_cfg.MIN_IOU_OVERLAP}

            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names, 
                pseudo_class_dict=pseudo_class_dict  if self.pseudo_classes_cfg is not None else None
            )
            return ap_result_str, ap_dict

        # def waymo_eval(eval_det_annos, eval_gt_annos, eval_levels_list_cfg=None):
        #     from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
        #     eval = OpenPCDetWaymoDetectionMetricsEstimator()
        # 
        #     # Overall Evaluation
        #     ap_dict = eval.waymo_evaluation(
        #         eval_det_annos, eval_gt_annos, class_name=class_names,
        #         distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        #     )
        #     ap_result_str = '\n'
        #     for key in ap_dict:
        #         ap_dict[key] = ap_dict[key][0]
        #         ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
        # 
        #     # Evaluation across range
        #     if eval_levels_list_cfg:
        #         ap_result_str_list, ap_dict_list = [], []
        #         ap_result_str_list.append(ap_result_str)
        #         ap_dict_list.append(ap_dict)
        # 
        #         for i in range(len(eval_levels_list_cfg['RANGE_LIST']) - 1):
        #             lower_bound, upper_bound = eval_levels_list_cfg['RANGE_LIST'][i], \
        #                                        eval_levels_list_cfg['RANGE_LIST'][i + 1]
        # 
        #             ap_dict = eval.waymo_evaluation(
        #                 eval_det_annos, eval_gt_annos, class_name=class_names,
        #                 distance_thresh=upper_bound, lower_bound=lower_bound,
        #                 fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
        #             )
        #             ap_result_str = '\n'
        #             for key in ap_dict:
        #                 ap_dict[key] = ap_dict[key][0]
        #                 ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])
        # 
        #             ap_result_str_list.append(ap_result_str)
        #             ap_dict_list.append(ap_dict)
        # 
        #         return ap_result_str_list, ap_dict_list
        # 
        #     return ap_result_str, ap_dict

        def waymo_eval(eval_det_annos, eval_gt_annos):
            from .waymo_eval import OpenPCDetWaymoDetectionMetricsEstimator
            eval = OpenPCDetWaymoDetectionMetricsEstimator()

            ap_dict = eval.waymo_evaluation(
                eval_det_annos, eval_gt_annos, class_name=class_names,
                distance_thresh=1000, fake_gt_infos=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            ap_result_str = '\n'
            for key in ap_dict:
                ap_dict[key] = ap_dict[key][0]
                ap_result_str += '%s: %.4f \n' % (key, ap_dict[key])

            return ap_result_str, ap_dict

        eval_det_annos = copy.deepcopy(det_annos)
        eval_gt_annos = [copy.deepcopy(info['annos']) for info in self.infos]

        if kwargs['eval_metric'] == 'kitti':
            ap_result_str, ap_dict = kitti_eval(eval_det_annos, eval_gt_annos)
        elif kwargs['eval_metric'] == 'waymo':
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos)
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict

    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                    processed_data_tag=None):
        database_save_path = save_path / ('%s_gt_database_%s' % (processed_data_tag, split))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
        # db_data_save_path = save_path / ('%s_gt_database_%s_sampled_%d_global.npy' % (processed_data_tag, split, sampled_interval))
        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        point_offset_cnt = 0
        stacked_gt_points = []
        for k in range(0, len(infos), sampled_interval):
            print('gt_database sample: %d/%d' % (k + 1, len(infos)))
            info = infos[k]

            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = self.get_lidar(sequence_name, sample_idx)

            annos = info['annos']
            names = annos['name']
            difficulty = annos['difficulty']
            gt_boxes = annos['gt_boxes_lidar']

            if k % 4 != 0 and len(names) > 0:
                mask = (names == 'Vehicle')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            if k % 2 != 0 and len(names) > 0:
                mask = (names == 'Pedestrian')
                names = names[~mask]
                difficulty = difficulty[~mask]
                gt_boxes = gt_boxes[~mask]

            num_obj = gt_boxes.shape[0]
            if num_obj == 0:
                continue

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    gt_points = gt_points.astype(np.float32)
                    assert gt_points.dtype == np.float32
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                    # it will be used if you choose to use shared memory for gt sampling
                    stacked_gt_points.append(gt_points)
                    db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                    point_offset_cnt += gt_points.shape[0]

                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

        # # it will be used if you choose to use shared memory for gt sampling
        # stacked_gt_points = np.concatenate(stacked_gt_points, axis=0)
        # np.save(db_data_save_path, stacked_gt_points)

    def create_gt_database_of_single_scene(self, info_with_idx, database_save_path=None, used_classes=None, total_samples=0):
        info, info_idx = info_with_idx
        print('gt_database sample: %d/%d' % (info_idx, total_samples))

        all_db_infos = {}
        point_offset_cnt = 0

        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        points = self.get_lidar(sequence_name, sample_idx)

        annos = info['annos']
        names = annos['name']
        difficulty = annos['difficulty']
        gt_boxes = annos['gt_boxes_lidar']

        if info_with_idx % 4 != 0 and len(names) > 0:
            mask = (names == 'Vehicle')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        if info_with_idx % 2 != 0 and len(names) > 0:
            mask = (names == 'Pedestrian')
            names = names[~mask]
            difficulty = difficulty[~mask]
            gt_boxes = gt_boxes[~mask]

        num_obj = gt_boxes.shape[0]
        if num_obj == 0:
            return {}

        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
            torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
            torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
        ).long().squeeze(dim=0).cpu().numpy()

        for i in range(num_obj):
            filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
            filepath = database_save_path / filename
            gt_points = points[box_idxs_of_pts == i]
            gt_points[:, :3] -= gt_boxes[i, :3]

            if (used_classes is None) or names[i] in used_classes:
                gt_points = gt_points.astype(np.float32)
                assert gt_points.dtype == np.float32
                with open(filepath, 'w') as f:
                    gt_points.tofile(f)

                db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                            'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                            'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}

                # it will be used if you choose to use shared memory for gt sampling
                db_info['global_data_offset'] = [point_offset_cnt, point_offset_cnt + gt_points.shape[0]]
                point_offset_cnt += gt_points.shape[0]

                if names[i] in all_db_infos:
                    all_db_infos[names[i]].append(db_info)
                else:
                    all_db_infos[names[i]] = [db_info]
        
        return all_db_infos
    
    def create_groundtruth_database_parallel(self, info_path, save_path, used_classes=None, split='train', sampled_interval=1,
                                             processed_data_tag=None, num_workers=16):
        database_save_path = save_path / ('%s_gt_database_%s' % (processed_data_tag, split))
        db_info_save_path = save_path / ('%s_waymo_dbinfos_%s_sampled_%d.pkl' % (processed_data_tag, split, sampled_interval))
            
        database_save_path.mkdir(parents=True, exist_ok=True)

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

        if sampled_interval > 1:
            sampled_infos=[]
            for k in range(0, len(infos), sampled_interval):
                sampled_infos.append(infos[k])
            infos = sampled_infos
        
        print(f'Number workers: {num_workers}')
        create_gt_database_of_single_scene = partial(
            self.create_gt_database_of_single_scene, database_save_path=database_save_path,
            used_classes=used_classes, total_samples=len(infos)
        )
        # create_gt_database_of_single_scene((infos[300], 0))
        with multiprocessing.Pool(num_workers) as p:
            all_db_infos_list = list(p.map(create_gt_database_of_single_scene, zip(infos, np.arange(len(infos)))))

        all_db_infos = {}

        for cur_db_infos in all_db_infos_list:
            for key, val in cur_db_infos.items():
                if key not in all_db_infos:
                    all_db_infos[key] = val
                else:
                    all_db_infos[key].extend(val)

        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)

def create_waymo_gt_database(
    dataset_cfg, class_names, data_path, save_path, processed_data_tag='waymo_processed_data',
    workers=min(16, multiprocessing.cpu_count()), use_parallel=False):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split = dataset_cfg.DATA_SPLIT['train']
    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)

    if use_parallel:
        dataset.create_groundtruth_database_parallel(
            info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
            used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag,
            num_workers=workers
        )
    else:
        dataset.create_groundtruth_database(
            info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
            used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
        )
    print('---------------Data preparation Done---------------')

def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=min(16, multiprocessing.cpu_count()), only_extract_seg_labels=False):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = dataset_cfg.DATA_SPLIT['train'], dataset_cfg.DATA_SPLIT['test']

    train_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, train_split))
    val_filename = save_path / ('%s_infos_%s.pkl' % (processed_data_tag, val_split))

    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, # 10 to make waymo_processed_data_10 and 1 to make gtdb
        use_two_returns=dataset_cfg.USE_TWO_RETURNS,
        only_extract_seg_labels=only_extract_seg_labels
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)

    dataset.set_split(val_split)
    waymo_infos_val = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1, # 10 to make waymo_processed_data_10
        use_two_returns=dataset_cfg.USE_TWO_RETURNS,
        only_extract_seg_labels=only_extract_seg_labels
    )
    with open(val_filename, 'wb') as f:
        pickle.dump(waymo_infos_val, f)
    print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    # print('---------------Start create groundtruth database for data augmentation---------------')
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    # dataset.set_split(train_split)
    # dataset.create_groundtruth_database(
    #     info_path=train_filename, save_path=save_path, split='train', sampled_interval=1,
    #     used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    # )

    print('---------------Data preparation Done---------------')

# def simulate_rain(clear_weather_split, sim_percent, sim_data_tag, data_path, processed_data_tag):
#     simulator = LISA(rmax=200)
#     rain_rates = np.linspace(0.5, 10.0, 20)
#
#     data_save_path = data_path / 'rainy_processed_data'
#     data_save_path.mkdir(parents=True, exist_ok=True)
#     clear_data_path = data_path / processed_data_tag
#
#     for split, percent in zip(clear_weather_split, sim_percent):
#         split_dir = data_path / 'ImageSets' / (split + '.txt')
#         clear_sequence_list = [x.strip() for x in open(split_dir).readlines()]
#         sim_upto_len = int(percent * len(clear_sequence_list))
#         #rainy_split = split + '_' + sim_data_tag
#
#         # # Create train_rainy.txt (50% rainy and 50% clear) and val_rainy.txt
#         # dst_txt = data_path / 'ImageSets' / (rainy_split + '.txt')
#         # with open(dst_txt, 'w') as f:
#         #     for i in range(sim_upto_len):
#         #         seq_name = clear_sequence_list[i]
#         #         # create a rainy name for this seq
#         #         if '_with_camera_labels' in seq_name:
#         #             rain_seq_name = seq_name.split('_with_camera_labels')[
#         #                                 0] + '_' + sim_data_tag + '_with_camera_labels' + \
#         #                             seq_name.split('_with_camera_labels')[1]
#         #         else:
#         #             rain_seq_name = seq_name.split('.')[0] + '_' + sim_data_tag + seq_name.split('.')[1]
#         #
#         #         f.write(rain_seq_name)
#         #         if i != len(clear_sequence_list) - 1:
#         #             f.write('\n')
#         #     for i in range(sim_upto_len, len(clear_sequence_list)):
#         #         f.write(clear_sequence_list[i])
#         #         if i != len(clear_sequence_list) - 1:
#         #             f.write('\n')
#
#
#         for i in tqdm(range(sim_upto_len)): #, desc='Number of sequences processed'
#             seq_name = clear_sequence_list[i].split('.')[0]
#
#             # create a rainy name for this seq
#             if '_with_camera_labels' in seq_name:
#                 rain_seq_name = seq_name.split('_with_camera_labels')[0] + '_' + sim_data_tag + '_with_camera_labels'
#             else:
#                 rain_seq_name = seq_name.split('.')[0] + '_' + sim_data_tag
#
#             clear_seq_dir = clear_data_path / seq_name
#             rainy_seq_dir = data_save_path / rain_seq_name
#
#             # Copy clear processed seq into a new rainy seq dir
#             shutil.copytree(clear_seq_dir, rainy_seq_dir, dirs_exist_ok=True)
#             seq_info_path = rainy_seq_dir / (seq_name + '.pkl')
#             rainy_seq_info_path = rainy_seq_dir / (rain_seq_name + '.pkl')
#             with open(seq_info_path, 'rb') as f:
#                 seq_infos = pickle.load(f)
#
#             lidar_files = sorted(glob.glob(str(rainy_seq_dir) + "/*.npy"))
#             for file, info in tqdm(zip(lidar_files, seq_infos)): #, desc=f'Number of frames in one seq: {rain_seq_name}'
#                 points = np.load(file)  # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
#                 points[:,3] = np.tanh(points[:,3])
#                 #genHistogram(points)
#                 Rr = np.random.choice(rain_rates)
#                 #print(f'Rain rate: {Rr}')
#                 #Simulate Rain
#                 noisy_pc = simulator.msu_rain(points[:, :4], Rr)  # augment_mc(points[:,:4], Rr)
#                 if noisy_pc.max() == np.nan or noisy_pc.max() > 200:
#                     rows, cols = np.where(noisy_pc > 200)
#                     print(f'num outliers: {rows.shape[0]}')
#                     for row, col in zip(rows, cols):
#                         print(f'Row:{row}, Col:{col}, Value:{noisy_pc[row,col]}')
#                         if col != 3:
#                             c=1
#                 #genHistogram(noisy_pc, Rr)
#                 #b=1
#
#                 if False:
#                     # Add labels 0=lost, 1=scattered, 2=attenuated #(N, 7): [x, y, z, intensity, elongation, NLZ_flag, labels]
#                     save_points = np.concatenate((noisy_pc[:,:4], points[:,4:6], noisy_pc[:, 4].reshape(-1,1)), axis=1).astype(float32)
#                     np.save(Path(file), save_points)
#
#                     # # Check points
#                     # noisy_points = np.load(file)
#                     # norm = np.linalg.norm(save_points[:, :4] - noisy_points[:, :4], axis=1)
#                     # c=1
#
#                 if False:
#                     lost_points = np.where(noisy_pc[:, 4] == 0)
#                     scattered_points = np.where(noisy_pc[:, 4] == 1)
#                     attenuated_points = np.where(noisy_pc[:, 4] == 2)
#
#                     noisy_pc[lost_points, :4] = points[lost_points, :4]
#                     V.draw_scenes(points=noisy_pc[:, :3], point_colors=get_colors(noisy_pc, color_feature=4))
#                     b=1
#
#                 info['point_cloud']['lidar_sequence'] = rain_seq_name
#                 info['frame_id'] = rain_seq_name + info['frame_id'][-4:] # seq_name + frame id (_000 to _198)
#                 info['precipitation_rate'] = Rr
#                 info['annos']['num_points_in_gt_clear'] = copy.deepcopy(info['annos']['num_points_in_gt'])
#
#
#                 gt_boxes = info['annos']['gt_boxes_lidar']
#                 num_obj = gt_boxes.shape[0]
#
#                 box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
#                     torch.from_numpy(noisy_pc[:, 0:3]).unsqueeze(dim=0).float().cuda(),
#                     torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
#                 ).long().squeeze(dim=0).cpu().numpy()
#
#                 for i in range(num_obj):
#                     gt_points = noisy_pc[box_idxs_of_pts == i]
#                     # if gt_points.shape[0] > 0:
#                     #     print(gt_points[:, :3].max())
#                     # gt_points[:, :3] -= gt_boxes[i, :3]
#                     # if gt_points.shape[0] > 0:
#                     #     print(gt_points[:, :3].max())
#
#                     info['annos']['num_points_in_gt'][i] = gt_points.shape[0]
#
#             #print(f'-------Simulated {rain_seq_name}--------')
#             # with open(rainy_seq_info_path, 'wb') as f:
#             #     pickle.dump(seq_infos, f)
#
#             # Remove old seq_info
#             #seq_info_path.unlink()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    parser.add_argument('--use_parallel', action='store_true', default=False, help='')
    args = parser.parse_args()
    ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
    import yaml
    from easydict import EasyDict
    try:
        yaml_config = yaml.safe_load(open(ROOT_DIR / args.cfg_file), Loader=yaml.FullLoader)
    except:
        yaml_config = yaml.safe_load(open(ROOT_DIR / args.cfg_file))
    dataset_cfg = EasyDict(yaml_config)


    if args.func == 'create_waymo_infos':
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )
    elif args.func == 'create_waymo_seg_labels':
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            only_extract_seg_labels=True
        )
    elif args.func == 'create_waymo_gt_database':
        create_waymo_gt_database(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=ROOT_DIR / 'data' / 'waymo',
            save_path=ROOT_DIR / 'data' / 'waymo',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG,
            use_parallel=args.use_parallel
        )
    else:
        raise NotImplementedError
    
    # if args.func == 'simulate_rain':
    #     simulate_rain(
    #         clear_weather_split = ['train_clear'],
    #         sim_percent= [0.5],
    #         sim_data_tag = 'sim_rain',
    #         data_path=Path('/media/barza/WD_BLACK/datasets/waymo'),
    #         processed_data_tag='waymo_processed_data_10')

