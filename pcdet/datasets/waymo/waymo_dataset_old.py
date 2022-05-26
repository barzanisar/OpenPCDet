# OpenPCDet PyTorch Dataloader and Evaluation Tools for Waymo Open Dataset
# Reference https://github.com/open-mmlab/OpenPCDet
# Written by Shaoshuai Shi, Chaoxu Guo
# All Rights Reserved 2019-2020.

import os
import pickle
import copy
import shutil
import glob

import numpy as np
import torch
import multiprocessing
from tqdm import tqdm
from pathlib import Path
from ...ops.roiaware_pool3d import roiaware_pool3d_utils
from ...utils import box_utils, common_utils
from ..dataset import DatasetTemplate
from ..simulator.vis import open3d_vis_utils as V
from ..simulator.atmos_models import LISA
from ..simulator.simulate_rain_waymo import *
#from ....scripts.simulate import simulate_rain_waymo


class WaymoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )
        self.data_path = self.root_path / self.dataset_cfg.PROCESSED_DATA_TAG
        self.split = self.dataset_cfg.DATA_SPLIT[self.mode]
        split_dir = self.root_path / 'ImageSets' / (self.split + '.txt')
        self.sample_sequence_list = [x.strip() for x in open(split_dir).readlines()]

        self.infos = []
        self.include_waymo_data(self.mode)

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

    def include_waymo_data(self, mode):
        self.logger.info('Loading Waymo dataset')
        waymo_infos = []

        num_skipped_infos = 0
        for k in range(len(self.sample_sequence_list)):
            sequence_name = os.path.splitext(self.sample_sequence_list[k])[0]
            info_path = self.data_path / sequence_name / ('%s.pkl' % sequence_name)
            info_path = self.check_sequence_name_with_all_version(info_path)
            if not info_path.exists():
                num_skipped_infos += 1
                continue
            with open(info_path, 'rb') as f:
                infos = pickle.load(f)
                waymo_infos.extend(infos)

        self.infos.extend(waymo_infos[:])
        self.logger.info('Total skipped info %s' % num_skipped_infos)
        self.logger.info('Total samples for Waymo dataset: %d' % (len(waymo_infos)))

        if self.dataset_cfg.SAMPLED_INTERVAL[mode] > 1:
            sampled_waymo_infos = []
            for k in range(0, len(self.infos), self.dataset_cfg.SAMPLED_INTERVAL[mode]):
                sampled_waymo_infos.append(self.infos[k])
            self.infos = sampled_waymo_infos
            self.logger.info('Total sampled samples for Waymo dataset: %d' % len(self.infos))

    @staticmethod
    def check_sequence_name_with_all_version(sequence_file):
        if 'sim_rain' in str(sequence_file):
            return sequence_file
        if '_with_camera_labels' not in str(sequence_file) and not sequence_file.exists():
            if '.pkl' not in str(sequence_file):
                b = 1
            sequence_file = Path(str(sequence_file)[:-4] + '_with_camera_labels.tfrecord')
        if '_with_camera_labels' in str(sequence_file) and not sequence_file.exists():
            sequence_file = Path(str(sequence_file).replace('_with_camera_labels', ''))

        return sequence_file

    def get_infos(self, raw_data_path, save_path, num_workers=multiprocessing.cpu_count(), has_label=True, sampled_interval=1):
        import concurrent.futures as futures
        from functools import partial
        from . import waymo_utils
        print('---------------The waymo sample interval is %d, total sequecnes is %d-----------------'
              % (sampled_interval, len(self.sample_sequence_list)))

        process_single_sequence = partial(
            waymo_utils.process_single_sequence,
            save_path=save_path, sampled_interval=sampled_interval, has_label=has_label
        )
        sample_sequence_file_list = [
            self.check_sequence_name_with_all_version(raw_data_path / sequence_file)
            for sequence_file in self.sample_sequence_list
        ]

        # process_single_sequence(sample_sequence_file_list[0])
        with futures.ThreadPoolExecutor(num_workers) as executor:
            sequence_infos = list(tqdm(executor.map(process_single_sequence, sample_sequence_file_list),
                                       total=len(sample_sequence_file_list)))
        all_sequences_infos = [item for infos in sequence_infos for item in infos]
        return all_sequences_infos

    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

        points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
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
        points = self.get_lidar(sequence_name, sample_idx)

        input_dict = {
            'points': points,
            'frame_id': info['frame_id'],
        }

        if 'annos' in info:
            annos = info['annos']
            annos = common_utils.drop_info_with_name(annos, name='unknown')

            if self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False):
                gt_boxes_lidar = box_utils.boxes3d_kitti_fakelidar_to_lidar(annos['gt_boxes_lidar'])
            else:
                gt_boxes_lidar = annos['gt_boxes_lidar']

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
            kitti_utils.transform_annotations_to_kitti_format(eval_det_annos, map_name_to_kitti=map_name_to_kitti)
            kitti_utils.transform_annotations_to_kitti_format(
                eval_gt_annos, map_name_to_kitti=map_name_to_kitti,
                info_with_fakelidar=self.dataset_cfg.get('INFO_WITH_FAKELIDAR', False)
            )
            kitti_class_names = [map_name_to_kitti[x] for x in class_names]
            ap_result_str, ap_dict = kitti_eval.get_official_eval_result(
                gt_annos=eval_gt_annos, dt_annos=eval_det_annos, current_classes=kitti_class_names
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
            ap_result_str, ap_dict = waymo_eval(eval_det_annos, eval_gt_annos, kwargs.get('eval_levels_list_cfg',None))
        else:
            raise NotImplementedError

        return ap_result_str, ap_dict
        
    def create_groundtruth_database(self, info_path, save_path, used_classes=None, split='train', sampled_interval=10,
                                    processed_data_tag=None):
        database_save_path = save_path / ('pcdet_gt_database_%s_sampled_%d' % (split, sampled_interval))
        db_info_save_path = save_path / ('pcdet_waymo_dbinfos_%s_sampled_%d.pkl' % (split, sampled_interval))

        database_save_path.mkdir(parents=True, exist_ok=True)
        all_db_infos = {}

        with open(info_path, 'rb') as f:
            infos = pickle.load(f)

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

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()


            # seq_dict_path = Path(save_path / processed_data_tag / sequence_name / (sequence_name + '.pkl'))
            # if not seq_dict_path.exists():
            #     original_clear_weather_path = Path(save_path / "waymo_processed_data" / sequence_name / (sequence_name + '.pkl'))
            #     shutil.copyfile(original_clear_weather_path, seq_dict_path)
            # with open(seq_dict_path, 'rb') as f:
            #     seq_dict = pickle.load(f)

            for i in range(num_obj):
                filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
                filepath = database_save_path / filename
                gt_points = points[box_idxs_of_pts == i]
                gt_points[:, :3] -= gt_boxes[i, :3]

                if (used_classes is None) or names[i] in used_classes:
                    with open(filepath, 'w') as f:
                        gt_points.tofile(f)
                    with open(filepath, 'r') as f:
                        obj_points = np.fromfile(str(filepath), dtype=np.float32).reshape(
                            [-1, 5])
                        if obj_points.shape[0] > 0 and obj_points.max() == np.nan:
                            b=1

                    db_path = str(filepath.relative_to(self.root_path))  # gt_database/xxxxx.bin
                    db_info = {'name': names[i], 'path': db_path, 'sequence_name': sequence_name,
                               'sample_idx': sample_idx, 'gt_idx': i, 'box3d_lidar': gt_boxes[i],
                               'num_points_in_gt': gt_points.shape[0], 'difficulty': difficulty[i]}
                    # infos[k]['annos']['num_points_in_gt'][i] = gt_points.shape[0]
                    # seq_dict[sample_idx]['annos']['num_points_in_gt'][i] = gt_points.shape[0]
                    if names[i] in all_db_infos:
                        all_db_infos[names[i]].append(db_info)
                    else:
                        all_db_infos[names[i]] = [db_info]
            
            # # Update num_points_in_gt in processed seq pkl
            # with open(seq_dict_path, 'wb') as f:
            #     pickle.dump(seq_dict, f)
        for k, v in all_db_infos.items():
            print('Database %s: %d' % (k, len(v)))

        with open(db_info_save_path, 'wb') as f:
            pickle.dump(all_db_infos, f)
        
        # # Update num_points_in_gt in waymo_infos_train.pkl
        # with open(info_path, 'wb') as f:
        #     pickle.dump(infos, f)
        


def create_waymo_infos(dataset_cfg, class_names, data_path, save_path,
                       raw_data_tag='raw_data', processed_data_tag='waymo_processed_data',
                       workers=multiprocessing.cpu_count()):
    dataset = WaymoDataset(
        dataset_cfg=dataset_cfg, class_names=class_names, root_path=data_path,
        training=False, logger=common_utils.create_logger()
    )
    train_split, val_split = dataset_cfg.DATA_SPLIT['train'], dataset_cfg.DATA_SPLIT['test']

    train_filename = save_path / ('waymo_infos_%s.pkl' % train_split)
    val_filename = save_path / ('waymo_infos_%s.pkl' % val_split)

    print('---------------Start to generate data infos---------------')

    dataset.set_split(train_split)
    waymo_infos_train = dataset.get_infos(
        raw_data_path=data_path / raw_data_tag,
        save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
        sampled_interval=1
    )
    with open(train_filename, 'wb') as f:
        pickle.dump(waymo_infos_train, f)
    print('----------------Waymo info train file is saved to %s----------------' % train_filename)
    #
    # dataset.set_split(val_split)
    # waymo_infos_val = dataset.get_infos(
    #     raw_data_path=data_path / raw_data_tag,
    #     save_path=save_path / processed_data_tag, num_workers=workers, has_label=True,
    #     sampled_interval=1
    # )
    # with open(val_filename, 'wb') as f:
    #     pickle.dump(waymo_infos_val, f)
    # print('----------------Waymo info val file is saved to %s----------------' % val_filename)

    print('---------------Start create groundtruth database for data augmentation---------------')
    dataset.set_split(train_split)
    dataset.create_groundtruth_database(
        info_path=train_filename, save_path=save_path, split=train_split, sampled_interval=10,
        used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    )

    # dataset.set_split(val_split)
    # dataset.create_groundtruth_database(
    #     info_path=val_filename, save_path=save_path, split=val_split, sampled_interval=10,
    #     used_classes=['Vehicle', 'Pedestrian', 'Cyclist'], processed_data_tag=processed_data_tag
    # )
    # print('---------------Data preparation Done---------------')

def simulate_rain(clear_weather_split, sim_percent, sim_data_tag, data_path, processed_data_tag):
    simulator = LISA(rmax=200)
    rain_rates = np.linspace(0.5, 10.0, 20)
    
    data_save_path = data_path / 'rainy_processed_data'
    data_save_path.mkdir(parents=True, exist_ok=True)
    clear_data_path = data_path / processed_data_tag

    for split, percent in zip(clear_weather_split, sim_percent):
        split_dir = data_path / 'ImageSets' / (split + '.txt')
        clear_sequence_list = [x.strip() for x in open(split_dir).readlines()]
        sim_upto_len = int(percent * len(clear_sequence_list))
        #rainy_split = split + '_' + sim_data_tag

        # # Create train_rainy.txt (50% rainy and 50% clear) and val_rainy.txt
        # dst_txt = data_path / 'ImageSets' / (rainy_split + '.txt')
        # with open(dst_txt, 'w') as f:
        #     for i in range(sim_upto_len):
        #         seq_name = clear_sequence_list[i]
        #         # create a rainy name for this seq
        #         if '_with_camera_labels' in seq_name:
        #             rain_seq_name = seq_name.split('_with_camera_labels')[
        #                                 0] + '_' + sim_data_tag + '_with_camera_labels' + \
        #                             seq_name.split('_with_camera_labels')[1]
        #         else:
        #             rain_seq_name = seq_name.split('.')[0] + '_' + sim_data_tag + seq_name.split('.')[1]
        #
        #         f.write(rain_seq_name)
        #         if i != len(clear_sequence_list) - 1:
        #             f.write('\n')
        #     for i in range(sim_upto_len, len(clear_sequence_list)):
        #         f.write(clear_sequence_list[i])
        #         if i != len(clear_sequence_list) - 1:
        #             f.write('\n')
                    
        
        for i in tqdm(range(sim_upto_len)): #, desc='Number of sequences processed'
            seq_name = clear_sequence_list[i].split('.')[0]
            
            # create a rainy name for this seq
            if '_with_camera_labels' in seq_name:
                rain_seq_name = seq_name.split('_with_camera_labels')[0] + '_' + sim_data_tag + '_with_camera_labels'
            else:
                rain_seq_name = seq_name.split('.')[0] + '_' + sim_data_tag

            clear_seq_dir = clear_data_path / seq_name
            rainy_seq_dir = data_save_path / rain_seq_name

            # Copy clear processed seq into a new rainy seq dir
            shutil.copytree(clear_seq_dir, rainy_seq_dir, dirs_exist_ok=True)
            seq_info_path = rainy_seq_dir / (seq_name + '.pkl')
            rainy_seq_info_path = rainy_seq_dir / (rain_seq_name + '.pkl')
            with open(seq_info_path, 'rb') as f:
                seq_infos = pickle.load(f)
            
            lidar_files = sorted(glob.glob(str(rainy_seq_dir) + "/*.npy"))
            for file, info in tqdm(zip(lidar_files, seq_infos)): #, desc=f'Number of frames in one seq: {rain_seq_name}'
                points = np.load(file)  # (N, 6): [x, y, z, intensity, elongation, NLZ_flag]
                points[:,3] = np.tanh(points[:,3])
                #genHistogram(points)
                Rr = np.random.choice(rain_rates)
                #print(f'Rain rate: {Rr}')
                #Simulate Rain
                noisy_pc = simulator.msu_rain(points[:, :4], Rr)  # augment_mc(points[:,:4], Rr)
                if noisy_pc.max() == np.nan or noisy_pc.max() > 200:
                    rows, cols = np.where(noisy_pc > 200)
                    print(f'num outliers: {rows.shape[0]}')
                    for row, col in zip(rows, cols):
                        print(f'Row:{row}, Col:{col}, Value:{noisy_pc[row,col]}')
                        if col != 3:
                            c=1
                #genHistogram(noisy_pc, Rr)
                #b=1
                
                if False:
                    # Add labels 0=lost, 1=scattered, 2=attenuated #(N, 7): [x, y, z, intensity, elongation, NLZ_flag, labels]
                    save_points = np.concatenate((noisy_pc[:,:4], points[:,4:6], noisy_pc[:, 4].reshape(-1,1)), axis=1).astype(float32)
                    np.save(Path(file), save_points)

                    # # Check points
                    # noisy_points = np.load(file)
                    # norm = np.linalg.norm(save_points[:, :4] - noisy_points[:, :4], axis=1)
                    # c=1

                if False:
                    lost_points = np.where(noisy_pc[:, 4] == 0)
                    scattered_points = np.where(noisy_pc[:, 4] == 1)
                    attenuated_points = np.where(noisy_pc[:, 4] == 2)
                    
                    noisy_pc[lost_points, :4] = points[lost_points, :4]
                    V.draw_scenes(points=noisy_pc[:, :3], point_colors=get_colors(noisy_pc, color_feature=4))
                    b=1

                info['point_cloud']['lidar_sequence'] = rain_seq_name
                info['frame_id'] = rain_seq_name + info['frame_id'][-4:] # seq_name + frame id (_000 to _198)
                info['precipitation_rate'] = Rr
                info['annos']['num_points_in_gt_clear'] = copy.deepcopy(info['annos']['num_points_in_gt'])


                gt_boxes = info['annos']['gt_boxes_lidar']
                num_obj = gt_boxes.shape[0]

                box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                    torch.from_numpy(noisy_pc[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                    torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
                ).long().squeeze(dim=0).cpu().numpy()

                for i in range(num_obj):
                    gt_points = noisy_pc[box_idxs_of_pts == i]
                    # if gt_points.shape[0] > 0:
                    #     print(gt_points[:, :3].max())
                    # gt_points[:, :3] -= gt_boxes[i, :3]
                    # if gt_points.shape[0] > 0:
                    #     print(gt_points[:, :3].max())

                    info['annos']['num_points_in_gt'][i] = gt_points.shape[0]

                b=1

            #print(f'-------Simulated {rain_seq_name}--------')
            # with open(rainy_seq_info_path, 'wb') as f:
            #     pickle.dump(seq_infos, f)

            # Remove old seq_info
            #seq_info_path.unlink()



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config of dataset')
    parser.add_argument('--func', type=str, default='create_waymo_infos', help='')
    args = parser.parse_args()

    if args.func == 'create_waymo_infos':
        import yaml
        from easydict import EasyDict
        dataset_cfg = EasyDict(yaml.load(open(args.cfg_file)))
        ROOT_DIR = (Path(__file__).resolve().parent / '../../../').resolve()
        create_waymo_infos(
            dataset_cfg=dataset_cfg,
            class_names=['Vehicle', 'Pedestrian', 'Cyclist'],
            data_path=Path('/media/barza/WD_BLACK/datasets/waymo'),
            save_path=Path('/media/barza/WD_BLACK/datasets/waymo'),
            raw_data_tag='raw_data',
            processed_data_tag=dataset_cfg.PROCESSED_DATA_TAG
        )
    
    if args.func == 'simulate_rain':
        simulate_rain(
            clear_weather_split = ['train_clear'],
            sim_percent= [0.5],
            sim_data_tag = 'sim_rain',
            data_path=Path('/media/barza/WD_BLACK/datasets/waymo'),
            processed_data_tag='waymo_processed_data_10')

