import pickle

import numpy as np
from torch.utils import data

from ...ops.iou3d_nms import iou3d_nms_utils
from ...utils import box_utils


class DataBaseSampler(object):
    def __init__(self, root_path, sampler_cfg, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.sampler_cfg = sampler_cfg
        self.logger = logger
        self.db_infos = {}
        for class_name in class_names:
            self.db_infos[class_name] = []

        for db_info_path in sampler_cfg.DB_INFO_PATH:
            db_info_path = self.root_path.resolve() / db_info_path
            with open(str(db_info_path), 'rb') as f:
                infos = pickle.load(f)
                [self.db_infos[cur_class].extend(infos[cur_class]) for cur_class in class_names]

        for func_name, val in sampler_cfg.PREPARE.items():
            self.db_infos = getattr(self, func_name)(self.db_infos, val)

        self.sample_groups = {}
        self.sample_class_num = {}
        self.limit_whole_scene = sampler_cfg.get('LIMIT_WHOLE_SCENE', False)
        for x in sampler_cfg.SAMPLE_GROUPS:
            class_name, sample_num = x.split(':')
            if class_name not in class_names:
                continue
            self.sample_class_num[class_name] = sample_num
            self.sample_groups[class_name] = {
                'sample_num': sample_num,
                'pointer': len(self.db_infos[class_name]),
                'indices': np.arange(len(self.db_infos[class_name]))
            }

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)

    def filter_by_difficulty(self, db_infos, removed_difficulty):
        new_db_infos = {}
        for key, dinfos in db_infos.items():
            pre_len = len(dinfos)
            new_db_infos[key] = [
                info for info in dinfos
                if info['difficulty'] not in removed_difficulty
            ]
            if self.logger is not None:
                self.logger.info('Database filter by difficulty %s: %d => %d' % (key, pre_len, len(new_db_infos[key])))
        return new_db_infos

    def filter_by_min_points(self, db_infos, min_gt_points_list):
        for name_num in min_gt_points_list:
            name, min_num = name_num.split(':')
            min_num = int(min_num)
            if min_num > 0 and name in db_infos.keys():
                filtered_infos = []
                for info in db_infos[name]:
                    if info['num_points_in_gt'] >= min_num:
                        filtered_infos.append(info)

                if self.logger is not None:
                    self.logger.info('Database filter by min points %s: %d => %d' %
                                     (name, len(db_infos[name]), len(filtered_infos)))
                db_infos[name] = filtered_infos

        return db_infos

    def sample_with_fixed_number(self, class_name, sample_group):
        """
        Args:
            class_name:
            sample_group:
        Returns:

        """
        sample_num, pointer, indices = int(sample_group['sample_num']), sample_group['pointer'], sample_group['indices']
        if pointer >= len(self.db_infos[class_name]):
            indices = np.random.permutation(len(self.db_infos[class_name]))
            pointer = 0

        sampled_dict = [self.db_infos[class_name][idx] for idx in indices[pointer: pointer + sample_num]]
        pointer += sample_num
        sample_group['pointer'] = pointer
        sample_group['indices'] = indices
        return sampled_dict

    @staticmethod
    def put_boxes_on_road_planes(gt_boxes, road_planes, calib):
        """
        Only validate in KITTIDataset
        Args:
            gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]
            road_planes: [a, b, c, d]
            calib:

        Returns:
        """
        a, b, c, d = road_planes
        center_cam = calib.lidar_to_rect(gt_boxes[:, 0:3])
        cur_height_cam = (-d - a * center_cam[:, 0] - c * center_cam[:, 2]) / b
        center_cam[:, 1] = cur_height_cam
        cur_lidar_height = calib.rect_to_lidar(center_cam)[:, 2]
        mv_height = gt_boxes[:, 2] - gt_boxes[:, 5] / 2 - cur_lidar_height
        gt_boxes[:, 2] -= mv_height  # lidar view
        return gt_boxes, mv_height

    def add_sampled_boxes_to_scene(self, data_dict, sampled_gt_boxes, total_valid_sampled_dict):
        gt_boxes_mask = data_dict['gt_boxes_mask']
        gt_boxes = data_dict['gt_boxes'][gt_boxes_mask]
        gt_names = data_dict['gt_names'][gt_boxes_mask]
        points = data_dict['points']
        if self.sampler_cfg.get('USE_ROAD_PLANE', False):
            sampled_gt_boxes, mv_height = self.put_boxes_on_road_planes(
                sampled_gt_boxes, data_dict['road_plane'], data_dict['calib']
            )
            data_dict.pop('calib')
            data_dict.pop('road_plane')

        obj_points_list = []
        for idx, info in enumerate(total_valid_sampled_dict):
            file_path = self.root_path / info['path']
            obj_points = np.fromfile(str(file_path), dtype=np.float32).reshape(
                [-1, self.sampler_cfg.NUM_POINT_FEATURES])
            if points.shape[1] == self.sampler_cfg.NUM_POINT_FEATURES + 1:
                obj_points = np.concatenate([obj_points, np.zeros((obj_points.shape[0],1))], axis=1)

            obj_points[:, :3] += info['box3d_lidar'][:3]

            if self.sampler_cfg.get('USE_ROAD_PLANE', False):
                # mv height
                obj_points[:, 2] -= mv_height[idx]

            obj_points_list.append(obj_points)

        obj_points = np.concatenate(obj_points_list, axis=0)
        sampled_gt_names = np.array([x['name'] for x in total_valid_sampled_dict])

        large_sampled_gt_boxes = box_utils.enlarge_box3d(
            sampled_gt_boxes[:, 0:7], extra_width=self.sampler_cfg.REMOVE_EXTRA_WIDTH
        )
        points = box_utils.remove_points_in_boxes3d(points, large_sampled_gt_boxes)
        points = np.concatenate([obj_points, points], axis=0)
        gt_names = np.concatenate([gt_names, sampled_gt_names], axis=0)
        gt_boxes = np.concatenate([gt_boxes, sampled_gt_boxes], axis=0)
        data_dict['gt_boxes'] = gt_boxes
        data_dict['gt_names'] = gt_names
        data_dict['points'] = points
        return data_dict

    def __call__(self, data_dict):
        """
        Args:
            data_dict:
                gt_boxes: (N, 7 + C) [x, y, z, dx, dy, dz, heading, ...]

        Returns:

        """
        gt_boxes = data_dict['gt_boxes']
        gt_names = data_dict['gt_names'].astype(str)
        existed_boxes = gt_boxes
        total_valid_sampled_dict = []

        for class_name, sample_group in self.sample_groups.items():
            if self.limit_whole_scene:
                num_gt = np.sum(class_name == gt_names)
                sample_group['sample_num'] = str(int(self.sample_class_num[class_name]) - num_gt)
            if int(sample_group['sample_num']) > 0:
                sampled_dict = self.sample_with_fixed_number(class_name, sample_group)

                sampled_boxes = np.stack([x['box3d_lidar'] for x in sampled_dict], axis=0).astype(np.float32)
                sampled_cam_bboxes = np.stack([x['bbox'] for x in sampled_dict], axis=0).astype(np.float32)

                if self.sampler_cfg.get('DATABASE_WITH_FAKELIDAR', False):
                    sampled_boxes = box_utils.boxes3d_kitti_fakelidar_to_lidar(sampled_boxes)

                iou1 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], existed_boxes[:, 0:7])
                iou2 = iou3d_nms_utils.boxes_bev_iou_cpu(sampled_boxes[:, 0:7], sampled_boxes[:, 0:7])
                iou2[range(sampled_boxes.shape[0]), range(sampled_boxes.shape[0])] = 0
                iou1 = iou1 if iou1.shape[1] > 0 else iou2
                valid_mask = ((iou1.max(axis=1) + iou2.max(axis=1)) == 0).nonzero()[0]
                valid_sampled_dict = [sampled_dict[x] for x in valid_mask]
                valid_sampled_boxes = sampled_boxes[valid_mask]
                # Ground truth samples heatmap
                valid_sampled_cam_bboxes = sampled_cam_bboxes[valid_mask]
                if self.sampler_cfg.get('PROJECT_GT_SAMPLES', False):
                    self.populate_2d_detection_with_gt_sampled_boxes(data_dict,
                    valid_sampled_cam_bboxes_2d = valid_sampled_cam_bboxes, class_name=class_name)

                existed_boxes = np.concatenate((existed_boxes, valid_sampled_boxes), axis=0)
                total_valid_sampled_dict.extend(valid_sampled_dict)

        sampled_gt_boxes = existed_boxes[gt_boxes.shape[0]:, :]
        if total_valid_sampled_dict.__len__() > 0:
            data_dict = self.add_sampled_boxes_to_scene(data_dict, sampled_gt_boxes, total_valid_sampled_dict)

        data_dict.pop('gt_boxes_mask')
        return data_dict

    def populate_2d_detection_with_gt_sampled_boxes(self, data_dict, valid_sampled_cam_bboxes_2d, class_name):
        if 'gt_samples_2d_detections' in data_dict:
            detection_heat_map = data_dict['gt_samples_2d_detections']
        else:
            image_shape = data_dict['images'].shape[0:2]
            detection_heat_map = np.zeros((image_shape[0], image_shape[1], len(self.class_names)), dtype=np.float32)
            data_dict['gt_samples_2d_detections'] = detection_heat_map

        if class_name == 'Car' or class_name == 'VEHICLE':
            index = 0
        elif class_name == 'Pedestrian' or class_name == 'PEDESTRIAN':
            index = 1
        elif class_name == 'Cyclist' or class_name == 'CYCLIST':
            index = 2
        else:
            raise NotImplementedError
        MIN_PROB = self.sampler_cfg.get('MIN_BBOX_DETECTION_THRES', 1.0)
        MAX_PROB = self.sampler_cfg.get('MAX_BBOX_DETECTION_THRES', 1.0)
        MAX_BB_PIXEL_SHIFT = self.sampler_cfg.get('MAX_BB_PIXEL_SHIFT', 0)
        PROJECT_PERCENTAGE = self.sampler_cfg.get('PROJECT_PERCENTAGE', 100.0)/100.0 # defaults to adding all 2d bounding box of ground truth samples to detection_heat_map
        assert MAX_PROB >= MIN_PROB
        MAX_FRONTVIEW_INTERSECTION = self.sampler_cfg.get('MAX_FRONTVIEW_INTERSECTION', 1.0)
        if MAX_FRONTVIEW_INTERSECTION < 1.0:
            max_intersection_per_gt_sample = self.compute_gt_samples_intersection(data_dict, valid_sampled_cam_bboxes_2d)
        for idx, bbox in enumerate(valid_sampled_cam_bboxes_2d):
            if MAX_FRONTVIEW_INTERSECTION == 1.0 or max_intersection_per_gt_sample[idx] < MAX_FRONTVIEW_INTERSECTION:
                gt_sample_detection_confidence = np.random.uniform(MIN_PROB, MAX_PROB)
                gt_sample_project_on_image = np.random.uniform(0.0, 1.0)
                if MAX_BB_PIXEL_SHIFT > 0:
                    shift = np.random.randint(low=-MAX_BB_PIXEL_SHIFT, high=MAX_BB_PIXEL_SHIFT, size=(4,), dtype=np.int16)
                else:
                    shift = np.zeros((4,), dtype=np.int16)

                # Condition for projecting bounding box from point cloud ground truth sampling
                # Note: this  condition ensures that lidar stream doesnt rely fully on image stream weighting for gt samples
                if gt_sample_project_on_image <= PROJECT_PERCENTAGE:
                    v1 = int(bbox[1] + shift[0]) if int(bbox[1] + shift[0]) >= 0 else 0
                    v2 = int(bbox[3] + shift[1]) if int(bbox[3] + shift[1]) < detection_heat_map.shape[0] else detection_heat_map.shape[0]
                    u1 = int(bbox[0] + shift[2]) if int(bbox[0] + shift[2]) >= 0 else 0
                    u2 = int(bbox[2] + shift[3]) if int(bbox[2] + shift[3]) < detection_heat_map.shape[1] else detection_heat_map.shape[1]
                    detection_heat_map[v1:v2,
                                    u1:u2, index] = gt_sample_detection_confidence

    # compute 2d intersection between gt samples and gt_boxes2d
    # returns an one-dimensional array of maximum intersection of gt_boxes2d with gt samples
    def compute_gt_samples_intersection(self, data_dict, valid_sampled_cam_bboxes_2d):
        gt_boxes2d = data_dict['gt_boxes2d']
        gt_sampled2d = valid_sampled_cam_bboxes_2d
        # intersection value for each gt_sample with each gt 2d box
        intersection = np.zeros((gt_sampled2d.shape[0], gt_boxes2d.shape[0]))
        # boxes in the scene
        X1 = gt_boxes2d[:, 1]
        X2 = gt_boxes2d[:, 3] # X2 > X1
        Y1 = gt_boxes2d[:, 0]
        Y2 = gt_boxes2d[:, 2] # Y2 > Y1
        # gt boxes
        X3 = gt_sampled2d[:, 1]
        X4 = gt_sampled2d[:, 3] # X4 > X3
        Y3 = gt_sampled2d[:, 0]
        Y4 = gt_sampled2d[:, 2] # Y4 > Y3
        # loop over boxes in scene and compute intersection with each gt sampled box
        # adapted from https://stackoverflow.com/questions/19753134/get-the-points-of-intersection-from-2-rectangles
        for i in range(0, intersection.shape[1]):
            X5 = np.maximum(X3, X1[i])
            Y5 = np.maximum(Y3, Y1[i])
            X6 = np.minimum(X4, X2[i])
            Y6 = np.minimum(Y4, Y2[i])
            delta_X = X6-X5
            delta_Y = Y6-Y5
            # remove degenerate cases where objects/rectangles dont intersect
            delta_X[delta_X<0] = 0
            delta_Y[delta_Y<0] = 0
            # compute intersection
            size_of_2d_box = (X2[i] - X1[i]) * (Y2[i] - Y1[i])
            intersect = np.multiply(delta_X, delta_Y) / size_of_2d_box
            #assert intersect >= 0 and intersect <= 1
            intersection[:, i] = intersect

        # maximum intersection
        max_intersection_per_gt_sample = np.amax(intersection, axis=1)
        return max_intersection_per_gt_sample
