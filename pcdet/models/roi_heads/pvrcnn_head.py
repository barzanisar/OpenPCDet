import torch.nn as nn
import torch

from ...ops.pointnet2.pointnet2_stack import pointnet2_modules as pointnet2_stack_modules
from ...utils import common_utils, transform_utils
from .roi_head_template import RoIHeadTemplate
import kornia
import torch.nn.functional as F


class PVRCNNHead(RoIHeadTemplate):
    def __init__(self, input_channels, model_cfg, num_class=1):
        super().__init__(num_class=num_class, model_cfg=model_cfg)
        self.model_cfg = model_cfg

        mlps = self.model_cfg.ROI_GRID_POOL.MLPS
        for k in range(len(mlps)):
            mlps[k] = [input_channels] + mlps[k]

        self.roi_grid_pool_layer = pointnet2_stack_modules.StackSAModuleMSG(
            radii=self.model_cfg.ROI_GRID_POOL.POOL_RADIUS,
            nsamples=self.model_cfg.ROI_GRID_POOL.NSAMPLE,
            mlps=mlps,
            use_xyz=True,
            pool_method=self.model_cfg.ROI_GRID_POOL.POOL_METHOD,
        )

        GRID_SIZE = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        c_out = sum([x[-1] for x in mlps])
        pre_channel = GRID_SIZE * GRID_SIZE * GRID_SIZE * c_out

        shared_fc_list = []
        for k in range(0, self.model_cfg.SHARED_FC.__len__()):
            shared_fc_list.extend([
                nn.Conv1d(pre_channel, self.model_cfg.SHARED_FC[k], kernel_size=1, bias=False),
                nn.BatchNorm1d(self.model_cfg.SHARED_FC[k]),
                nn.ReLU()
            ])
            pre_channel = self.model_cfg.SHARED_FC[k]

            if k != self.model_cfg.SHARED_FC.__len__() - 1 and self.model_cfg.DP_RATIO > 0:
                shared_fc_list.append(nn.Dropout(self.model_cfg.DP_RATIO))

        self.shared_fc_layer = nn.Sequential(*shared_fc_list)

        self.cls_layers = self.make_fc_layers(
            input_channels=pre_channel, output_channels=self.num_class, fc_list=self.model_cfg.CLS_FC
        )
        self.reg_layers = self.make_fc_layers(
            input_channels=pre_channel,
            output_channels=self.box_coder.code_size * self.num_class,
            fc_list=self.model_cfg.REG_FC
        )
        self.init_weights(weight_init='xavier')

    def init_weights(self, weight_init='xavier'):
        if weight_init == 'kaiming':
            init_func = nn.init.kaiming_normal_
        elif weight_init == 'xavier':
            init_func = nn.init.xavier_normal_
        elif weight_init == 'normal':
            init_func = nn.init.normal_
        else:
            raise NotImplementedError

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                if weight_init == 'normal':
                    init_func(m.weight, mean=0, std=0.001)
                else:
                    init_func(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        nn.init.normal_(self.reg_layers[-1].weight, mean=0, std=0.001)

    def roi_grid_pool(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size:
                rois: (B, num_rois, 7 + C)
                point_coords: (num_points, 4)  [bs_idx, x, y, z]
                point_features: (num_points, C)
                point_cls_scores: (N1 + N2 + N3 + ..., 1)
                point_part_offset: (N1 + N2 + N3 + ..., 3)
        Returns:

        """
        batch_size = batch_dict['batch_size']
        rois = batch_dict['rois']
        point_coords = batch_dict['point_coords']
        point_features = batch_dict['point_features']

        # weighting based on foreground segmentation from image stream
        if self.model_cfg.IMAGE_BASED_KPW:
            keypoints_img_weights = self.get_kp_image_cls_scores(batch_dict)
            point_features = point_features * keypoints_img_weights.view(-1, 1)
        else:
            point_features = point_features * batch_dict['point_cls_scores'].view(-1, 1)

        global_roi_grid_points, local_roi_grid_points = self.get_global_grid_points_of_roi(
            rois, grid_size=self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        )  # (BxN, 6x6x6, 3)
        global_roi_grid_points = global_roi_grid_points.view(batch_size, -1, 3)  # (B, Nx6x6x6, 3)

        xyz = point_coords[:, 1:4]
        xyz_batch_cnt = xyz.new_zeros(batch_size).int()
        batch_idx = point_coords[:, 0]
        for k in range(batch_size):
            xyz_batch_cnt[k] = (batch_idx == k).sum()

        new_xyz = global_roi_grid_points.view(-1, 3)
        new_xyz_batch_cnt = xyz.new_zeros(batch_size).int().fill_(global_roi_grid_points.shape[1])
        pooled_points, pooled_features = self.roi_grid_pool_layer(
            xyz=xyz.contiguous(),
            xyz_batch_cnt=xyz_batch_cnt,
            new_xyz=new_xyz,
            new_xyz_batch_cnt=new_xyz_batch_cnt,
            features=point_features.contiguous(),
        )  # (M1 + M2 ..., C)

        pooled_features = pooled_features.view(
            -1, self.model_cfg.ROI_GRID_POOL.GRID_SIZE ** 3,
            pooled_features.shape[-1]
        )  # (BxN, 6x6x6, C)
        return pooled_features

    def get_kp_image_cls_scores(self, batch_dict):
        B = batch_dict['trans_lidar_to_cam'].shape[0]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = batch_dict['point_coords'] # N
        ## This assumes number of keypoints per batch is fixed
        num_keypoints_per_sample = torch.sum(point_coords[..., 0].int() == 0).item()
        point_coords_kornia = torch.zeros((B, num_keypoints_per_sample, 3), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            points_in_batch_b = (point_coords[..., 0].int() == b)
            point_coords_kornia[b,...] = point_coords[points_in_batch_b, 1:]

        # # Undo augmentations
        if self.training:
            undo_global_transform = batch_dict['undo_global_transform']
            point_coords_kornia = point_coords_kornia @ undo_global_transform

        # Transform to camera frame
        points_camera_frame = kornia.transform_points(trans_01=C_V, points_1=point_coords_kornia)

        # Project to image to get keypoint projection on image plane
        I_C = I_C.reshape(B, 3, 4)
        keypoints_img, keypoints_depths = transform_utils.project_to_image(project=I_C, points=points_camera_frame)

        # Get foreground weighting mask
        segment_logits = batch_dict['segment_logits']
        foreground_channel = 1
        foreground_probs = F.softmax(segment_logits, dim=1)[:, foreground_channel, :, :]
        # Flip foreground mask if image is flipped for correct kp projection
        if self.training:
            if 'image_flip' in batch_dict and batch_dict['image_flip'] == 1:
                foreground_probs = torch.flip(foreground_probs, [2])

        # Extract foregound weights for keypoints
        # Downsample coordinates of keypoint pixel coordinates to match reduced image dimensions
        keypoints_img = keypoints_img / 4

        # get weighting for each point
        # TODO should interpolate instead of casting to int
        keypoints_img = keypoints_img.long()
        keypoints_img_weights = torch.zeros((B, num_keypoints_per_sample), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            u = keypoints_img[b, :, 0]
            v = keypoints_img[b, :, 1]
            keypoints_img_weights[b, ...] = foreground_probs[b, v, u]

        #### FOR DEBUGGING
        VISUALIZE_MASK = False
        if VISUALIZE_MASK:
            import matplotlib.pyplot as plt
            import numpy as np
            image_orig = batch_dict['images'].cpu()
            image_reduced = F.interpolate(image_orig, size=[foreground_probs.size(1), foreground_probs.size(2)], mode="bilinear")
            image_gray = (0.2989 * image_reduced[:, 0, :, :] + 0.5870 * image_reduced[:, 1, :, :] + 0.1140 * image_reduced[:, 2, :, :]).squeeze()
            seg_mask = foreground_probs.cpu().detach().numpy().squeeze()
            kp_image = np.zeros(seg_mask.shape).squeeze()
            normalized_kp_depth = (keypoints_depths / torch.max(keypoints_depths)).cpu().detach().numpy()
            kp_image[keypoints_img.cpu().squeeze()[:, 1].long(), keypoints_img.cpu().squeeze()[:, 0].long()] = normalized_kp_depth
            empty = np.zeros(seg_mask.shape)
            draw_vector = np.stack([kp_image, seg_mask, image_gray])
            plt.imshow(np.moveaxis(draw_vector, 0, -1))
            plt.show()

        VISUALIZE_FLIP = False
        if VISUALIZE_FLIP:
            if 'image_flip' in batch_dict and batch_dict['image_flip'] == 1:
                import matplotlib.pyplot as plt
                import numpy as np
                image_orig = batch_dict['images'].permute(0, 2, 3, 1).cpu().squeeze()
                image_flipped = torch.fliplr(batch_dict['images'].permute(0, 2, 3, 1).squeeze()).cpu()
                plt.imshow(np.concatenate([image_orig, image_flipped]))
                plt.show()

        return keypoints_img_weights

    def get_global_grid_points_of_roi(self, rois, grid_size):
        rois = rois.view(-1, rois.shape[-1])
        batch_size_rcnn = rois.shape[0]

        local_roi_grid_points = self.get_dense_grid_points(rois, batch_size_rcnn, grid_size)  # (B, 6x6x6, 3)
        global_roi_grid_points = common_utils.rotate_points_along_z(
            local_roi_grid_points.clone(), rois[:, 6]
        ).squeeze(dim=1)
        global_center = rois[:, 0:3].clone()
        global_roi_grid_points += global_center.unsqueeze(dim=1)
        return global_roi_grid_points, local_roi_grid_points

    @staticmethod
    def get_dense_grid_points(rois, batch_size_rcnn, grid_size):
        faked_features = rois.new_ones((grid_size, grid_size, grid_size))
        dense_idx = faked_features.nonzero()  # (N, 3) [x_idx, y_idx, z_idx]
        dense_idx = dense_idx.repeat(batch_size_rcnn, 1, 1).float()  # (B, 6x6x6, 3)

        local_roi_size = rois.view(batch_size_rcnn, -1)[:, 3:6]
        roi_grid_points = (dense_idx + 0.5) / grid_size * local_roi_size.unsqueeze(dim=1) \
                          - (local_roi_size.unsqueeze(dim=1) / 2)  # (B, 6x6x6, 3)
        return roi_grid_points

    def forward(self, batch_dict):
        """
        :param input_data: input dict
        :return:
        """

        targets_dict = self.proposal_layer(
            batch_dict, nms_config=self.model_cfg.NMS_CONFIG['TRAIN' if self.training else 'TEST']
        )
        if self.training:
            targets_dict = self.assign_targets(batch_dict)
            batch_dict['rois'] = targets_dict['rois']
            batch_dict['roi_labels'] = targets_dict['roi_labels']

        # RoI aware pooling
        pooled_features = self.roi_grid_pool(batch_dict)  # (BxN, 6x6x6, C)

        grid_size = self.model_cfg.ROI_GRID_POOL.GRID_SIZE
        batch_size_rcnn = pooled_features.shape[0]
        pooled_features = pooled_features.permute(0, 2, 1).\
            contiguous().view(batch_size_rcnn, -1, grid_size, grid_size, grid_size)  # (BxN, C, 6, 6, 6)

        shared_features = self.shared_fc_layer(pooled_features.view(batch_size_rcnn, -1, 1))
        rcnn_cls = self.cls_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, 1 or 2)
        rcnn_reg = self.reg_layers(shared_features).transpose(1, 2).contiguous().squeeze(dim=1)  # (B, C)

        if not self.training:
            batch_cls_preds, batch_box_preds = self.generate_predicted_boxes(
                batch_size=batch_dict['batch_size'], rois=batch_dict['rois'], cls_preds=rcnn_cls, box_preds=rcnn_reg
            )
            batch_dict['batch_cls_preds'] = batch_cls_preds
            batch_dict['batch_box_preds'] = batch_box_preds
            batch_dict['cls_preds_normalized'] = False
        else:
            targets_dict['rcnn_cls'] = rcnn_cls
            targets_dict['rcnn_reg'] = rcnn_reg

            self.forward_ret_dict = targets_dict

        return batch_dict
