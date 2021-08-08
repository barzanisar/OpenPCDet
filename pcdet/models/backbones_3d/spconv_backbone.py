from functools import partial
from os import stat

import spconv
import torch.nn as nn
import torch
from ...utils import common_utils, transform_utils, loss_utils
import kornia
import torch.nn.functional as F


def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='subm', norm_fn=None):

    if conv_type == 'subm':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'spconv':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'inverseconv':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m


class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out.features = self.bn1(out.features)
        out.features = self.relu(out.features)

        out = self.conv2(out)
        out.features = self.bn2(out.features)

        if self.downsample is not None:
            identity = self.downsample(x)

        out.features += identity.features
        out.features = self.relu(out.features)

        return out


class VoxelBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelResBackBone8x(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, **kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
            SparseBasicBlock(16, 16, norm_fn=norm_fn, indice_key='res1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
            SparseBasicBlock(32, 32, norm_fn=norm_fn, indice_key='res2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
            SparseBasicBlock(64, 64, norm_fn=norm_fn, indice_key='res3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 128, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
            SparseBasicBlock(128, 128, norm_fn=norm_fn, indice_key='res4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(128, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )
        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 128
        }

    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })

        return batch_dict


class VoxelBackBone8xFuse(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(64, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 16,
            'x_conv2': 32,
            'x_conv3': 64,
            'x_conv4': 64
        }

        assert 'FUSE_LAYERS' in self.model_cfg and 'FUSE_MODE' in self.model_cfg
        # Fusion mode/layers - create mlp that learns channel-wise weight for each conv feature maps
        if 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv1_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv1']), nn.Sigmoid())
        if 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv2_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv2']), nn.Sigmoid())
        if 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv3_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv3']), nn.Sigmoid())
        if 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv4_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv4']), nn.Sigmoid())
    
    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        if self.model_cfg['FUSE_MODE'] == 'channel-fixed-weight':
            return (voxel_feature * image_foreground_weights.view(-1, 1)) + voxel_feature
        elif self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
            assert vox_conv_layer is not None
            if vox_conv_layer == 'x_conv1':
                learned_channel_layer = self.x_conv1_mlp
            elif vox_conv_layer == 'x_conv2':
                learned_channel_layer = self.x_conv2_mlp
            elif vox_conv_layer == 'x_conv3':
                learned_channel_layer = self.x_conv3_mlp
            elif vox_conv_layer == 'x_conv4':
                learned_channel_layer = self.x_conv4_mlp
            else:
                raise NotImplementedError
            learned_channel_weights = learned_channel_layer(image_foreground_weights.view(-1, 1))
            return  (voxel_feature * learned_channel_weights) + voxel_feature
        else:
            raise NotImplementedError

    
    # Ground truth training with grid sampler - no loss of precision
    def get_voxel_image_weights(self, batch_dict, conv_x_coords):
        B = batch_dict['trans_lidar_to_cam'].shape[0]
        image_h, image_w = batch_dict['images'].shape[-2:]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = conv_x_coords  # N
        ## This assumes number of keypoints per batch is fixed
        num_keypoints_per_sample = torch.sum(point_coords[..., 0].int() == 0).item()
        point_coords_kornia = torch.zeros((B, num_keypoints_per_sample, 3), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            points_in_batch_b = (point_coords[..., 0].int() == b)
            point_coords_kornia[b, ...] = point_coords[points_in_batch_b, 1:]

        # # Undo augmentations
        if 'undo_global_transform' in batch_dict:
            undo_global_transform = batch_dict['undo_global_transform']
            point_coords_kornia = point_coords_kornia @ undo_global_transform.inverse()


        # Transform to camera frame
        points_camera_frame = kornia.transform_points(trans_01=C_V, points_1=point_coords_kornia)

        # Project to image to get keypoint projection on image plane
        I_C = I_C.reshape(B, 3, 4)
        keypoints_img, keypoints_depths = transform_utils.project_to_image(project=I_C, points=points_camera_frame)

        if 'segment_logits' not in batch_dict: # a segmentation network is not used 
            # Get foreground weighting mask
            assert not('gt_boxes2d' in batch_dict and '2d_detections' in batch_dict) # only one source image can be sampled
            if 'gt_boxes2d' in batch_dict:
                gt_boxes2d = batch_dict['gt_boxes2d']
                image = batch_dict['images']
                mask_shape = (image.shape[0], image.shape[2] + 1, image.shape[3] + 1)
                foreground_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                                    shape=mask_shape,
                                                    downsample_factor=1,
                                                    device=keypoints_img.device)
                segmentation_targets = torch.zeros(foreground_mask.shape, dtype=torch.float32, device=foreground_mask.device)
                segmentation_targets[foreground_mask.long() == True] = 1.0
            elif '2d_detections' in batch_dict:  
                segmentation_targets = batch_dict['2d_detections']
        else:
            # use output of segmentation network for voxel feature weighting
            segment_logits = batch_dict['segment_logits']
            foreground_channel = 1
            segmentation_targets = F.softmax(segment_logits, dim=1)[:, foreground_channel, :, :]

        if 'image_flip' in batch_dict and batch_dict['image_flip'] == 1:
            segmentation_targets = torch.flip(segmentation_targets, [2])
        
        # in-place keypoint location conversion to normalized pixel coordinates [-1, 1] for grid sampler
        self.convert_to_normalized_range(image_range=(image_h, image_w), keypoint_pixel=keypoints_img)
        # sampler
        image_voxel_features = torch.nn.functional.grid_sample(input=segmentation_targets.view(B, 1, segmentation_targets.shape[1], segmentation_targets.shape[2]),
                                        grid=keypoints_img.view(B, num_keypoints_per_sample, 1, 2), mode='bilinear',
                                        padding_mode='zeros', align_corners=None)
        
        return image_voxel_features

    
    # Takes keypoints locations in pixel coordinates and converts them to normalized coordinates [-1, 1] for grid sampling
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    @staticmethod
    def convert_to_normalized_range(image_range, keypoint_pixel):
        B = keypoint_pixel.shape[0]
        old_h_img_range, old_w_img_range = image_range
        old_min = 0
        # need output to be in range -1 to 1
        new_max = 1.0
        new_min = -1.0
        new_h_norm_range = new_max - new_min
        new_w_norm_range = new_max - new_min
        for b in range(B):
            u = keypoint_pixel[b, :, 0]
            v = keypoint_pixel[b, :, 1]
            keypoint_pixel[b, :, 0] = (((u - old_min) * new_w_norm_range) / old_w_img_range) + new_min
            keypoint_pixel[b, :, 1] = (((v - old_min) * new_h_norm_range) / old_h_img_range) + new_min



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        FUSE_SPARSE = self.model_cfg.get('FUSE_SPARSE', True)
        if FUSE_SPARSE:
            x_conv1 = self.conv1(x)
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
            
            x_conv2 = self.conv2(x_conv1)
            ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########

            x_conv3 = self.conv3(x_conv2)
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########

            x_conv4 = self.conv4(x_conv3)
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            # Fuse for ROI - POST RPN Network
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
             ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
            


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelBackBone8xFuseConcatWeightedVoxelsSMALL(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        conv2_in_features = 16 * 2 # image features concatenated here
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(conv2_in_features, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        conv3_in_features = 32 * 2 # image features concatenated here
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(conv3_in_features, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        conv4_in_features = 64 * 2 # image features concatenated here
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(conv4_in_features, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        conv4_out_features = 64 * 2 # image features concatenated here
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(conv4_out_features, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 128
        }

        assert 'FUSE_LAYERS' in self.model_cfg and 'FUSE_MODE' in self.model_cfg
        # Fusion mode/layers - create mlp that learns channel-wise weight for each conv feature maps
        if 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv1_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv1']//2), nn.Sigmoid())
        if 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv2_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv2']//2), nn.Sigmoid())
        if 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv3_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv3']//2), nn.Sigmoid())
        if 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv4_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv4']//2), nn.Sigmoid())
    
    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        if self.model_cfg['FUSE_MODE'] == 'channel-fixed-weight':
            weighted_voxel_features = voxel_feature * image_foreground_weights.view(-1, 1)
            # Concat both image + voxel features # B, num_points, 2*features 
            voxel_image_concat = torch.cat((voxel_feature, weighted_voxel_features), dim=1)
            return voxel_image_concat
        elif self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
            assert vox_conv_layer is not None
            if vox_conv_layer == 'x_conv1':
                learned_channel_layer = self.x_conv1_mlp
            elif vox_conv_layer == 'x_conv2':
                learned_channel_layer = self.x_conv2_mlp
            elif vox_conv_layer == 'x_conv3':
                learned_channel_layer = self.x_conv3_mlp
            elif vox_conv_layer == 'x_conv4':
                learned_channel_layer = self.x_conv4_mlp
            else:
                raise NotImplementedError
            learned_channel_weights = learned_channel_layer(image_foreground_weights.view(-1, 1))
            return torch.cat((voxel_feature, (voxel_feature * learned_channel_weights)), dim=1)
        else:
            raise NotImplementedError

    
    # Ground truth training with grid sampler - no loss of precision
    def get_voxel_image_weights(self, batch_dict, conv_x_coords):
        B = batch_dict['trans_lidar_to_cam'].shape[0]
        image_h, image_w = batch_dict['images'].shape[-2:]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = conv_x_coords  # N
        ## This assumes number of keypoints per batch is fixed
        num_keypoints_per_sample = torch.sum(point_coords[..., 0].int() == 0).item()
        point_coords_kornia = torch.zeros((B, num_keypoints_per_sample, 3), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            points_in_batch_b = (point_coords[..., 0].int() == b)
            point_coords_kornia[b, ...] = point_coords[points_in_batch_b, 1:]

        # # Undo augmentations
        if 'undo_global_transform' in batch_dict:
            undo_global_transform = batch_dict['undo_global_transform']
            point_coords_kornia = point_coords_kornia @ undo_global_transform.inverse()

        # Transform to camera frame
        points_camera_frame = kornia.transform_points(trans_01=C_V, points_1=point_coords_kornia)

        # Project to image to get keypoint projection on image plane
        I_C = I_C.reshape(B, 3, 4)
        keypoints_img, keypoints_depths = transform_utils.project_to_image(project=I_C, points=points_camera_frame)

        # Get foreground weighting mask
        assert not('gt_boxes2d' in batch_dict and '2d_detections' in batch_dict) # only one source image can be sampled
        if 'gt_boxes2d' in batch_dict:
            gt_boxes2d = batch_dict['gt_boxes2d']
            image = batch_dict['images']
            mask_shape = (image.shape[0], image.shape[2] + 1, image.shape[3] + 1)
            foreground_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                                shape=mask_shape,
                                                downsample_factor=1,
                                                device=keypoints_img.device)
            segmentation_targets = torch.zeros(foreground_mask.shape, dtype=torch.float32, device=foreground_mask.device)
            segmentation_targets[foreground_mask.long() == True] = 1.0
        elif '2d_detections' in batch_dict:  
            segmentation_targets = batch_dict['2d_detections']
        
        
        # in-place keypoint location conversion to normalized pixel coordinates [-1, 1] for grid sampler
        self.convert_to_normalized_range(image_range=(image_h, image_w), keypoint_pixel=keypoints_img)
        # sampler
        image_voxel_features = torch.nn.functional.grid_sample(input=segmentation_targets.view(B, 1, segmentation_targets.shape[1], segmentation_targets.shape[2]),
                                        grid=keypoints_img.view(B, num_keypoints_per_sample, 1, 2), mode='bilinear',
                                        padding_mode='zeros', align_corners=None)
        
        return image_voxel_features

    
    # Takes keypoints locations in pixel coordinates and converts them to normalized coordinates [-1, 1] for grid sampling
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    @staticmethod
    def convert_to_normalized_range(image_range, keypoint_pixel):
        B = keypoint_pixel.shape[0]
        old_h_img_range, old_w_img_range = image_range
        old_min = 0
        # need output to be in range -1 to 1
        new_max = 1.0
        new_min = -1.0
        new_h_norm_range = new_max - new_min
        new_w_norm_range = new_max - new_min
        for b in range(B):
            u = keypoint_pixel[b, :, 0]
            v = keypoint_pixel[b, :, 1]
            keypoint_pixel[b, :, 0] = (((u - old_min) * new_w_norm_range) / old_w_img_range) + new_min
            keypoint_pixel[b, :, 1] = (((v - old_min) * new_h_norm_range) / old_h_img_range) + new_min



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        FUSE_SPARSE = self.model_cfg.get('FUSE_SPARSE', True)
        if FUSE_SPARSE:
            x_conv1 = self.conv1(x)
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
            
            x_conv2 = self.conv2(x_conv1)
            ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########

            x_conv3 = self.conv3(x_conv2)
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########

            x_conv4 = self.conv4(x_conv3)
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            # Fuse for ROI - POST RPN Network
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
             ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
            


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict


class VoxelBackBone8xFuseConcatWeightedVoxelsBIG(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        conv2_in_features = 16 * 2 # image features concatenated here
        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(conv2_in_features, conv2_in_features*2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(conv2_in_features*2, conv2_in_features*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(conv2_in_features*2, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        conv3_in_features = 32 * 2 # image features concatenated here
        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(conv3_in_features, conv3_in_features*2, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(conv3_in_features*2, conv3_in_features*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(conv3_in_features*2, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        conv4_in_features = 64 * 2 # image features concatenated here
        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(conv4_in_features, conv4_in_features*2, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(conv4_in_features*2, conv4_in_features*2, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(conv4_in_features*2, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        conv4_out_features = 64 * 2 # image features concatenated here
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(conv4_out_features, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 32,
            'x_conv2': 64,
            'x_conv3': 128,
            'x_conv4': 128
        }

        assert 'FUSE_LAYERS' in self.model_cfg and 'FUSE_MODE' in self.model_cfg
        # Fusion mode/layers - create mlp that learns channel-wise weight for each conv feature maps
        if 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv1_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv1']//2), nn.Sigmoid())
        if 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv2_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv2']//2), nn.Sigmoid())
        if 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv3_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv3']//2), nn.Sigmoid())
        if 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
            if self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
                self.x_conv4_mlp = nn.Sequential(nn.Linear(in_features=1, out_features=self.backbone_channels['x_conv4']//2), nn.Sigmoid())
    
    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        if self.model_cfg['FUSE_MODE'] == 'channel-fixed-weight':
            weighted_voxel_features = voxel_feature * image_foreground_weights.view(-1, 1)
            # Concat both image + voxel features # B, num_points, 2*features 
            voxel_image_concat = torch.cat((voxel_feature, weighted_voxel_features), dim=1)
            return voxel_image_concat
        elif self.model_cfg['FUSE_MODE'] == 'channel-learned-weight':
            assert vox_conv_layer is not None
            if vox_conv_layer == 'x_conv1':
                learned_channel_layer = self.x_conv1_mlp
            elif vox_conv_layer == 'x_conv2':
                learned_channel_layer = self.x_conv2_mlp
            elif vox_conv_layer == 'x_conv3':
                learned_channel_layer = self.x_conv3_mlp
            elif vox_conv_layer == 'x_conv4':
                learned_channel_layer = self.x_conv4_mlp
            else:
                raise NotImplementedError
            learned_channel_weights = learned_channel_layer(image_foreground_weights.view(-1, 1))
            return torch.cat((voxel_feature, (voxel_feature * learned_channel_weights)), dim=1)
        else:
            raise NotImplementedError

    
    # Ground truth training with grid sampler - no loss of precision
    def get_voxel_image_weights(self, batch_dict, conv_x_coords):
        B = batch_dict['trans_lidar_to_cam'].shape[0]
        image_h, image_w = batch_dict['images'].shape[-2:]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = conv_x_coords  # N
        ## This assumes number of keypoints per batch is fixed
        num_keypoints_per_sample = torch.sum(point_coords[..., 0].int() == 0).item()
        point_coords_kornia = torch.zeros((B, num_keypoints_per_sample, 3), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            points_in_batch_b = (point_coords[..., 0].int() == b)
            point_coords_kornia[b, ...] = point_coords[points_in_batch_b, 1:]

        # # Undo augmentations
        if 'undo_global_transform' in batch_dict:
            undo_global_transform = batch_dict['undo_global_transform']
            point_coords_kornia = point_coords_kornia @ undo_global_transform.inverse()

        # Transform to camera frame
        points_camera_frame = kornia.transform_points(trans_01=C_V, points_1=point_coords_kornia)

        # Project to image to get keypoint projection on image plane
        I_C = I_C.reshape(B, 3, 4)
        keypoints_img, keypoints_depths = transform_utils.project_to_image(project=I_C, points=points_camera_frame)

        # Get foreground weighting mask
        assert not('gt_boxes2d' in batch_dict and '2d_detections' in batch_dict) # only one source image can be sampled
        if 'gt_boxes2d' in batch_dict:
            gt_boxes2d = batch_dict['gt_boxes2d']
            image = batch_dict['images']
            mask_shape = (image.shape[0], image.shape[2] + 1, image.shape[3] + 1)
            foreground_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                                shape=mask_shape,
                                                downsample_factor=1,
                                                device=keypoints_img.device)
            segmentation_targets = torch.zeros(foreground_mask.shape, dtype=torch.float32, device=foreground_mask.device)
            segmentation_targets[foreground_mask.long() == True] = 1.0
        elif '2d_detections' in batch_dict:  
            segmentation_targets = batch_dict['2d_detections']
        
        
        # in-place keypoint location conversion to normalized pixel coordinates [-1, 1] for grid sampler
        self.convert_to_normalized_range(image_range=(image_h, image_w), keypoint_pixel=keypoints_img)
        # sampler
        image_voxel_features = torch.nn.functional.grid_sample(input=segmentation_targets.view(B, 1, segmentation_targets.shape[1], segmentation_targets.shape[2]),
                                        grid=keypoints_img.view(B, num_keypoints_per_sample, 1, 2), mode='bilinear',
                                        padding_mode='zeros', align_corners=None)
        
        return image_voxel_features

    
    # Takes keypoints locations in pixel coordinates and converts them to normalized coordinates [-1, 1] for grid sampling
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    @staticmethod
    def convert_to_normalized_range(image_range, keypoint_pixel):
        B = keypoint_pixel.shape[0]
        old_h_img_range, old_w_img_range = image_range
        old_min = 0
        # need output to be in range -1 to 1
        new_max = 1.0
        new_min = -1.0
        new_h_norm_range = new_max - new_min
        new_w_norm_range = new_max - new_min
        for b in range(B):
            u = keypoint_pixel[b, :, 0]
            v = keypoint_pixel[b, :, 1]
            keypoint_pixel[b, :, 0] = (((u - old_min) * new_w_norm_range) / old_w_img_range) + new_min
            keypoint_pixel[b, :, 1] = (((v - old_min) * new_h_norm_range) / old_h_img_range) + new_min



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        FUSE_SPARSE = self.model_cfg.get('FUSE_SPARSE', True)
        if FUSE_SPARSE:
            x_conv1 = self.conv1(x)
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
            
            x_conv2 = self.conv2(x_conv1)
            ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########

            x_conv3 = self.conv3(x_conv2)
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########

            x_conv4 = self.conv4(x_conv3)
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            # Fuse for ROI - POST RPN Network
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
             ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
            


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict



class VoxelBackBone8xFuseConcatForegroundWeight(nn.Module):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__()
        self.model_cfg = model_cfg
        self.voxel_size = voxel_size
        self.point_cloud_range = point_cloud_range
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(input_channels, 16, 3, padding=1, bias=False, indice_key='subm1'),
            norm_fn(16),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            block(16, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(17, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(33, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(65, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
        )

        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(65, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

        self.num_point_features = 128
        self.backbone_channels = {
            'x_conv1': 17,
            'x_conv2': 33,
            'x_conv3': 65,
            'x_conv4': 65
        }

        assert 'FUSE_LAYERS' in self.model_cfg and 'FUSE_MODE' in self.model_cfg
  
    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        if self.model_cfg['FUSE_MODE'] == 'channel-fixed-weight':
            voxel_image_concat = torch.cat((voxel_feature, image_foreground_weights.view(-1, 1)), dim=1)
            return voxel_image_concat
        else:
            raise NotImplementedError

    
    # Ground truth training with grid sampler - no loss of precision
    def get_voxel_image_weights(self, batch_dict, conv_x_coords):
        B = batch_dict['trans_lidar_to_cam'].shape[0]
        image_h, image_w = batch_dict['images'].shape[-2:]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = conv_x_coords  # N
        ## This assumes number of keypoints per batch is fixed
        num_keypoints_per_sample = torch.sum(point_coords[..., 0].int() == 0).item()
        point_coords_kornia = torch.zeros((B, num_keypoints_per_sample, 3), dtype=torch.float32,
                                          device=point_coords.device)
        for b in range(B):
            points_in_batch_b = (point_coords[..., 0].int() == b)
            point_coords_kornia[b, ...] = point_coords[points_in_batch_b, 1:]

        # # Undo augmentations
        if 'undo_global_transform' in batch_dict:
            undo_global_transform = batch_dict['undo_global_transform']
            point_coords_kornia = point_coords_kornia @ undo_global_transform.inverse()

        # Transform to camera frame
        points_camera_frame = kornia.transform_points(trans_01=C_V, points_1=point_coords_kornia)

        # Project to image to get keypoint projection on image plane
        I_C = I_C.reshape(B, 3, 4)
        keypoints_img, keypoints_depths = transform_utils.project_to_image(project=I_C, points=points_camera_frame)

        # Get foreground weighting mask
        assert not('gt_boxes2d' in batch_dict and '2d_detections' in batch_dict) # only one source image can be sampled
        if 'gt_boxes2d' in batch_dict:
            gt_boxes2d = batch_dict['gt_boxes2d']
            image = batch_dict['images']
            mask_shape = (image.shape[0], image.shape[2] + 1, image.shape[3] + 1)
            foreground_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                                shape=mask_shape,
                                                downsample_factor=1,
                                                device=keypoints_img.device)
            segmentation_targets = torch.zeros(foreground_mask.shape, dtype=torch.float32, device=foreground_mask.device)
            segmentation_targets[foreground_mask.long() == True] = 1.0
        elif '2d_detections' in batch_dict:  
            segmentation_targets = batch_dict['2d_detections']
        
        
        # in-place keypoint location conversion to normalized pixel coordinates [-1, 1] for grid sampler
        self.convert_to_normalized_range(image_range=(image_h, image_w), keypoint_pixel=keypoints_img)
        # sampler
        image_voxel_features = torch.nn.functional.grid_sample(input=segmentation_targets.view(B, 1, segmentation_targets.shape[1], segmentation_targets.shape[2]),
                                        grid=keypoints_img.view(B, num_keypoints_per_sample, 1, 2), mode='bilinear',
                                        padding_mode='zeros', align_corners=None)
        
        return image_voxel_features

    
    # Takes keypoints locations in pixel coordinates and converts them to normalized coordinates [-1, 1] for grid sampling
    # NewValue = (((OldValue - OldMin) * NewRange) / OldRange) + NewMin
    @staticmethod
    def convert_to_normalized_range(image_range, keypoint_pixel):
        B = keypoint_pixel.shape[0]
        old_h_img_range, old_w_img_range = image_range
        old_min = 0
        # need output to be in range -1 to 1
        new_max = 1.0
        new_min = -1.0
        new_h_norm_range = new_max - new_min
        new_w_norm_range = new_max - new_min
        for b in range(B):
            u = keypoint_pixel[b, :, 0]
            v = keypoint_pixel[b, :, 1]
            keypoint_pixel[b, :, 0] = (((u - old_min) * new_w_norm_range) / old_w_img_range) + new_min
            keypoint_pixel[b, :, 1] = (((v - old_min) * new_h_norm_range) / old_h_img_range) + new_min



    def forward(self, batch_dict):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)

        FUSE_SPARSE = self.model_cfg.get('FUSE_SPARSE', True)
        if FUSE_SPARSE:
            x_conv1 = self.conv1(x)
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
            
            x_conv2 = self.conv2(x_conv1)
            ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########

            x_conv3 = self.conv3(x_conv2)
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########

            x_conv4 = self.conv4(x_conv3)
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            # Fuse for ROI - POST RPN Network
            ### APPLY foreground weights on features of conv1
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 1
                # weight voxel features based on foreground mask
                # x_conv1.features # voxels * 16
                conv_voxel_coord = x_conv1.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv1.features = self.fuse(voxel_feature=x_conv1.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv1')
            ########
             ### APPLY foreground weights on features of conv2
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 2
                # weight voxel features based on foreground mask
                # x_conv2.features # voxels * 32
                conv_voxel_coord = x_conv2.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv2.features = self.fuse(voxel_feature=x_conv2.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv2')
            ########
            ### APPLY foreground weights on features of conv3
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 4
                # weight voxel features based on foreground mask
                # x_conv3.features # voxels * 64
                conv_voxel_coord = x_conv3.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv3.features = self.fuse(voxel_feature=x_conv3.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv3')
            ########
            ### APPLY foreground weights on features of conv4
            if 'FUSE_LAYERS' in self.model_cfg and 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                VOXEL_GRID_DOWNSAMPLE = 8
                # weight voxel features based on foreground mask
                # x_conv4.features # voxels * 64
                conv_voxel_coord = x_conv4.indices # voxels * 4 (first dim batch number)
                voxel_centers_xyz = common_utils.get_voxel_centers(
                        conv_voxel_coord[:, 1:4],
                        downsample_times=VOXEL_GRID_DOWNSAMPLE,
                        voxel_size=self.voxel_size,
                        point_cloud_range=self.point_cloud_range
                    )
                voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)
                image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz)
                x_conv4.features = self.fuse(voxel_feature=x_conv4.features, image_foreground_weights=image_voxel_features, vox_conv_layer='x_conv4')
            ########
            


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        batch_dict.update({
            'encoded_spconv_tensor': out,
            'encoded_spconv_tensor_stride': 8
        })
        batch_dict.update({
            'multi_scale_3d_features': {
                'x_conv1': x_conv1,
                'x_conv2': x_conv2,
                'x_conv3': x_conv3,
                'x_conv4': x_conv4,
            }
        })
        batch_dict.update({
            'multi_scale_3d_strides': {
                'x_conv1': 1,
                'x_conv2': 2,
                'x_conv3': 4,
                'x_conv4': 8,
            }
        })

        return batch_dict



