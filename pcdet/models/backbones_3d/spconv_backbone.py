from functools import partial
from os import stat
from pickle import ADDITEMS
from numpy.lib.utils import source

import spconv
from torch._C import device
import torch.nn as nn
import torch
import numpy as np
from ...utils import common_utils, transform_utils, loss_utils
import kornia
import torch.nn.functional as F
from pcdet.ops.pointnet2.pointnet2_stack.pointnet2_utils import QueryAndGroup
from pcdet.ops.roiaware_pool3d.roiaware_pool3d_utils import points_in_boxes_gpu


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
        
        
        WEIGHT_SRC = self.model_cfg.get('WEIGHT_SRC', None)
        if WEIGHT_SRC == 'POINTS'  or WEIGHT_SRC == 'WEIGHTED_POINTS':
            if 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
                downsample_times = 1
                voxel_size = [vox_dim * downsample_times for vox_dim in self.voxel_size]
                approximate_radius = np.sqrt(voxel_size[0] * voxel_size[0] + voxel_size[1] * voxel_size[1] + voxel_size[2] * voxel_size[2])/2
                self.query_and_group_conv1 = QueryAndGroup(radius=approximate_radius, nsample=16)
            if 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
                downsample_times = 2
                voxel_size = [vox_dim * downsample_times for vox_dim in self.voxel_size]
                approximate_radius = np.sqrt(voxel_size[0] * voxel_size[0] + voxel_size[1] * voxel_size[1] + voxel_size[2] * voxel_size[2])/2
                self.query_and_group_conv2 = QueryAndGroup(radius=approximate_radius, nsample=16)
            if 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
                downsample_times = 4
                voxel_size = [vox_dim * downsample_times for vox_dim in self.voxel_size]
                approximate_radius = np.sqrt(voxel_size[0] * voxel_size[0] + voxel_size[1] * voxel_size[1] + voxel_size[2] * voxel_size[2])/2
                self.query_and_group_conv3 = QueryAndGroup(radius=approximate_radius, nsample=32)
            if 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
                downsample_times = 8
                voxel_size = [vox_dim * downsample_times for vox_dim in self.voxel_size]
                approximate_radius = np.sqrt(voxel_size[0] * voxel_size[0] + voxel_size[1] * voxel_size[1] + voxel_size[2] * voxel_size[2])/2
                self.query_and_group_conv4 = QueryAndGroup(radius=approximate_radius, nsample=32)

    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        if self.model_cfg['FUSE_MODE'] == 'channel-fixed-weight':
            if self.model_cfg.get('FEATURE_DECAY', False):
                alpha = 0.5
                return ((1 - alpha) * (voxel_feature * image_foreground_weights.view(-1, 1))) + (alpha * voxel_feature)
            else:
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
            if self.model_cfg.get('FEATURE_DECAY', False):
                alpha = 0.5
                return ((1 - alpha) * (voxel_feature * learned_channel_weights)) + (alpha * voxel_feature)
            else:
                return (voxel_feature * learned_channel_weights) + voxel_feature 
        else:
            raise NotImplementedError

    
    
    # use points_in_boxes_gpu function to map points to voxels and then compute reverse - VERY SLOW but more accurate than ball query
    # cube in a sphere problem
    def point_aggregation_weighting_using_boxes(self, batch_dict, conv_layer_name, voxel_centers_xyz, raw_points_weights):
        raw_points = batch_dict['points'][:,1:-1].unsqueeze(0)
        boxes = torch.zeros((1, voxel_centers_xyz.shape[0], 7), device=raw_points.device, dtype=raw_points.dtype)
        # voxel centeriod x, y, z
        boxes[0, :, 0] = voxel_centers_xyz[:, 1]
        boxes[0, :, 1] = voxel_centers_xyz[:, 2]
        boxes[0, :, 2] = voxel_centers_xyz[:, 3]
        l = self.voxel_size[0]
        w = self.voxel_size[1]
        h = self.voxel_size[2]
        if conv_layer_name == 'x_conv1': 
            downsample = 1
        elif conv_layer_name == 'x_conv2':
            downsample = 2
        elif  conv_layer_name == 'x_conv3':
            downsample = 4
        elif  conv_layer_name == 'x_conv4':
            downsample = 8
        else:
            raise NotImplementedError
        l = downsample * l
        w = downsample * w
        h = downsample * h
        boxes[0, :, 3] = l
        boxes[0, :, 4] = w
        boxes[0, :, 5] = h
        box_idxs_of_pts = points_in_boxes_gpu(raw_points, boxes).view(-1, 1)
        num_points_in_voxel = torch.zeros((voxel_centers_xyz.shape[0], 1), device=raw_points.device, dtype=raw_points.dtype)
        fg_weight_points_in_voxel = torch.zeros((voxel_centers_xyz.shape[0], 1), device=raw_points.device, dtype=raw_points.dtype)
        for pts_idx in range(0, box_idxs_of_pts.shape[0]):
            voxel_idx = box_idxs_of_pts[pts_idx].long().item()
            num_points_in_voxel[voxel_idx] += 1
            fg_weight_points_in_voxel[voxel_idx] += raw_points_weights[pts_idx]

        image_voxel_features = fg_weight_points_in_voxel/num_points_in_voxel
        # if num_points_in_voxel is zero, this will result in nan - trick to remove nan
        image_voxel_features[image_voxel_features != image_voxel_features] = 0
            
        return image_voxel_features
    
    # use ball guery to aggregate points weights using mean or weighted mean
    def point_aggregation_weighting(self, batch_dict, conv_layer_name, voxel_centers_xyz, raw_points_weights):

        batch_size = batch_dict['batch_size']

        # Get raw points and batch count (we use a batch size of one per GPU)
        raw_points = batch_dict['points']
        raw_xyz = raw_points[:, 1:4]
        raw_xyz_batch_cnt = raw_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            raw_xyz_batch_cnt[bs_idx] = (raw_points[:, 0] == bs_idx).sum()


        new_xyz = voxel_centers_xyz[:, 1:]
        new_xyz_batch_cnt = new_xyz.new_zeros(batch_size).int()
        for bs_idx in range(batch_size):
            new_xyz_batch_cnt[bs_idx] = (voxel_centers_xyz[:, 0] == bs_idx).sum()

        if conv_layer_name == 'x_conv1': 
            new_features, idx = self.query_and_group_conv1(xyz=raw_xyz.contiguous(), xyz_batch_cnt=raw_xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt,features = raw_points_weights)
        elif conv_layer_name == 'x_conv2':
            new_features, idx = self.query_and_group_conv2(xyz=raw_xyz.contiguous(), xyz_batch_cnt=raw_xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt,features = raw_points_weights)
        elif  conv_layer_name == 'x_conv3':
            new_features, idx = self.query_and_group_conv3(xyz=raw_xyz.contiguous(), xyz_batch_cnt=raw_xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt,features = raw_points_weights) 
        elif  conv_layer_name == 'x_conv4':
            new_features, idx = self.query_and_group_conv4(xyz=raw_xyz.contiguous(), xyz_batch_cnt=raw_xyz_batch_cnt, new_xyz=new_xyz.contiguous(), new_xyz_batch_cnt=new_xyz_batch_cnt,features = raw_points_weights) 
        else:
            raise NotImplementedError
        
        feature_dims = new_features.shape[1]
        
        # remove repeated first quered points
        first_queried_points_idx = idx[...,0]
        idx_cleaned = torch.where(idx > first_queried_points_idx.view(-1,1), idx, torch.tensor(-1, dtype=torch.int32, device=idx.device))
        idx_cleaned[...,0] = first_queried_points_idx
        # set C of repeated/redundant features within ball query samples to -1 for all voxels
        flag_idx = idx_cleaned > -1 # create bool for non-redundant indices within a ball query
        flag_idx = flag_idx.reshape(flag_idx.shape[0], 1, flag_idx.shape[1])
        flag_idx_repeat = flag_idx.repeat(1, feature_dims, 1)
        # cleaned features from redundunt points
        new_features = new_features * flag_idx_repeat
        
        ## Distance-based weighting of point foreground weights within ball query
        if self.model_cfg.get('WEIGHT_SRC', None) == 'WEIGHTED_POINTS':
            ball_query_delta_xyz = new_features[:, 0:3, :]
            absolute_distance = torch.sqrt(torch.sum(ball_query_delta_xyz * ball_query_delta_xyz, dim=1))
            weight_per_point = 1.0 / absolute_distance
            weight_per_point[weight_per_point == float("Inf")] = 0
            sum_weight_per_query = torch.sum(weight_per_point, dim=1).view(-1, 1)
            normalized_weight_per_point = weight_per_point / sum_weight_per_query
            # ball query doesnt not remove empty balls (sum_weight_per_query can be zero!) - trick to remove nans
            normalized_weight_per_point[normalized_weight_per_point != normalized_weight_per_point] = 0
            # weight foreground mask by inverse of distance weighting per query 
            # (closer points to voxel centers in 3D should contribute more to voxel feature weight)
            foreground_mask = new_features[:, feature_dims-1, :].squeeze()
            weighted_foreground_mask = foreground_mask * normalized_weight_per_point
            image_voxel_features = torch.sum(weighted_foreground_mask, dim=1)
        elif self.model_cfg.get('WEIGHT_SRC', None) == 'POINTS':
            # NO weighting based on distance from voxel center
            # count num of valid points for each voxel 
            num_valid_points_per_voxel = torch.sum(flag_idx, dim=2)
            sum_point_weights_per_voxel = torch.sum(new_features[:, feature_dims - 1, :], dim=1)
            image_voxel_features = sum_point_weights_per_voxel.view(-1, 1) / num_valid_points_per_voxel.view(-1, 1)
        else:
            raise NotImplementedError

        return image_voxel_features
    
    
    def point_to_img(self, batch_dict, points):
        B = batch_dict['trans_lidar_to_cam'].shape[0]
        image_h, image_w = batch_dict['images'].shape[-2:]

        # Create transformation matricies
        C_V = batch_dict['trans_lidar_to_cam']  # LiDAR -> Camera (B, 4, 4)
        I_C = batch_dict['trans_cam_to_img']  # Camera -> Image (B, 3, 4)

        # Reshape points coordinates from (N, 4) to (B, N, 3)
        point_coords = points  # N
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
        return keypoints_img
        
    
    def visualise(self, batch_dict, voxel_centers_xyz, voxel_weights):
        source_img = batch_dict['images'][0].permute(1, 2, 0).cpu().numpy()
        source_img[..., 0] = 0
        source_img[..., 1] = 0
        img_pixels = self.point_to_img(batch_dict ,voxel_centers_xyz).squeeze().long().cpu().numpy()
        img_pixels[img_pixels[:, 0] >= source_img.shape[1]] = 0
        img_pixels[img_pixels[:, 1] >= source_img.shape[0]] = 0        

        THRESHOLD = 0.99 
        img_pixels[voxel_weights.cpu().numpy().reshape(-1,) < THRESHOLD] = 0
        
        source_img[img_pixels[:,1], img_pixels[:, 0], 0] = 1

        return source_img

    def visualise_raw_points(self, batch_dict):
        raw_points = batch_dict['points'][:, 0:-1]
        source_img = batch_dict['images'][0].permute(1, 2, 0).cpu().numpy()
        source_img[..., 0] = 0
        source_img[..., 1] = 0
        img_pixels = self.point_to_img(batch_dict , raw_points).squeeze().long().cpu().numpy()
        img_pixels[img_pixels[:, 0] >= source_img.shape[1]] = 0
        img_pixels[img_pixels[:, 1] >= source_img.shape[0]] = 0

        source_img[img_pixels[:, 1], img_pixels[:, 0], 0] = 1

        return source_img

    
    def visualize_fg_bg_voxels(self, batch_dict, voxel_centers_xyz, voxel_weights):
        source_img = batch_dict['images'][0].permute(1, 2, 0).cpu().numpy()
        source_img[..., 0] = 0
        source_img[..., 1] = 0
        img_pixels = self.point_to_img(batch_dict ,voxel_centers_xyz).squeeze().long().cpu().numpy()
        img_pixels[img_pixels[:, 0] >= source_img.shape[1]] = 0
        img_pixels[img_pixels[:, 1] >= source_img.shape[0]] = 0        

        THRESHOLD = 0.99 
        img_pixels_fg = np.copy(img_pixels)
        img_pixels_bg = np.copy(img_pixels)
        
        img_pixels_fg[voxel_weights.cpu().numpy().reshape(-1,) < THRESHOLD] = 0
        img_pixels_bg[img_pixels_fg >=  THRESHOLD] = 0
        
        source_img[img_pixels_fg[:,1], img_pixels_fg[:, 0], 0] = 1

        return source_img


    def get_image_feature_map(self, batch_dict):
        if 'segment_logits' not in batch_dict: # a segmentation network is not used 
            # Get foreground weighting mask - # only one source image can be sampled
            assert not('gt_boxes2d' in batch_dict and '2d_detections' in batch_dict) 
            assert not('gt_boxes2d' in batch_dict and 'segmentation_mask' in batch_dict) 
            assert not('2d_detections' in batch_dict and 'segmentation_mask' in batch_dict)
            if 'gt_boxes2d' in batch_dict:
                gt_boxes2d = batch_dict['gt_boxes2d']
                image = batch_dict['images']
                mask_shape = (image.shape[0], image.shape[2] + 1, image.shape[3] + 1)
                foreground_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                                    shape=mask_shape,
                                                    downsample_factor=1,
                                                    device=batch_dict['gt_boxes2d'].device,
                                                    keep_boundingbox_percentage=100.0, 
                                                    max_boundary_shift=0)
                segmentation_targets = torch.zeros(foreground_mask.shape, dtype=torch.float32, device=foreground_mask.device)
                segmentation_targets[foreground_mask.long() == True] = 1.0
                # blurred_torch = kornia.gaussian_blur2d(torch.unsqueeze(segmentation_targets, 0), (5, 5), (3, 3))
                # segmentation_targets = torch.squeeze(blurred_torch, 0)
            elif '2d_detections' in batch_dict:  
                segmentation_targets = batch_dict['2d_detections']
            elif 'segmentation_mask' in batch_dict:
                segmentation_targets = batch_dict['segmentation_mask']
            else:
                raise NotImplementedError
        else:
            # use output of segmentation network for voxel feature weighting
            segment_logits = batch_dict['segment_logits']
            foreground_channel = 1
            segmentation_targets = F.softmax(segment_logits, dim=1)[:, foreground_channel, :, :]

        if 'image_flip' in batch_dict and batch_dict['image_flip'] == 1:
            segmentation_targets = torch.flip(segmentation_targets, [2])
        
        return segmentation_targets
        
    
    # Ground truth training with grid sampler - no loss of precision
    def get_voxel_image_weights(self, batch_dict, conv_x_coords, image_feature_map):
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
        
        # in-place keypoint location conversion to normalized pixel coordinates [-1, 1] for grid sampler
        self.convert_to_normalized_range(image_range=(image_h, image_w), keypoint_pixel=keypoints_img)
        # sampler
        image_voxel_features = torch.nn.functional.grid_sample(input=image_feature_map.view(B, 1, image_feature_map.shape[1], image_feature_map.shape[2]),
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


    # returns the image features/ foreground weights of each non-empty voxel in a specific conv layer
    def get_conv_layer_image_based_feature(self, batch_dict, conv_name, conv_output, image_feature_map, raw_point_weights):
        """
        Compute foreground mask for images
        Args:
            conv_name: str, name of convolution layer
            conv_output: sparse convolution output containing indices and features of non-empty voxels 
            image_feature_map: torch.tensor (1, H, W), image feature map used for weighting voxel features
            raw_point_weights: torch.tensor, foreground/background weight of each point in raw point cloud,
            raw_point_weights is only used if WEIGHT_SRC is not None
        Returns:
            image_voxel_features: weight of voxel features 
        """
        if conv_name == 'x_conv1':
            VOXEL_GRID_DOWNSAMPLE = 1
        elif conv_name == 'x_conv2':
            VOXEL_GRID_DOWNSAMPLE = 2
        elif conv_name == 'x_conv3':
            VOXEL_GRID_DOWNSAMPLE = 4
        elif conv_name == 'x_conv4':
            VOXEL_GRID_DOWNSAMPLE = 8
        else:
            raise NotImplementedError

        # associate voxel feature weight based on voxel center xyz projection or based on projection of points nearby or a combination
        # default is voxel center, i.e., WEIGHT is None
        WEIGHT_SRC = self.model_cfg.get('WEIGHT_SRC', None)
        AGGREGATE = self.model_cfg.get('AGGREGATE', False)
        AGGREGATE_METHOD = self.model_cfg.get('AGGREGATE_METHOD', None)
        
        # Extract voxel centers
        conv_voxel_coord = conv_output.indices # voxels * 4 (first dim batch number)
        voxel_centers_xyz = common_utils.get_voxel_centers(
                conv_voxel_coord[:, 1:4],
                downsample_times=VOXEL_GRID_DOWNSAMPLE,
                voxel_size=self.voxel_size,
                point_cloud_range=self.point_cloud_range
            )
        voxel_centers_xyz = torch.cat((conv_voxel_coord[:, 0].view((-1, 1)), voxel_centers_xyz), axis=1)# convert voxel centers from (N,3) back to (N,4)

        # get voxel foreground weights based on raw point weights OR from voxel centers weights
        if  WEIGHT_SRC == 'POINTS'  or WEIGHT_SRC == 'WEIGHTED_POINTS':
            image_voxel_features = self.point_aggregation_weighting(batch_dict, conv_name, voxel_centers_xyz, raw_points_weights=raw_point_weights)
            if AGGREGATE:
                # aggregate voxel center weights AND point-aggregation weights
                voxel_center_weights = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map)
                if AGGREGATE_METHOD == 'MEAN':
                    image_voxel_features = (image_voxel_features.view(-1, 1) + voxel_center_weights.view(-1, 1))/2
                elif AGGREGATE_METHOD == 'MAX':
                    image_voxel_features = torch.max(image_voxel_features.view(-1, 1), voxel_center_weights.view(-1, 1))
                else:
                    raise NotImplementedError

        else:
            image_voxel_features = self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map)
        
        return image_voxel_features, voxel_centers_xyz 
    
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

        
        DEBUG_FLAG = False
        
        WEIGHT_SRC = self.model_cfg.get('WEIGHT_SRC', None)
        AGGREGATE = self.model_cfg.get('AGGREGATE', False)
        AGGREGATE_METHOD = self.model_cfg.get('AGGREGATE_METHOD', None)
        # cant aggregate without method of aggregation
        if AGGREGATE:
            assert AGGREGATE_METHOD != 'None'
            # Two methods to aggregate voxel_center weights and point aggregation weights
            assert (AGGREGATE_METHOD == 'MEAN' or AGGREGATE_METHOD == 'MAX')


        # get image information
        image_feature_map = batch_dict['image_foreground_mask']
        
        if WEIGHT_SRC == 'POINTS'  or WEIGHT_SRC == 'WEIGHTED_POINTS':
            # Find foreground weights of points from image
            point_weights = torch.reshape(self.get_voxel_image_weights(batch_dict, conv_x_coords=batch_dict['points'][:, 0:-1], image_feature_map=image_feature_map), (-1, 1))
            raw_point_on_image = self.visualise_raw_points(batch_dict)
        else:
            point_weights = None
            

        if self.model_cfg.get('BLURR', False):
            kernel_x, kernel_y = 5, 5
            sigma_x, sigma_y = 5, 5
            image_feature_map = torch.reshape(image_feature_map, (1, 1, image_feature_map.shape[1], image_feature_map.shape[2]))
            image_feature_map = torch.squeeze(kornia.gaussian_blur2d(image_feature_map, (kernel_x, kernel_y), (sigma_x, sigma_y)), 0)
        
        FUSE_SPARSE = self.model_cfg.get('FUSE_SPARSE', True)
        if FUSE_SPARSE:
            x_conv1 = self.conv1(x)
            conv_name = 'x_conv1'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
            if DEBUG_FLAG:
                voxel_center_weights1 = torch.squeeze(self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map))
                voxel_center1 = voxel_centers_xyz
            
            
            x_conv2 = self.conv2(x_conv1)
            conv_name = 'x_conv2'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
            if DEBUG_FLAG:
                voxel_center_weights2 = torch.squeeze(self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map))
                voxel_center2 = voxel_centers_xyz



            x_conv3 = self.conv3(x_conv2)
            conv_name = 'x_conv3'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
            if DEBUG_FLAG:
                voxel_center_weights3 = torch.squeeze(self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map))
                voxel_center3 = voxel_centers_xyz
 

            x_conv4 = self.conv4(x_conv3)
            conv_name = 'x_conv4'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
            if DEBUG_FLAG:
                voxel_center_weights4 = torch.squeeze(self.get_voxel_image_weights(batch_dict, conv_x_coords=voxel_centers_xyz, image_feature_map=image_feature_map))
                voxel_center4 = voxel_centers_xyz

        else:
            x_conv1 = self.conv1(x)
            x_conv2 = self.conv2(x_conv1)
            x_conv3 = self.conv3(x_conv2)
            x_conv4 = self.conv4(x_conv3)
            # post conv DVF 
            
            conv_name = 'x_conv1'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
            conv_name = 'x_conv2'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)

            conv_name = 'x_conv3'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)

            conv_name = 'x_conv4'
            conv_output = locals()[conv_name]
            if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
                image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name=conv_name, 
                                                                            conv_output=conv_output, 
                                                                            image_feature_map=image_feature_map, 
                                                                            raw_point_weights=point_weights)
                conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)


        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)
        if DEBUG_FLAG:
            # visualize voxel centroid weighting
            import numpy as np
            import matplotlib.pyplot as plt
            VISUALIZE_HORIZONTAL = False
            if VISUALIZE_HORIZONTAL:
                voxel_centers_xyz_all = torch.cat((voxel_center1, voxel_center2, voxel_center3, voxel_center4), 0)
                voxel_center_weights_all = torch.cat((voxel_center_weights1, voxel_center_weights2, voxel_center_weights3, voxel_center_weights4), 0)
                img_combine_fg_bg_voxels = self.visualize_fg_bg_voxels(batch_dict, voxel_centers_xyz_all, voxel_center_weights_all)
                raw_point_on_image = self.visualise_raw_points(batch_dict)
                raw_image = (torch.squeeze(batch_dict['images'][0,...]).transpose(0, 1)).transpose(1, 2).cpu().numpy()
                both = np.concatenate([raw_image, raw_point_on_image, img_combine_fg_bg_voxels], axis=0)
                plt.figure(dpi=1200)
                plt.axis('off')
                plt.imshow(both)
                plt.show()
            else:
                # vertical
                voxel_centers_xyz_all = voxel_center1
                voxel_center_weights_all = voxel_center_weights1
                img_combine_fg_bg_voxels = self.visualize_fg_bg_voxels(batch_dict, voxel_centers_xyz_all, voxel_center_weights_all)
                raw_point_on_image = self.visualise_raw_points(batch_dict)
                both1 = np.concatenate([raw_point_on_image, img_combine_fg_bg_voxels], axis=0)
                
                voxel_centers_xyz_all = voxel_center2
                voxel_center_weights_all = voxel_center_weights2
                img_combine_fg_bg_voxels = self.visualize_fg_bg_voxels(batch_dict, voxel_centers_xyz_all, voxel_center_weights_all)
                raw_point_on_image = self.visualise_raw_points(batch_dict)
                both2 = np.concatenate([raw_point_on_image, img_combine_fg_bg_voxels], axis=0)

                voxel_centers_xyz_all = voxel_center3
                voxel_center_weights_all = voxel_center_weights3
                img_combine_fg_bg_voxels = self.visualize_fg_bg_voxels(batch_dict, voxel_centers_xyz_all, voxel_center_weights_all)
                raw_point_on_image = self.visualise_raw_points(batch_dict)
                both3 = np.concatenate([raw_point_on_image, img_combine_fg_bg_voxels], axis=0)

                voxel_centers_xyz_all = voxel_center4
                voxel_center_weights_all = voxel_center_weights4
                img_combine_fg_bg_voxels = self.visualize_fg_bg_voxels(batch_dict, voxel_centers_xyz_all, voxel_center_weights_all)
                raw_point_on_image = self.visualise_raw_points(batch_dict)
                both4 = np.concatenate([raw_point_on_image, img_combine_fg_bg_voxels], axis=0)
                
                both = np.concatenate([both1, both2, both3, both4], axis=1)
                plt.figure(dpi=1200)
                plt.axis('off')
                plt.imshow(both)
                plt.show()
                # plt.savefig("/home/nas/Desktop/vs-open-pcdet/OpenPCDet/output/kitti_models/visualize_voxels/{}.png".format(str(batch_dict['frame_id'][0])))
                # plt.close()

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


class VoxelBackBone8xFuseConCat(VoxelBackBone8xFuse):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block

        if 'x_conv1' in self.model_cfg['FUSE_LAYERS']:
            self.conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(17, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            )
        else:
            self.conv2 = spconv.SparseSequential(
                # [1600, 1408, 41] <- [800, 704, 21]
                block(16, 32, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv2', conv_type='spconv'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
                block(32, 32, 3, norm_fn=norm_fn, padding=1, indice_key='subm2'),
            )

        if 'x_conv2' in self.model_cfg['FUSE_LAYERS']:
            self.conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(33, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            )
        else:
            self.conv3 = spconv.SparseSequential(
                # [800, 704, 21] <- [400, 352, 11]
                block(32, 64, 3, norm_fn=norm_fn, stride=2, padding=1, indice_key='spconv3', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm3'),
            )

        if 'x_conv3' in self.model_cfg['FUSE_LAYERS']:
            self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(65, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )
        else:
            self.conv4 = spconv.SparseSequential(
                # [400, 352, 11] <- [200, 176, 5]
                block(64, 64, 3, norm_fn=norm_fn, stride=2, padding=(0, 1, 1), indice_key='spconv4', conv_type='spconv'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
                block(64, 64, 3, norm_fn=norm_fn, padding=1, indice_key='subm4'),
            )


        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        if 'x_conv4' in self.model_cfg['FUSE_LAYERS']:
            self.conv_out = spconv.SparseSequential(
                # [200, 150, 5] -> [200, 150, 2]
                spconv.SparseConv3d(65, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                    bias=False, indice_key='spconv_down2'),
                norm_fn(128),
                nn.ReLU(),
            )
        else:
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
    
    def fuse(self, voxel_feature, image_foreground_weights, vox_conv_layer=None):
        concat = torch.cat((voxel_feature, image_foreground_weights.view(-1, 1)), 1)
        return concat

# Colorize last voxel grid before height comppression 
class VoxelBackBone8xFuseMultiClass(VoxelBackBone8xFuse):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        last_pad = 0
        last_pad = self.model_cfg.get('last_pad', last_pad)
        self.conv_out_multi = spconv.SparseSequential(
            # [200, 176, 5] -> [200, 176, 2]
            spconv.SparseConv3d(67, 128, (3, 1, 1), stride=(2, 1, 1), padding=last_pad,
                                bias=False, indice_key='spconv_down2'),
            norm_fn(128),
            nn.ReLU(),
        )

    def forward(self, batch_dict):
        super().forward(batch_dict=batch_dict)
        # Implement class-based voxel colorization
        if self.training:
            objects_in_scene_and_gt_samples_mask = torch.max(batch_dict['combined_multiclass_mask'], batch_dict['gt_samples_2d_detections'])
        else:
            objects_in_scene_and_gt_samples_mask = batch_dict['combined_multiclass_mask']  
        
        image_feature_map_car = objects_in_scene_and_gt_samples_mask[...,0]
        image_feature_map_ped = objects_in_scene_and_gt_samples_mask[...,1]
        image_feature_map_cyc = objects_in_scene_and_gt_samples_mask[...,2]

        x_conv4 = batch_dict['multi_scale_3d_features']['x_conv4']
        image_voxel_features_car, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv4', 
                                                                            conv_output=x_conv4, 
                                                                            image_feature_map=image_feature_map_car, 
                                                                            raw_point_weights=None)
        image_voxel_features_ped, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv4', 
                                                                            conv_output=x_conv4, 
                                                                            image_feature_map=image_feature_map_ped, 
                                                                            raw_point_weights=None)
        image_voxel_features_cyc, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv4', 
                                                                            conv_output=x_conv4, 
                                                                            image_feature_map=image_feature_map_cyc, 
                                                                            raw_point_weights=None)

        voxel_multiclass_weights = torch.cat([image_voxel_features_car.view(-1, 1), image_voxel_features_ped.view(-1, 1), 
                                              image_voxel_features_cyc.view(-1, 1)], dim=1)
        # features = 64 + class_weights = 3
        x_conv_features_and_class_weights = torch.cat([x_conv4.features, voxel_multiclass_weights], dim=1)
        # create new sparse tensor
        x = spconv.SparseConvTensor(features=x_conv_features_and_class_weights, 
                                    indices=x_conv4.indices, 
                                    spatial_shape=x_conv4.spatial_shape, 
                                    batch_size=x_conv4.batch_size)
        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out_multi(x)
        batch_dict['encoded_spconv_tensor'] = out
        return batch_dict


# Colorize first voxel grid before sparse conv
class VoxelBackBone8xFuseEarlyMultiClass(VoxelBackBone8xFuse):
    def __init__(self, model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs):
        super().__init__(model_cfg, input_channels, grid_size, voxel_size, point_cloud_range,**kwargs)
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        block = post_act_block
        # features = 16 + class_weights = 3
        self.conv1 = spconv.SparseSequential(
            block(19, 16, 3, norm_fn=norm_fn, padding=1, indice_key='subm1'),
        )

    def forward(self, batch_dict):
        # Implement class-based voxel colorization
        if self.training:
            objects_in_scene_and_gt_samples_mask = torch.max(batch_dict['combined_multiclass_mask'], batch_dict['gt_samples_2d_detections'])
        else:
            objects_in_scene_and_gt_samples_mask = batch_dict['combined_multiclass_mask']  
        
        image_feature_map_car = objects_in_scene_and_gt_samples_mask[...,0]
        image_feature_map_ped = objects_in_scene_and_gt_samples_mask[...,1]
        image_feature_map_cyc = objects_in_scene_and_gt_samples_mask[...,2]

        # get image information
        image_feature_map = batch_dict['image_foreground_mask']

        voxel_features, voxel_coords = batch_dict['voxel_features'], batch_dict['voxel_coords']
        batch_size = batch_dict['batch_size']
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size
        )

        x = self.conv_input(input_sp_tensor)
        # use x_conv1 as conv_name as it has the same spatial dimensions are x
        image_voxel_features_car, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv1', 
                                                                            conv_output=x, 
                                                                            image_feature_map=image_feature_map_car, 
                                                                            raw_point_weights=None)
        image_voxel_features_ped, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv1', 
                                                                            conv_output=x, 
                                                                            image_feature_map=image_feature_map_ped, 
                                                                            raw_point_weights=None)
        image_voxel_features_cyc, voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                            conv_name='x_conv1', 
                                                                            conv_output=x, 
                                                                            image_feature_map=image_feature_map_cyc, 
                                                                            raw_point_weights=None)

        voxel_multiclass_weights = torch.cat([image_voxel_features_car.view(-1, 1), image_voxel_features_ped.view(-1, 1), 
                                              image_voxel_features_cyc.view(-1, 1)], dim=1)
        # features = 64 + class_weights = 3
        x_conv_features_and_class_weights = torch.cat([x.features, voxel_multiclass_weights], dim=1)
        # create new sparse tensor
        x = spconv.SparseConvTensor(features=x_conv_features_and_class_weights, 
                                    indices=x.indices, 
                                    spatial_shape=x.spatial_shape, 
                                    batch_size=x.batch_size)
        
        x_conv1 = self.conv1(x)
        conv_name = 'x_conv1'
        conv_output = locals()[conv_name]
        if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
            image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                        conv_name=conv_name, 
                                                                        conv_output=conv_output, 
                                                                        image_feature_map=image_feature_map, 
                                                                        raw_point_weights=None)
            conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)        
        
        x_conv2 = self.conv2(x_conv1)
        conv_name = 'x_conv2'
        conv_output = locals()[conv_name]
        if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
            image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                        conv_name=conv_name, 
                                                                        conv_output=conv_output, 
                                                                        image_feature_map=image_feature_map, 
                                                                        raw_point_weights=None)
            conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
        
        x_conv3 = self.conv3(x_conv2)
        conv_name = 'x_conv3'
        conv_output = locals()[conv_name]
        if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
            image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                        conv_name=conv_name, 
                                                                        conv_output=conv_output, 
                                                                        image_feature_map=image_feature_map, 
                                                                        raw_point_weights=None)
            conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
        
        x_conv4 = self.conv4(x_conv3)
        conv_name = 'x_conv4'
        conv_output = locals()[conv_name]
        if 'FUSE_LAYERS' in self.model_cfg and conv_name in self.model_cfg['FUSE_LAYERS']:
            image_voxel_features,voxel_centers_xyz  = self.get_conv_layer_image_based_feature(batch_dict, 
                                                                        conv_name=conv_name, 
                                                                        conv_output=conv_output, 
                                                                        image_feature_map=image_feature_map, 
                                                                        raw_point_weights=None)
            conv_output.features = self.fuse(voxel_feature=conv_output.features, image_foreground_weights=image_voxel_features, vox_conv_layer=conv_name)
            
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
        
