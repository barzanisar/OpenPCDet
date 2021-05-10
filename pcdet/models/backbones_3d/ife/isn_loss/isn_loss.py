import torch
import torch.nn as nn
import kornia

from .balancer import Balancer
from pcdet.utils import loss_utils

import matplotlib.pyplot as plt

class ISNLoss(nn.Module):

    def __init__(self,
                 weight,
                 alpha,
                 gamma,
                 fg_weight,
                 bg_weight,
                 downsample_factor=1):
        """
        Initializes ISNLoss module
        Args:
            weight [float]: Loss function weight
            alpha [float]: Alpha value for Focal Loss
            gamma [float]: Gamma value for Focal Loss
            fg_weight [float]: Foreground loss weight
            bg_weight [float]: Background loss weight
            downsample_factor [int]: Image segmentation downsample factor
        """
        super().__init__()
        self.downsample_factor = downsample_factor
        self.device = torch.cuda.current_device()
        self.balancer = Balancer(downsample_factor=downsample_factor,
                                 fg_weight=fg_weight,
                                 bg_weight=bg_weight)

        # Set loss function
        self.alpha = alpha
        self.gamma = gamma
        self.loss_func = kornia.losses.FocalLoss(alpha=self.alpha, gamma=self.gamma, reduction="none")
        self.weight = weight

    def forward(self, segment_logits, gt_boxes2d):
        """
        Gets ISN loss
        Args:
            segment_logits: torch.Tensor(B, C+1, H, W)]: Predicted segmentation logits
            gt_boxes2d [torch.Tensor (B, N, 4)]: 2D box labels for foreground/background balancing
        Returns:
            loss [torch.Tensor(1)]: Segmentation classification network loss
            tb_dict [dict[float]]: All losses to log in tensorboard
        """
        tb_dict = {}

        # Generate foreground/background masks as targets
        # Compute masks
        fg_mask = loss_utils.compute_fg_mask(gt_boxes2d=gt_boxes2d,
                                             shape=(segment_logits.shape[0], segment_logits.shape[2], segment_logits.shape[3]),
                                             downsample_factor=self.downsample_factor,
                                             device=segment_logits.device)

        segmentation_targets = torch.zeros(fg_mask.shape, dtype=torch.int64, device=fg_mask.device)
        segmentation_targets[fg_mask.long() == True] = 1

        # Compute loss
        loss = self.loss_func(segment_logits, segmentation_targets)

        # Compute foreground/background balancing
        loss, tb_dict = self.balancer(loss=loss, gt_boxes2d=gt_boxes2d)

        # Final loss
        loss *= self.weight
        tb_dict.update({"isn_loss": loss.item()})

        return loss, tb_dict
