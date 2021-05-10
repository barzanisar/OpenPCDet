import torch.nn as nn
from pcdet.models.backbones_3d.ife import isn
from pcdet.models.backbones_3d.ife.isn_loss import ISNLoss


class ImageSegmentation(nn.Module):
    def __init__(self, model_cfg, downsample_factor):
        """
            Initialize foreground classification network
            Args:
                model_cfg [EasyDict]: Foreground classification network config
                downsample_factor [int]: Foreground mask downsample factor
        """
        super().__init__()
        self.model_cfg = model_cfg
        self.downsample_factor = downsample_factor
        self.num_classes = 1
        # Create modules

        # segmentation network
        isn_cfg = model_cfg.ISN
        self.isn = isn.__all__[isn_cfg.NAME](
            num_classes=self.num_classes+1,
            backbone_name=isn_cfg.BACKBONE_NAME,
            **isn_cfg.ARGS
        )

        # loss
        self.isn_loss = ISNLoss(downsample_factor=self.downsample_factor,
                                **model_cfg.ISN_LOSS)

        self.forward_ret_dict = {}

    def get_output_feature_dim(self):
        pass

    def forward(self, batch_dict):
        """
        Predicts foreground class
        Args:
            batch_dict:
                images [torch.Tensor(N, 3, H_in, W_in)]: Input images
        Returns:
            batch_dict:
                image_features [torch.Tensor(N, C, H_out, W_out)]: Image features
        """
        # Pixel-wise semantic classification
        images = batch_dict["images"]
        isn_result = self.isn(images)
        image_features = isn_result["features"]
        segment_logits = isn_result["logits"]

        batch_dict["image_features"] = image_features
        batch_dict["segment_logits"] = segment_logits

        if self.training:
            self.forward_ret_dict["gt_boxes2d"] = batch_dict["gt_boxes2d"]
            self.forward_ret_dict["segment_logits"] = batch_dict["segment_logits"]
        return batch_dict

    def get_loss(self):
        loss, tb_dict = self.isn_loss(**self.forward_ret_dict)
        return loss, tb_dict