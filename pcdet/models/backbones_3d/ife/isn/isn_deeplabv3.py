import torchvision

from .isn_template import ISNTemplate


class ISNDeepLabV3(ISNTemplate):

    def __init__(self, backbone_name, **kwargs):
        """
        Initializes ISNDeepLabV3 model
        Args:
            backbone_name [str]: ResNet Backbone Name
        """
        if backbone_name == "ResNet50":
            constructor = torchvision.models.segmentation.deeplabv3_resnet50
        elif backbone_name == "ResNet101":
            constructor = torchvision.models.segmentation.deeplabv3_resnet101
        else:
            raise NotImplementedError

        super().__init__(constructor=constructor, **kwargs)