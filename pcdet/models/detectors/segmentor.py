from .detector3d_template import Detector3DTemplate

class Segmentor(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        ret_dict = {
            'loss': 0,
            'batch_dict': batch_dict
        }
        return ret_dict, None, None, None
