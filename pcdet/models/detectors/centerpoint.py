from .detector3d_template import Detector3DTemplate


class CenterPoint(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
                'batch_dict': batch_dict
            }
            return ret_dict, tb_dict, disp_dict, None
        else:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
                'batch_dict': batch_dict
            }
            pred_dicts, recall_dicts = self.post_processing(batch_dict)
            return ret_dict, tb_dict, pred_dicts, recall_dicts
        
    def get_training_loss(self):
        disp_dict = {}

        loss = 0
        tb_dict=None
        if self.dense_head is not None:
            loss_rpn, tb_dict = self.dense_head.get_loss()
            loss += loss_rpn
        
            tb_dict = {
                'loss_rpn': loss.item(),
                **tb_dict
            }
        return loss, tb_dict, disp_dict

    def post_processing(self, batch_dict):
        if 'POST_PROCESSING' not in self.model_cfg:
            return None, None
        post_process_cfg = self.model_cfg.POST_PROCESSING
        batch_size = batch_dict['batch_size']
        final_pred_dict = batch_dict['final_box_dicts']
        recall_dict = {}
        for index in range(batch_size):
            pred_boxes = final_pred_dict[index]['pred_boxes']

            recall_dict = self.generate_recall_record(
                box_preds=pred_boxes,
                recall_dict=recall_dict, batch_index=index, data_dict=batch_dict,
                thresh_list=post_process_cfg.RECALL_THRESH_LIST
            )

        return final_pred_dict, recall_dict
