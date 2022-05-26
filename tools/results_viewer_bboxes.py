import argparse
from pathlib import Path
import open3d
from visual_utils import open3d_vis_utils as V
import numpy as np
import pickle
import matplotlib as mpl
import matplotlib.cm as cm

train_clear_test_da = '/home/barza/OpenPCDet/output/waymo_models/pv_rcnn/default/eval/epoch_60/val_da/default/result.pkl'
train_sim_test_da = '/home/barza/OpenPCDet/output/waymo_models/pv_rcnn_sim_rain/sim_rain/eval/eval_with_train/epoch_60/val_da/result.pkl'
train_clear_test_train_da = '/home/barza/OpenPCDet/output/waymo_models/pv_rcnn/default/eval/epoch_60/train_da/default/result.pkl'
train_clear_val_clear = '/home/barza/OpenPCDet/output/waymo_models/pv_rcnn/default/eval/epoch_60/val_clear/default/result.pkl'
path = train_clear_test_train_da
data_path = Path('/media/barza/WD_BLACK/datasets/waymo/waymo_processed_data_10')
#split_file = Path('/home/barza/OpenPCDet/data/waymo/ImageSets/val_da.txt')
max_intensity_value = 1
class_labels = {'Vehicle':1, 'Pedestrian': 2, 'Cyclist': 3}

def get_colors(pc, color_feature=None):
    # create colormap
    if color_feature == 0:
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 1:

        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 2:
        feature = pc[:, 2]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 3:
        feature = pc[:, 3]
        min_value = 0
        max_value = max_intensity_value

    elif color_feature == 4:
        feature = pc[:, 4]
        min_value = np.min(feature)
        max_value = np.max(feature)

    else:
        feature = np.linalg.norm(pc[:, 0:3], axis=1)
        min_value = np.min(feature)
        max_value = np.max(feature)


    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)


    cmap = cm.jet  # sequential

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(feature)
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    colors[:, 3] = 0.5

    return colors[:, :3]


def main():
    with open(path, 'rb') as f:
        results = pickle.load(f)
    #seq_names_to_view = [x.strip().split('.')[0] for x in open(split_file).readlines()]
    results_by_seq = {}
    for data_dict in results:
        seq_name = data_dict['frame_id'][:-4]
        if seq_name not in results_by_seq:
            results_by_seq[seq_name] = []

        results_by_seq[seq_name].append(data_dict)


    for seq_name, data_dict_list in results_by_seq.items():
        gt_bbox_path = data_path / seq_name / (seq_name + '.pkl')
        with open(gt_bbox_path, 'rb') as f:
            gt_pkl = pickle.load(f)

        for i, data_dict in enumerate(data_dict_list):
            frame_index = data_dict['frame_id'][-3:]
            lidar_file = data_path / seq_name / (frame_index.zfill(4) + '.npy')
            points_intensity_elong = np.load(lidar_file)
            if 'sim_rain' not in seq_name:
                points_intensity_elong[:, 3] = np.tanh(points_intensity_elong[:, 3])

            idx_valid_classes = [j for j, name in enumerate(gt_pkl[i]['annos']['name']) if name in class_labels]

            gt_boxes = gt_pkl[i]['annos']['gt_boxes_lidar'][idx_valid_classes,:]
            pred_boxes = data_dict['boxes_lidar']
            pred_labels = np.array([class_labels[name] for name in data_dict['name']])

            difficulty_thresh = gt_pkl[i]['annos']['difficulty'][idx_valid_classes] < 2
            score_thresh = data_dict['score'] > 0.3
           
            V.draw_scenes(
                points=points_intensity_elong[:, :3], gt_boxes=gt_boxes, ref_boxes=pred_boxes[score_thresh],
                ref_scores=data_dict['score'][score_thresh], ref_labels=pred_labels[score_thresh], point_colors=get_colors(points_intensity_elong)
            )
            break



if __name__ == '__main__':
    main()

