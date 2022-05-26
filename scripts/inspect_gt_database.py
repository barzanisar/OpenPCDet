import glob
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

root_path = Path('/media/barza/WD_BLACK/datasets/waymo/pagal')

db_info_path = root_path / 'pcdet_waymo_dbinfos_train_sim_rain_sampled_10.pkl'
with open(db_info_path, 'rb') as f:
    infos = pickle.load(f)

for k in range(0, len(infos['Vehicle'])):
    info = infos['Vehicle'][k]
    filepath = root_path / info['path']
    obj_points = np.fromfile(str(filepath), dtype=np.float32).reshape(
        [-1, 5])

    obj_points[:, :3] += info['box3d_lidar'][:3]

    # pc_info = info['point_cloud']
    # sequence_name = pc_info['lidar_sequence']
    # sample_idx = pc_info['sample_idx']
    # points = self.get_lidar(sequence_name, sample_idx)
    #
    # annos = info['annos']
    # names = annos['name']
    # difficulty = annos['difficulty']
    # gt_boxes = annos['gt_boxes_lidar']
    # num_obj = gt_boxes.shape[0]
    #
    # for i in range(num_obj):
    #     filename = '%s_%04d_%s_%d.bin' % (sequence_name, sample_idx, names[i], i)
    #     filepath = database_save_path / filename
