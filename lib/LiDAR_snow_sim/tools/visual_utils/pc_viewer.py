from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from pathlib import Path
import pickle

ROOT_PATH = (Path(__file__) / '../../../../..').resolve() #DepthContrast
DENSE_ROOT = ROOT_PATH / 'data' / 'dense'
alpha = 0.45
sensor_type = 'hdl64'
signal_type = 'strongest'

samples = ['2019-01-09_14-54-03,03700',
'2018-12-10_11-38-09,02600',
'2018-02-12_15-55-09,00100',
'2019-01-09_10-52-08,00100',
'2018-02-06_14-40-36,00100',
'2018-12-11_12-53-37,01540',
'2019-01-09_11-33-12,01200',
'2018-12-19_10-59-51,02200',
'2019-01-09_12-55-44,01000',
'2018-02-04_12-53-35,00000',
'2018-02-06_14-21-31,00000',
'2018-02-07_11-56-57,00560']
if __name__ == '__main__':

    info_path = DENSE_ROOT / '360deg_Infos' / 'dense_infos_test_clear_25.pkl'
    dense_infos = []
    dror_path_not_exists=0

    with open(info_path, 'rb') as i:
        infos = pickle.load(i)
        dense_infos.extend(infos)
    
    for info in dense_infos:
        sample_idx = info['point_cloud']['lidar_idx']
        
        dror_path = DENSE_ROOT / 'DROR_downloaded' / f'alpha_{alpha}' / \
                    'all' / sensor_type / signal_type / 'full' / f'{sample_idx}.pkl'

        if dror_path.exists():
            lidar_file = DENSE_ROOT / 'lidar_hdl64_strongest' / f'{sample_idx}.bin'
            pc = np.fromfile(lidar_file, dtype=np.float32).reshape((-1,5))


            with open(str(dror_path), 'rb') as f:
                snow_indices = pickle.load(f)

            keep_indices = np.ones(len(pc), dtype=bool)
            keep_indices[snow_indices] = False

            # Before DROR
            V.draw_scenes(points=pc, color_feature=3)

            # Apply DROR
            pc = pc[keep_indices]

            # After DROR
            V.draw_scenes(points=pc, color_feature=3)
        else:
            dror_path_not_exists += 1
            print(f'{sample_idx} snow indices do not exist')
    
    print(f'Not exists num: {dror_path_not_exists} / {len(dense_infos)}')