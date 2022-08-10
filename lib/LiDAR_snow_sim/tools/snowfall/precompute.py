__author__ = "Martin Hahner"
__contact__ = "martin.hahner@pm.me"
__license__ = "CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/)"

import copy
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from lib.LiDAR_snow_sim.tools.snowfall.simulation import augment
from third_party.OpenPCDet.pcdet.utils import calibration_kitti
from lib.LiDAR_snow_sim.tools.snowfall.sampling import compute_occupancy, snowfall_rate_to_rainfall_rate
#from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import time
import numpy as np

ROOT_PATH = (Path(__file__) / '../../../../..').resolve() #DepthContrast
DATA_PATH = ROOT_PATH /'data' / 'dense'
SPLIT_FOLDER =  DATA_PATH/ 'ImageSets' / 'train_clear_precompute'
LIDAR_FOLDER = DATA_PATH / 'lidar_hdl64_strongest'
SAVE_DIR_ROOT = ROOT_PATH / 'output' / 'snowfall_simulation' #for compute canada DATA_PATH / 'snowfall_simulation_FOV' #

SNOWFALL_RATES = [0.5, 0.5, 1.0, 2.0, 2.5, 1.5]  #[0.5, 1.0, 2.0, 2.5, 1.5]       # mm/h
TERMINAL_VELOCITIES = [2.0, 1.2, 1.6, 2.0, 1.6, 0.6] #[2.0, 1.6, 2.0, 1.6, 0.6]  # m/s


def split(a, n):
    k, m = divmod(len(a), n)
    return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))


def get_calib(sensor: str = 'hdl64'):
    calib_file = DATA_PATH / f'calib_{sensor}.txt'
    assert calib_file.exists(), f'{calib_file} not found'
    return calibration_kitti.Calibration(calib_file)


def get_fov_flag(pts_rect, img_shape, calib):

    pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
    val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
    val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
    val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
    pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

    return pts_valid_flag

def crop_pc(pc):
    point_cloud_range = np.array([0, -40, -3, 70.4, 40, 1], dtype=np.float32)
    upper_idx = np.sum((pc[:, 0:3] <= point_cloud_range[3:6]).astype(np.int32), 1) == 3
    lower_idx = np.sum((pc[:, 0:3] >= point_cloud_range[0:3]).astype(np.int32), 1) == 3

    new_pointidx = (upper_idx) & (lower_idx)
    pc = pc[new_pointidx, :]
    return pc

parser = argparse.ArgumentParser(description='Lidar snowfall sim')

parser.add_argument('--split', type=str, default='None', help='specify the config for training')
parser.add_argument('--snowfall_rate_index', default=-1, type=int, help='Index for snowfall_rate and terminal velocity')
parser.add_argument('--fov', action='store_true', default=False)


if __name__ == '__main__':
    
    args = parser.parse_args()
    SPLIT = SPLIT_FOLDER / args.split
    assert SPLIT.exists(), f'{SPLIT} does not exist'
    assert len(SNOWFALL_RATES) == len(TERMINAL_VELOCITIES), f'you need to provide an equal amount of ' \
                                                            f'snowfall_rates and terminal velocities'

    print(args) 
    rainfall_rates = []
    occupancy_ratios = []

    if args.snowfall_rate_index > -1 and args.snowfall_rate_index < len(TERMINAL_VELOCITIES):
        j=args.snowfall_rate_index
        rainfall_rates.append(snowfall_rate_to_rainfall_rate(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))
        occupancy_ratios.append(compute_occupancy(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))
    else:
        for j in range(len(SNOWFALL_RATES)):
            rainfall_rates.append(snowfall_rate_to_rainfall_rate(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))
            occupancy_ratios.append(compute_occupancy(SNOWFALL_RATES[j], TERMINAL_VELOCITIES[j]))

    combos = np.column_stack((rainfall_rates, occupancy_ratios))

    print('Combos rainfall_rates and occupancy ratios: ')
    print(combos)

    sample_id_list = sorted(['_'.join(x.strip().split(',')) for x in open(SPLIT).readlines()])
    total_skipped = 0

    for mode in ['gunn']: #'sekhon'

        p_bar = tqdm(sample_id_list, desc=mode)

        for sample_idx in p_bar:

            lidar_file = LIDAR_FOLDER / f'{sample_idx}.bin'

            points = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 5)
            calibration = get_calib()
            print(f'Processing sample: {sample_idx}')

            for combo in combos:

                rainfall_rate, occupancy_ratio = combo

                save_dir = SAVE_DIR_ROOT / mode / f'{LIDAR_FOLDER.name}_rainrate_{int(rainfall_rate)}'
                save_dir.mkdir(parents=True, exist_ok=True)

                save_path = save_dir / f'{sample_idx}.bin'

                if save_path.is_file():
                    continue

                pc = copy.deepcopy(points)

                #V.draw_scenes(points=pc, color_feature=3)
                #print(f'sample_idx: {sample_idx}, rr_ratio:{combo}')
                #print("pc: ", pc.shape)
                if args.fov:
                    pts_rectified = calibration.lidar_to_rect(pc[:, 0:3])
                    fov_flag = get_fov_flag(pts_rectified, (1024, 1920), calibration) #(1024, 1920)
                    pc = pc[fov_flag]
                    if pc.shape[0] < 3000:
                        print(f'Skipping {sample_idx} has less than 3000 points in FOV')
                        continue
                else:
                    pc = crop_pc(pc)
                #V.draw_scenes(points=pc, color_feature=3)
                #print("FOV_pc: ", pc.shape)

                snowflake_file_prefix = f'{mode}_{rainfall_rate}_{occupancy_ratio}'

                start = time.time()
                stats, aug_pc = augment(pc=pc, particle_file_prefix=snowflake_file_prefix,
                                        beam_divergence=float(np.degrees(3e-3)), root_path=DATA_PATH, only_camera_fov=args.fov)
                time_taken = time.time() - start
                
                #print("aug_pc: ", aug_pc.shape)
                #V.draw_scenes(points=aug_pc, color_feature=3)
                if stats == None:
                    print(f'ERROR processing sample: {sample_idx}')
                    total_skipped +=1
                    continue
                
                print(f'Time taken for {sample_idx}: {time_taken}, pc shape: {pc.shape[0]}, aug pc shape: {aug_pc.shape[0]}')
                if aug_pc.shape[0] < 3000:
                    print(f'LESS than 3000 pts! sample_idx: {sample_idx}, rr_ratio:{combo}, aug_pc:{aug_pc.shape}')

                aug_pc.astype(np.float32).tofile(save_path)
    
    print(f'Total skipped: {total_skipped}')
    print(f'Len samples: {len(sample_id_list)}')