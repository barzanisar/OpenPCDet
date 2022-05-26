import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


root_path = Path('/home/barza/OpenPCDet/data/waymo')
split = 'train'
clear_weather_lidar_dir = root_path / 'waymo_processed_data'
noisy_weather_lidar_dir = root_path / 'noisy_processed_data'
aug_lidar_dir =  root_path / 'aug_processed_data'


def aug2():
    """
        Make an aug_velodyne folder with lidar samples taken at random from clear weather and foggy weather velodyne.
        Add half samples of train (/val)txt from clear weather velo and the rest of the half from foggy weather 
    """
    aug_lidar_dir.mkdir(parents=True, exist_ok=True)

    src_txt = root_path / 'ImageSets' / 'train.txt'
    seq_id_list = [x.strip() for x in open(src_txt).readlines()]
    clear_weather_len = int(0.5 * len(seq_id_list))
    for i in tqdm(range(clear_weather_len), desc="Loading clear lidar samples from train.txt"):
        seq_name = seq_id_list[i].split('.')[0]
        
        clear_weather_lidar_sample_path = clear_weather_lidar_dir / seq_name
        aug_weather_lidar_sample_path = aug_lidar_dir / seq_name
        aug_weather_lidar_sample_path.mkdir(parents=True, exist_ok=True)

        shutil.copytree(clear_weather_lidar_sample_path, aug_weather_lidar_sample_path, dirs_exist_ok=True)
    
    for i in tqdm(range(clear_weather_len, len(seq_id_list)), desc="Loading rainy lidar samples from train.txt"):
        seq_name = seq_id_list[i].split('.')[0]

        noisy_weather_lidar_sample_path = noisy_weather_lidar_dir / seq_name
        aug_weather_lidar_sample_path = aug_lidar_dir / seq_name
        aug_weather_lidar_sample_path.mkdir(parents=True, exist_ok=True)

        shutil.copytree(noisy_weather_lidar_sample_path, aug_weather_lidar_sample_path, dirs_exist_ok=True)

#aug2()

def addValSamples(mode = 'clear'):
    if mode == 'clear':
        src_dir = clear_weather_lidar_dir
    else:
        src_dir = noisy_weather_lidar_dir

    src_txt = root_path / 'ImageSets' / 'val.txt'
    seq_id_list = [x.strip() for x in open(src_txt).readlines()]

    for i in tqdm(range(len(seq_id_list)), desc=f"Loading {mode} validation lidar samples from train.txt"):
        seq_name = seq_id_list[i].split('.')[0]

        src_lidar_sample_path = src_dir / seq_name
        aug_weather_lidar_sample_path = aug_lidar_dir / seq_name
        aug_weather_lidar_sample_path.mkdir(parents=True, exist_ok=True)

        shutil.copytree(src_lidar_sample_path, aug_weather_lidar_sample_path, dirs_exist_ok=True)

addValSamples('clear')