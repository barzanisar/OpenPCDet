import numpy as np
from pathlib import Path
import shutil
from tqdm import tqdm


root_path = Path('/home/barza/OpenPCDet/data/kitti')
split = 'train'
aug_velodyne_dirname = 'aug_velodyne'
root_split_path =  root_path / 'training'
clear_weather_lidar_dir = root_split_path / 'velodyne'
aug_lidar_dir = root_split_path / aug_velodyne_dirname

def aug1():
    """
    Make an aug_velodyne folder with 70 percent of lidar samples in train.txt 
    taken from clear weather velodyne and 30% from foggy velodyne. 
    TODO: Add samples from clear velodyne from Validation.txt
    """
    aug_lidar_dir.mkdir(parents=True, exist_ok=True)
    
    foggy_velodyne_dirname = 'velodyne_CVL_beta_0.10'
    clear_weather_percentage = 0.7
    
    foggy_velodyne_dir = root_split_path / foggy_velodyne_dirname
   
    
    src_txt = root_path / 'ImageSets' / (split + '.txt')
    sample_id_list = [x.strip() for x in open(src_txt).readlines()] #if split_dir.exists() else None
    clear_weather_len = int(clear_weather_percentage*len(sample_id_list))
    
    
    for i in tqdm(range(clear_weather_len), desc="Loading clear lidar samples"):
        clear_weather_lidar_sample_path = clear_weather_lidar_dir / ('%s.bin' % sample_id_list[i])
        aug_weather_path = aug_lidar_dir / ('%s.bin' % sample_id_list[i])
        shutil.copyfile(clear_weather_lidar_sample_path, aug_weather_path)
    
    for i in tqdm(range(clear_weather_len, len(sample_id_list)), desc="Loading foggy lidar samples"):
        foggy_sample_path = foggy_velodyne_dir / ('%s.bin' % sample_id_list[i])
        aug_weather_path = aug_lidar_dir / ('%s.bin' % sample_id_list[i])
        shutil.copyfile(foggy_sample_path, aug_weather_path)

def aug2():
    """
        Make an aug_velodyne folder with lidar samples taken at random from clear weather and foggy weather velodyne.
        Add half samples of train (/val)txt from clear weather velo and the rest of the half from foggy weather 
    """
    aug_lidar_dir.mkdir(parents=True, exist_ok=True)
    
    alphas = ['0.005', '0.010', '0.030', '0.060', '0.100', '0.200']
    

    src_txt = root_path / 'ImageSets' / 'train.txt'
    sample_id_list = [x.strip() for x in open(src_txt).readlines()]
    clear_weather_len = int(0.5 * len(sample_id_list))
    for i in tqdm(range(clear_weather_len), desc="Loading clear lidar samples from train.txt"):
        clear_weather_lidar_sample_path = clear_weather_lidar_dir / ('%s.bin' % sample_id_list[i])
        aug_weather_path = aug_lidar_dir / ('%s.bin' % sample_id_list[i])
        shutil.copyfile(clear_weather_lidar_sample_path, aug_weather_path)
    
    for i in tqdm(range(clear_weather_len, len(sample_id_list)), desc="Loading foggy lidar samples from train.txt"):
        alpha = np.random.choice(alphas)
        foggy_velodyne_dir = root_split_path / f'velodyne_CVL_beta_{alpha}'
        foggy_sample_path = foggy_velodyne_dir / ('%s.bin' % sample_id_list[i])
        aug_weather_path = aug_lidar_dir / ('%s.bin' % sample_id_list[i])
        shutil.copyfile(foggy_sample_path, aug_weather_path)

    src_txt = root_path / 'ImageSets' / 'val.txt'
    sample_id_list = [x.strip() for x in open(src_txt).readlines()]

    for i in tqdm(range(len(sample_id_list)), desc="Loading foggy lidar samples from val.txt"):
        alpha = np.random.choice(alphas)
        foggy_velodyne_dir = root_split_path / f'velodyne_CVL_beta_{alpha}'
        foggy_sample_path = foggy_velodyne_dir / ('%s.bin' % sample_id_list[i])
        aug_weather_path = aug_lidar_dir / ('%s.bin' % sample_id_list[i])
        shutil.copyfile(foggy_sample_path, aug_weather_path)

    
    
    

aug2()
    
    