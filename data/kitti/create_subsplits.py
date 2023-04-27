#Read split files and append frame ids to a list
#shuffle indices of len of list
#extract 60% of the list to train, 20% to val, 20% to test

from pathlib import Path
import numpy as np
from tqdm import tqdm
import random
import pickle
random.seed(100)

# ROOT_PATH = Path('/home/barza/OpenPCDet/data/kitti')
# pkl_path = ROOT_PATH / 'kitti_infos_train.pkl'

# split_percentage = 5 #10, 20

# with open(pkl_path, 'rb') as f:
#     infos = pickle.load(f)

# num_frames = len(infos)
# shuffled_indices = list(range(num_frames))
# random.shuffle(shuffled_indices)

# num_idx_select = int(split_percentage*num_frames/100)
# sub_idx = shuffled_indices[:num_idx_select]

# new_infos = infos[sub_idx]

# new_pkl_path  = ROOT_PATH / f'kitti_infos_train_{split_percentage}.pkl'
# with open(new_pkl_path, 'wb') as f:
#     pickle.dump(new_infos, f)

def create_subtxt_from_txt():
    ROOT_PATH = Path('/home/barza/OpenPCDet/data/kitti')
    splits_path = ROOT_PATH / 'ImageSets' / 'train.txt'

    split_percentages = [5] #10, 20
    split_ids = [x.strip() for x in open(splits_path).readlines()] 
    print(f'Read {splits_path} with {len(split_ids)} samples')

    num_frames = len(split_ids)
    shuffled_indices = list(range(num_frames))
    random.shuffle(shuffled_indices)

    for p in split_percentages:
        num_idx_select = int(p*num_frames/100)
        idx_selected = shuffled_indices[:num_idx_select]
        new_split_path = ROOT_PATH / 'ImageSets' / f'train_{p}.txt'
        print(f'Writing in {new_split_path} {len(idx_selected)} samples')
        with open(new_split_path, 'w') as f:
            for i, idx in enumerate(idx_selected):
                f.write(split_ids[idx])
                if i != len(idx_selected)-1:
                    f.write('\n')


def create_txt_from_pkl():
    ROOT_PATH = Path('/home/barza/OpenPCDet/data/kitti')
    pkl_path = ROOT_PATH / 'kitti_infos_train_5_2.pkl'
    with open(pkl_path, 'rb') as f:
        infos = pickle.load(f)

    new_txt_path = ROOT_PATH / 'ImageSets' / 'train_5_2.txt'

    ids = [info['point_cloud']['lidar_idx'] for info in infos]
    with open(new_txt_path, 'w') as f:
        for i, id in enumerate(ids):
            f.write(id)
            if i != len(ids)-1:
                f.write('\n')



create_txt_from_pkl()
