import glob
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path
import shutil

root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
rain_proc_dir = root_path / 'waymo_processed_train_sim_rain_val_da_10'
clear_proc_dir = root_path / 'waymo_processed_data_10'
seq_list_txt = root_path / 'ImageSets' / 'train_sim_rain.txt'
train_sim_rain_labels_dir = root_path / 'train_sim_rain_labels'
#train_sim_rain_labels_dir.mkdir(parents=True, exist_ok=True)
train_sim_rain_dir = root_path / 'waymo_processed_train_sim_rain'
#train_sim_rain_dir.mkdir(parents=True, exist_ok=True)


seq_list = [x.strip().split('.')[0] for x in open(seq_list_txt).readlines()]
# for seq in seq_list:
#     if 'sim_rain' not in seq:
#         continue
#
#     seq_dir = train_sim_rain_dir / seq
#
#     npy_files = glob.glob(str(seq_dir / '*.npy'))
#
#     for file in tqdm(npy_files):
#         rain_pc = np.load(file)
#         if rain_pc.max() == np.nan:
#             b=1
#         else:
#             print('max: ', rain_pc.max())



for seq in seq_list:
    if 'sim_rain' not in seq:
        continue

    seq_dir = rain_proc_dir / seq
    shutil.rmtree(seq_dir)

    # npy_files = glob.glob(str(seq_dir / '*.npy'))
    # pkl_file = seq_dir / (seq + '.pkl')
    #
    #
    # new_seqlabels_dir = train_sim_rain_labels_dir / seq
    # new_seqlabels_dir.mkdir(parents=True, exist_ok=True)
    #
    # new_seq_dir = train_sim_rain_dir / seq
    # new_seq_dir.mkdir(parents=True, exist_ok=True)
    # new_pkl_file = new_seq_dir / (seq + '.pkl')
    # shutil.copyfile(pkl_file, new_pkl_file)
    # for file in tqdm(npy_files):
    #     rain_pc = np.load(file).astype(np.float32)
    #     labels = rain_pc[:,-1].astype(np.int8)
    #     rain_pc_no_labels = rain_pc[:,:-1]
    #
    #     np.save(new_seqlabels_dir / Path(file).stem, labels)
    #     np.save(new_seq_dir / Path(file).stem, rain_pc_no_labels)
    #     b=1

