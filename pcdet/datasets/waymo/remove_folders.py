import pickle
import glob
from pathlib import Path
import shutil
from tqdm import tqdm

# remove_tag = 'sim_rain'
# root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
# processed_dir = root_path / 'waymo_processed_data_10'
# sim_rain_dirs = glob.glob(str(processed_dir / f'*{remove_tag}*'))
# for dir in sim_rain_dirs:
#     print('Removing dir', dir)
#     shutil.rmtree(dir)
#     b=1

root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
old_processed_dir = root_path / 'waymo_processed_data_10'
new_processed_dir = root_path / 'waymo_processed_train_all_val_da_10'
splits = ['train_all', 'val_da']

for split in splits:
    split_txt = root_path / 'ImageSets' / (split + '.txt')
    sequence_list = [x.strip().split('.')[0] for x in open(split_txt).readlines()]
    for seq in tqdm(sequence_list, desc=f'Split: {split}'):
        src_folder = old_processed_dir / seq
        dst_folder = new_processed_dir / seq
        shutil.move(src_folder, dst_folder)

