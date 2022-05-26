import pickle
import glob
from pathlib import Path
import shutil
from tqdm import tqdm

sample_interval = 10
root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
old_proccessed_dir =  root_path / 'waymo_processed_data_v_1_2_0'
new_processed_dir = root_path / 'waymo_processed_data_10'
#new_processed_dir.mkdir(parents=True, exist_ok=True)

# seq_info_list = glob.glob(str(old_proccessed_dir / '*/*.pkl'))
file = root_path / 'seq_info_list.pkl'
# with open(file, 'wb') as f:
#     pickle.dump(seq_info_list, f)

with open(file, 'rb') as f:
    seq_info_list = pickle.load(f)

for seq_info_path in tqdm(seq_info_list):
    p = Path(seq_info_path)
    lidar_npy_paths = sorted(glob.glob(str(p.parent / '*.npy')))

    with open(seq_info_path, 'rb') as f:
        seq_info = pickle.load(f)

    seq_name = p.parent.name
    new_seq_dir = new_processed_dir / seq_name
    new_seq_dir.mkdir(parents=True, exist_ok=True)
    new_seq_pkl_path = new_seq_dir / (seq_name + '.pkl')
    new_seq = []
    for i in range(0, len(seq_info), sample_interval):
        new_seq.append(seq_info[i])
        shutil.copyfile(lidar_npy_paths[i], str(new_seq_dir/Path(lidar_npy_paths[i]).name))

    with open(new_seq_pkl_path, 'wb') as f:
        pickle.dump(new_seq, f)


