import pickle
import numpy as np

np.random.seed(100)
root_path = "/home/barza/OpenPCDet/data/dense"
split_base_mode = ['train', '60'] #['train', '60'] #['test', '25'] #['val', '15']
percent_list = [1] #[5, 10, 20, 30, 40, 50, 100]  #[100]
weather_splits = [f'{split_base_mode[0]}_light_fog_FOV3000_{split_base_mode[1]}.pkl', 
f'{split_base_mode[0]}_dense_fog_FOV3000_{split_base_mode[1]}.pkl', 
f'{split_base_mode[0]}_snow_FOV3000_{split_base_mode[1]}.pkl']
clear_infos =[]

# Load all of clear weather split:
info_path = root_path + f"/dense_infos_{split_base_mode[0]}_clear_FOV3000_{split_base_mode[1]}.pkl"
with open(info_path, 'rb') as f:
    infos = pickle.load(f)
    clear_infos += infos
    

for percent in percent_list:
    all_infos = []
    all_infos += clear_infos
    print(f"All infos: {len(all_infos)}")
    all_infos_path = root_path + f"/dense_infos_{split_base_mode[0]}_all_FOV3000_{split_base_mode[1]}"
    # Load a "percent" of other weather splits:
    for split in weather_splits:
        info_path = root_path + "/dense_infos_" + split
        with open(info_path, 'rb') as f:
            infos = pickle.load(f)
            if percent < 100:
                shuffled_idx = np.arange(len(infos))
                np.random.shuffle(shuffled_idx)
                last_idx = int((percent/100) * len(infos))
                selected_idx = shuffled_idx[:last_idx]
                infos = (np.array(infos)[selected_idx]).tolist()
            print(f"Adding {split} infos: {len(infos)}")
            all_infos += infos
            print(f"All infos: {len(all_infos)}")

    if percent < 100:
        all_infos_path += '_' + str(percent)
    with open(all_infos_path + '.pkl', 'wb') as f:
        pickle.dump(all_infos, f)