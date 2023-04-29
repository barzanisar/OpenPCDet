import pickle
import numpy as np

######################## create kitti_infos_train_95_0.pkl ###################
# path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_train_5_0.pkl'
# with open(path, 'rb') as f:
#     infos_train_5 = pickle.load(f)

# lidar_idx_5=[]
# for info in infos_train_5:
#     lidar_idx_5.append(info['point_cloud']['lidar_idx'])

# path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_val.pkl'
# with open(path, 'rb') as f:
#     infos_val = pickle.load(f)

# # for info in infos_val:
# #     if info['point_cloud']['lidar_idx'] in lidar_idx_5:
# #         print('5_0 has val!')

# path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_train.pkl'
# with open(path, 'rb') as f:
#     infos_train = pickle.load(f)

# infos_train_95 = [ info for info in infos_train if info['point_cloud']['lidar_idx'] not in lidar_idx_5]

# print(len(lidar_idx_5))
# print(len(infos_train_95))

# path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_train_95_0.pkl'
# with open(path, 'wb') as f:
#     pickle.dump(infos_train_95, f)



# path = '/home/barza/OpenPCDet/data/kitti/kitti_dbinfos_train.pkl'
# with open(path, 'rb') as f:
#     infos_dbinfos_train = pickle.load(f)

# object_infos = infos_dbinfos_train['Pedestrian'] + infos_dbinfos_train['Car'] + infos_dbinfos_train['Cyclist'] 


################ Convert car, ped, cyc dbinfos into Object dbinfos ##################3
# path = '/home/barza/OpenPCDet/data/kitti/kitti_dbinfos_train_95_2.pkl'
# with open(path, 'rb') as f:
#     infos_dbinfos_train = pickle.load(f)

# for cls in ['Car', 'Cyclist', 'Pedestrian']:
#     cls_infos = infos_dbinfos_train[cls]
#     for info in cls_infos:
#         info['name'] = 'Object'
# object_dict = {'Object': infos_dbinfos_train['Pedestrian'] + infos_dbinfos_train['Car'] + infos_dbinfos_train['Cyclist']}

# path = '/home/barza/OpenPCDet/data/kitti/kitti_dbinfos_train_95_2_object.pkl'
# with open(path, 'wb') as f:
#     pickle.dump(object_dict,f)

################ Convert car, ped, cyc in train infos to object classes ##################
path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_train_5_1.pkl'
with open(path, 'rb') as f:
    infos_train = pickle.load(f)

for info in infos_train:
    name_list = []
    for name in info['annos']['name']:
        if name in ['Car', 'Pedestrian', 'Cyclist']:
            name_list.append('Object')
        else:
            name_list.append(name)
    info['annos']['name'] = np.array(name_list)


path = '/home/barza/OpenPCDet/data/kitti/kitti_infos_train_5_1_object.pkl'
with open(path, 'wb') as f:
    pickle.dump(infos_train,f)


# path2 = '/home/barza/OpenPCDet/data/kitti/depth_contrast_infos/kitti_infos_val.pkl' #'/home/barza/OpenPCDet/data/kitti/kitti_infos_val.pkl'
# with open(path2, 'rb') as f:
#     infos2 = pickle.load(f)
# classes_num = {}
# for i in range(len(infos1)):
#     for name in infos1[i]['annos']['name']:
#         if name not in classes_num:
#             classes_num[name] = 0
#         else:
#             classes_num[name] += 1

# print(classes_num)
        