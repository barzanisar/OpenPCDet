import matplotlib.pyplot as plt
import pickle
plt.style.use('seaborn-whitegrid')
import numpy as np

def genHistogram():
    fig = plt.figure('hist', figsize=(15,5))

    with open('/home/barza/OpenPCDet/tools/hist_range_intensity_clear.pkl', 'rb') as f:
        clear_data = pickle.load(f)

    with open('/home/barza/OpenPCDet/tools/hist_range_intensity_fog.pkl', 'rb') as f:
        fog_data = pickle.load(f)


    fig.add_subplot(1,2,1)
    n, bins, patches = plt.hist(x=clear_data['range'],
                                bins=np.linspace(start=0, stop=min(clear_data['range'].max(), 100), num=20 + 1, endpoint=True), rwidth=0.85, label='Clear')
    n, bins, patches = plt.hist(x=fog_data['range'],
                                bins=np.linspace(start=0, stop=min(fog_data['range'].max(), 100), num=20 + 1,
                                                 endpoint=True), rwidth=0.85, label='Fog')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Range (m)')
    plt.ylabel('Number of Points')
    plt.title(f'Visibility of Point Cloud')
    plt.legend()
    # maxfreq = n.max()
    # # Set a clean upper y-axis limit.
    # plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


    fig.add_subplot(1, 2, 2)
    n, bins, patches = plt.hist(x=clear_data['intensity'],
                                bins=np.linspace(start=0, stop=255, num=20 + 1, endpoint=True),
                                rwidth=0.85, label='Clear')
    n, bins, patches = plt.hist(x=fog_data['intensity'],
                                bins=np.linspace(start=0, stop=255, num=20 + 1, endpoint=True),
                                rwidth=0.85,  label='Fog')

    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Number of Points')
    plt.title(f'Intensities of Point Cloud')
    plt.legend()
    plt.savefig('hist_Rr_trunc.png')
    plt.show()
    plt.pause(0.001)

genHistogram()
b=1
# #attenuation coefficient
# alphas = [0, 0.005, 0.01, 0.02, 0.03, 0.06, 0.1, 0.15, 0.2]
# car_3d_AP = [83.69, 65.34, 64.96, 63.52, 60.92, 60.38, 60.66, 60.59, 60.75]
# plt.scatter(alphas, car_3d_AP, marker='x',color='b',linewidths=2)
# plt.plot(alphas, car_3d_AP)
# plt.ylabel('CAR 3D AP')
# plt.xlabel('Attenuation coeff')
# plt.title('PV-RCNN (trained on clear KITTI) tested on foggy KITTI (FOV only)')
# plt.savefig('clear_pvrcnn_foggy_kitti.png')
# #plt.show()
#

# x_pos = np.arange(0, 100, 10)
# percent_clear = 100 - np.array([90.6, 77.8, 68.7, 61.5, 56, 51.5, 48, 44.9, 42.3, 40.1])
# percent_rain = 100 - np.array([86.8, 69.4, 60.4, 54.3, 49.8, 45.9, 43.1, 40.4, 38.3, 36.5])
# percent_sim_rain = 100 - np.array([69.1, 52.7, 45.8, 41.3, 38.1, 35.5, 33.4, 31.6, 33.0, 28.6])
# plt.figure(figsize=(5,5))
# #plt.bar(x_pos, percent_clear, align='center', alpha=0.5, width = 2, label='Clear')
# plt.bar(x_pos, percent_sim_rain, align='center', alpha=0.5, width = 2, label='Sim Rain')
# plt.bar(x_pos, percent_rain, align='center', alpha=0.5, width = 2, label='Real Rain')
# plt.xticks(x_pos, x_pos)
# plt.title('')
# plt.ylabel('% of gt boxes')
# plt.xlabel('Max number of points in gt boxes')
# plt.legend()
# plt.show()
#b=1

range_levels = ['Overall', '0-30 m', '30-50 m', '50-1000 m']
range_level_id = 0
min_gt_pts_list = range(10, 100, 10)



l1_results_clear = []
l2_results_clear = []
l1_results=[]
l2_results=[]
for min_gt_pts in min_gt_pts_list:

    #  Collect results for Train on sim, Test on val_sim
    eval_across_range_path = f'/home/barza/OpenPCDet/output/waymo_models/pv_rcnn_sim_rain/default/eval/epoch_60/val_clear_sim_rain/default/eval_across_range/pv_rcnn_sim_rain_max_{min_gt_pts}_eval.pkl'
    with open(eval_across_range_path, 'rb') as f:
        results = pickle.load(f)
    
    print('Min num points: ', min_gt_pts, 'VEHICLE_LEVEL_1/AP: ' , results['distance'][range_level_id]['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'])
    l1_results.append(100*results['distance'][range_level_id]['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'])
    l2_results.append(100*results['distance'][range_level_id]['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'])

    # Collect results for Train on sim, Test on val_da
    clear_clear_eval_across_range_path = f'/home/barza/OpenPCDet/output/waymo_models/pv_rcnn_sim_rain/default/eval/epoch_60/all_da/default/eval_across_range/pv_rcnn_sim_rain_max_{min_gt_pts}_eval.pkl'
    with open(clear_clear_eval_across_range_path, 'rb') as f:
        results_clear = pickle.load(f)

    l1_results_clear.append(100*results_clear['distance'][range_level_id]['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_1/AP'])
    l2_results_clear.append(100*results_clear['distance'][range_level_id]['OBJECT_TYPE_TYPE_VEHICLE_LEVEL_2/AP'])

plt.plot(min_gt_pts_list, l1_results,  label= 'Test on Sim Rain')
plt.plot(min_gt_pts_list, l1_results_clear, label= 'Test on Real Rain')
#plt.plot(range(0,60,10), l2_results, label= 'Vehicle level 2 AP')
plt.ylabel('Vehicle AP')
plt.xlabel('Max number of points in gt boxes')
plt.legend()
plt.title(f'Train on Sim, Range: {range_levels[range_level_id]}')
plt.show()

