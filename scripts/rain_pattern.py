import matplotlib.pyplot as plt
import pickle
plt.style.use('seaborn-whitegrid')
import numpy as np
import glob
from tqdm import tqdm
from pathlib import Path
import shutil
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import torch
from pcdet.datasets.simulator.atmos_models import LISA
import open3d
from tools.visual_utils import open3d_vis_utils as V

classnames = ['Vehicle']
root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
processed_dir = root_path / 'waymo_processed_data_10'
imgsets_dir = root_path / 'ImageSets'
frame_sampling_interval = 100
seq_sampling_interval = 1
range_bin_step = 1
range_bins = np.arange(start=0, stop=80 + range_bin_step + 1, step=range_bin_step)

def readsplit(split, sim_rain=False):
    seq_list = []
    seq_list += [x.strip().split('.')[0] for x in open(imgsets_dir/(split + '.txt')).readlines()]

    new_seq_list = []
    if sim_rain:
            new_seq_list = [seq for seq in seq_list if 'sim_rain' in seq]
            return new_seq_list

    return seq_list

def get_lidar(sequence_name, sample_idx):
    lidar_file = processed_dir / sequence_name / ('%04d.npy' % sample_idx)
    point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]

    points_all, NLZ_flag = point_features[:, 0:5], point_features[:, 5]
    points_all = points_all[NLZ_flag == -1]
    if 'sim_rain' not in sequence_name:
        points_all[:, 3] = np.tanh(points_all[:, 3])
    return points_all

def range_wise_summary_pc(pc):
    ranges = np.linalg.norm(pc[:, 0:3], axis=1)
    intensity = pc[:, 3]

    bin_indices = np.digitize(ranges, range_bins)
    num_bins = range_bins.shape[0]-1
    means_intensity = np.zeros(num_bins)
    std_intensity = np.zeros(num_bins)
    range_wise_num_pts = np.zeros(num_bins)
    for bin_i in range(1, num_bins+1):
        idx_for_ranges_in_bin_i = np.argwhere(bin_indices == bin_i)
        means_intensity[bin_i-1] = intensity[idx_for_ranges_in_bin_i].mean()
        std_intensity[bin_i-1] = intensity[idx_for_ranges_in_bin_i].std()
        range_wise_num_pts[bin_i-1] = intensity[idx_for_ranges_in_bin_i].shape[0]

    return range_bins, means_intensity, std_intensity, range_wise_num_pts
def get_colors(pc, color_feature=5):
    # create colormap
    if color_feature < 3:
        feature = pc[:, color_feature]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 3:
        feature = pc[:, 3]
        min_value = 0
        max_value = 1

    elif color_feature == 4:
        feature = np.linalg.norm(pc[:, 0:3], axis=1)
        min_value = np.min(feature)
        max_value = np.max(feature)

    else:
        feature = pc[:, 4]
        colors = np.zeros((feature.shape[0], 3))

        colors[feature == 0, 0] = 1  # lost red
        colors[feature == 1, 1] = 1  # scattered green
        colors[feature == 2, 2] = 1  # original but noisy blue
        print(f'lost % #: {100 * np.count_nonzero(colors[:, 0]) / pc.shape[0]}')
        print(f'scattered % #: {100 * np.count_nonzero(colors[:, 1]) / pc.shape[0]}')
        print(f'attenuated % #: {100 * np.count_nonzero(colors[:, 2]) / pc.shape[0]}')


    if color_feature < 5:
        norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)


        cmap = cm.jet  # sequential

        m = cm.ScalarMappable(norm=norm, cmap=cmap)

        colors = m.to_rgba(feature)
        colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
        colors[:, 3] = 0.5



    return colors[:, :3]
def sim_rain_save_data(split, Rr, extra_tag='',sim_rain=True):
    simulator = LISA(rmax=150)
    seq_list = readsplit(split, sim_rain)

    data_zero_pts = []
    data_some_pts = []
    range_wise_summary_zero_pts = []
    range_wise_summary_some_pts = []
    range_wise_summary_all_pc = {'means_intensity' : [],
                                 'std_intensity': [],
                                 'num_pts': []}

    for seq_cnt, seq in tqdm(enumerate(seq_list)):
        if seq_cnt % seq_sampling_interval != 0:
            continue
        seq = seq.replace('_sim_rain', '') # get clear weather data
        seq_dir = processed_dir / seq
        seq_pkl = seq_dir / (seq + '.pkl')
        with open(seq_pkl, 'rb') as f:
            seq_info = pickle.load(f)

        for cnt, info in enumerate(seq_info):
            if cnt % frame_sampling_interval != 0:
                continue
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = get_lidar(sequence_name, sample_idx)

            if 'LISA' in extra_tag:
                points = simulator.augment_mc(points[:, :4], Rr)  # augment_mc(points[:,:4], Rr) msu_rain
            elif 'prob' in extra_tag:
                points = simulator.prob_rain(points[:, :4], Rr)
            else:
                points = simulator.msu_rain(points[:, :4], Rr)
            # lost_points = np.where(noisy_pc[:, 4] == 0)
            # #noisy_pc[lost_points, :4] = points[lost_points, :4]
            # V.draw_scenes(points=noisy_pc[:, :3], point_colors=get_colors(noisy_pc))

            range_bins, means_intensity, std_intensity, num_pts = range_wise_summary_pc(points)
            range_wise_summary_all_pc['means_intensity'].append(means_intensity)
            range_wise_summary_all_pc['std_intensity'].append(std_intensity)
            range_wise_summary_all_pc['num_pts'].append(num_pts)

            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            obj_dict = {}
            for i in range(num_obj):
                if names[i] in classnames:
                    gt_points = points[box_idxs_of_pts == i]

                    range_gt_box = np.linalg.norm(annos['location'][i, :])
                    dimensions = annos['dimensions'][i, :]
                    heading = annos['heading_angles'][i]
                    #num_pts_gt_waymo = annos['num_points_in_gt'][i]
                    num_pts_gt = gt_points.shape[0]

                    #assert(gt_points.shape[0] == num_pts)
                    obj_dict['seq_name'] = sequence_name
                    obj_dict['sample_idx'] = pc_info['sample_idx']
                    obj_dict['obj_id'] = i
                    obj_dict['range'] = range_gt_box
                    obj_dict['dimensions'] = dimensions
                    obj_dict['heading'] = heading
                    obj_dict['num_pts_gt'] = num_pts_gt

                    if num_pts_gt > 0:
                        intensities = gt_points[:,3]
                        obj_dict['mean_i'] = intensities.mean()
                        obj_dict['max_i'] = intensities.max()
                        obj_dict['min_i'] = intensities.min()
                        obj_dict['median_i'] = np.median(intensities)
                        obj_dict['std_i'] = intensities.std()
                        data_some_pts.append(obj_dict)
                        range_wise_summary_some_pts.append([range_gt_box,
                                                            num_pts_gt,
                                                            obj_dict['mean_i'],
                                                            obj_dict['std_i'],
                                                            obj_dict['median_i'],
                                                            obj_dict['min_i'],
                                                            obj_dict['max_i']])
                    else:
                        data_zero_pts.append(obj_dict)
                        range_wise_summary_zero_pts.append([range_gt_box,
                                                            num_pts_gt])



    summary = {'zero_pts': range_wise_summary_zero_pts,
               'some_pts': range_wise_summary_some_pts,
               'all_pc': range_wise_summary_all_pc}

    classname_string = ''
    for name in classnames:
        classname_string += name + '_'

    save_summary_pkl = root_path / (split + f'_{Rr}_{extra_tag}_gt_' + classname_string + 'summary.pkl')
    with open(save_summary_pkl, 'wb') as f:
        pickle.dump(summary, f)

    obj_wise_data_dict = {'zero_pts': data_zero_pts,
                 'some_pts': data_some_pts}

    save_pkl = root_path / (split + f'_{Rr}_{extra_tag}_gt_' + classname_string[:-1] + '.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump(obj_wise_data_dict, f)
        
def save_data(split, sim_rain=False):
    seq_list = readsplit(split, sim_rain)

    data_zero_pts = []
    data_some_pts = []
    range_wise_summary_zero_pts = []
    range_wise_summary_some_pts = []
    range_wise_summary_all_pc = {'means_intensity' : [],
                                 'std_intensity': [],
                                 'num_pts': []}

    for seq in tqdm(seq_list):
        seq_dir = processed_dir / seq
        seq_pkl = seq_dir / (seq + '.pkl')
        with open(seq_pkl, 'rb') as f:
            seq_info = pickle.load(f)

        for cnt, info in enumerate(seq_info):
            if cnt % frame_sampling_interval != 0:
                continue
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            points = get_lidar(sequence_name, sample_idx)
            range_bins, means_intensity, std_intensity, num_pts = range_wise_summary_pc(points)
            range_wise_summary_all_pc['means_intensity'].append(means_intensity)
            range_wise_summary_all_pc['std_intensity'].append(std_intensity)
            range_wise_summary_all_pc['num_pts'].append(num_pts)

            annos = info['annos']
            names = annos['name']
            gt_boxes = annos['gt_boxes_lidar']

            num_obj = gt_boxes.shape[0]

            box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
                torch.from_numpy(points[:, 0:3]).unsqueeze(dim=0).float().cuda(),
                torch.from_numpy(gt_boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
            ).long().squeeze(dim=0).cpu().numpy()

            obj_dict = {}
            for i in range(num_obj):
                if names[i] in classnames:
                    gt_points = points[box_idxs_of_pts == i]

                    range_gt_box = np.linalg.norm(annos['location'][i, :])
                    dimensions = annos['dimensions'][i, :]
                    heading = annos['heading_angles'][i]
                    #num_pts_gt_waymo = annos['num_points_in_gt'][i]
                    num_pts_gt = gt_points.shape[0]

                    #assert(gt_points.shape[0] == num_pts)
                    obj_dict['seq_name'] = sequence_name
                    obj_dict['sample_idx'] = pc_info['sample_idx']
                    obj_dict['obj_id'] = i
                    obj_dict['range'] = range_gt_box
                    obj_dict['dimensions'] = dimensions
                    obj_dict['heading'] = heading
                    obj_dict['num_pts_gt'] = num_pts_gt

                    if num_pts_gt > 0:
                        intensities = gt_points[:,3]
                        obj_dict['mean_i'] = intensities.mean()
                        obj_dict['max_i'] = intensities.max()
                        obj_dict['min_i'] = intensities.min()
                        obj_dict['median_i'] = np.median(intensities)
                        obj_dict['std_i'] = intensities.std()
                        data_some_pts.append(obj_dict)
                        range_wise_summary_some_pts.append([range_gt_box,
                                                            num_pts_gt,
                                                            obj_dict['mean_i'],
                                                            obj_dict['std_i'],
                                                            obj_dict['median_i'],
                                                            obj_dict['min_i'],
                                                            obj_dict['max_i']])
                    else:
                        data_zero_pts.append(obj_dict)
                        range_wise_summary_zero_pts.append([range_gt_box,
                                                            num_pts_gt])



    summary = {'zero_pts': range_wise_summary_zero_pts,
               'some_pts': range_wise_summary_some_pts,
               'all_pc': range_wise_summary_all_pc}

    classname_string = ''
    for name in classnames:
        classname_string += name + '_'

    save_summary_pkl = root_path / (split + '_gt_' + classname_string + 'summary.pkl')
    with open(save_summary_pkl, 'wb') as f:
        pickle.dump(summary, f)

    obj_wise_data_dict = {'zero_pts': data_zero_pts,
                 'some_pts': data_some_pts}

    save_pkl = root_path / (split + '_gt_' + classname_string[:-1] + '.pkl')
    with open(save_pkl, 'wb') as f:
        pickle.dump(obj_wise_data_dict, f)



def range_wise_num_pts_plot(save_summary_pkl, label, color=None, plot=True):
    with open(save_summary_pkl, 'rb') as f:
        summary = pickle.load(f)

    some_pts_summary = np.array(summary['some_pts'])
    some_pts_range = some_pts_summary[:,0]
    some_pts_num_pts_gt = some_pts_summary[:,1]

    zero_pts_summary = np.array(summary['zero_pts'])
    zero_pts_range = zero_pts_summary[:, 0]
    zero_pts_num_pts_gt = zero_pts_summary[:, 1]

    range_wise_means_all_pc = np.nanmean(np.array(summary['all_pc']['num_pts']), axis=0)
    range_wise_medians_all_pc = np.nanmedian(np.array(summary['all_pc']['num_pts']), axis=0)

    ranges = np.concatenate((some_pts_range, zero_pts_range))
    num_pts = np.concatenate((some_pts_num_pts_gt, zero_pts_num_pts_gt))

    bin_indices = np.digitize(ranges, range_bins)
    num_bins = range_bins.shape[0]-1
    means = np.zeros(num_bins)
    std = np.zeros(num_bins)

    for bin_i in range(1, num_bins+1):
        idx_for_ranges_in_bin_i = np.argwhere(bin_indices == bin_i)
        means[bin_i-1] = num_pts[idx_for_ranges_in_bin_i].mean()
        std[bin_i-1] = num_pts[idx_for_ranges_in_bin_i].std()


    if plot:
        p = plt.plot(range_bins[:-1], means, label=f'{label}_vehicle')
        # plt.plot(range_bins[:-1], means-std, '--', color=color)
        # plt.plot(range_bins[:-1], means + std, '--', color=color)
        plt.plot(range_bins[:-1], range_wise_means_all_pc, ':', label=f'{label}_all_pc', color=p[0].get_color())
        # plt.scatter(range_bins[:-1], means, marker='x')
        plt.xticks(range_bins)
        plt.ylabel('mean num pts gt')
        plt.xlabel('range bins')

        # Plot per object range vs num of pts in their gt box
        # plt.scatter(some_pts_range, some_pts_num_pts_gt, marker = 'x', linewidth=0.5)
        # plt.scatter(zero_pts_range, zero_pts_num_pts_gt, marker = 'x', linewidth=0.5)
        # plt.xlabel('Range')
        # plt.ylabel('num_pts_gt')

    return means, std, range_wise_means_all_pc, range_wise_medians_all_pc, p[0].get_color()

def range_wise_intensity_plot(save_summary_pkl, label, color=None, plot=True):
    with open(save_summary_pkl, 'rb') as f:
        summary = pickle.load(f)

    some_pts_summary = np.array(summary['some_pts'])
    some_pts_range = some_pts_summary[:,0]
    some_pts_intensity = some_pts_summary[:,2] # mean intensity in each gt vehicle

    summary_all_pc = summary['all_pc']['means_intensity']
    range_wise_means_all_pc = np.nanmean(np.array(summary['all_pc']['means_intensity']), axis=0)
    range_wise_medians_all_pc = np.nanmedian(np.array(summary['all_pc']['means_intensity']), axis=0)

    bin_indices = np.digitize(some_pts_range, range_bins)
    num_bins = range_bins.shape[0]-1
    means = np.zeros(num_bins)
    std = np.zeros(num_bins)
    range_wise_num_vehicles = np.zeros(num_bins)
    for bin_i in range(1, num_bins+1):
        idx_for_ranges_in_bin_i = np.argwhere(bin_indices == bin_i)
        means[bin_i-1] = some_pts_intensity[idx_for_ranges_in_bin_i].mean()
        std[bin_i-1] = some_pts_intensity[idx_for_ranges_in_bin_i].std()
        range_wise_num_vehicles[bin_i - 1] = idx_for_ranges_in_bin_i.shape[0]

    if plot:
        p = plt.plot(range_bins[:-1], means, label=f'{label}_vehicle')
        # plt.plot(range_bins[:-1], means - std, '--', color=color)
        # plt.plot(range_bins[:-1], means + std, '--', color=color)
        plt.plot(range_bins[:-1], range_wise_means_all_pc, ':', label=f'{label}_all_pc', color=p[0].get_color())
        #plt.plot(range_bins[:-1], range_wise_medians_all_pc, '--', color=color)
        #plt.scatter(range_bins[:-1], means, marker='x')
        plt.xticks(range_bins)
        plt.ylabel('mean intensity')
        plt.xlabel('range bins')

    return means, std, range_wise_means_all_pc, range_wise_medians_all_pc, range_wise_num_vehicles, p[0].get_color()

def plot_intensities_Rr(Rr, extra_tag):
    colors_Rr=[]
    means_sim_Rr = []
    means_sim_all_pc_Rr = []
    range_wise_num_vehicles_sim_Rr = []

    save_summary_pkl = root_path / 'all_da_gt_Vehicle_summary.pkl'
    label = 'real_rain'
    means_all_da, std_all_da, means_all_da_all_pc, medians_all_da_all_pc, range_wise_num_vehicles_da,_ = range_wise_intensity_plot(save_summary_pkl, label, color = 'r')

    label = 'clear'
    save_summary_pkl = root_path / 'train_clear_gt_Vehicle_summary.pkl'
    means_clear, std_clear, means_clear_all_pc, medians_clear_all_pc, range_wise_num_vehicles_clear,_  = range_wise_intensity_plot(save_summary_pkl, label, color = 'c')

    for r in Rr:
        label = f'sim_rain_{r}'
        save_summary_pkl = root_path / f'train_sim_rain_{r}_{extra_tag}_gt_Vehicle_summary.pkl'
        means_sim, std_sim, means_sim_all_pc, medians_sim_all_pc, range_wise_num_vehicles_sim, color = range_wise_intensity_plot(save_summary_pkl, label, color='b')
        means_sim_Rr.append(means_sim)
        means_sim_all_pc_Rr.append(means_sim_all_pc)
        range_wise_num_vehicles_sim_Rr.append(range_wise_num_vehicles_sim)
        colors_Rr.append(color)


    plt.legend()
    plt.show()

    plt.plot(range_bins[:-1], means_all_da_all_pc / means_clear_all_pc, ':', label='all_pc real', color='r')
    plt.plot(range_bins[:-1], means_all_da / means_clear, label='Vehicle real', color='r')
    for i in range(len(Rr)):
        plt.plot(range_bins[:-1], means_sim_Rr[i] / means_clear, label=f'Vehicle sim {Rr[i]}', color = colors_Rr[i])
        plt.plot(range_bins[:-1], means_sim_all_pc_Rr[i] / means_clear_all_pc, ':', label=f'all_pc sim {Rr[i]}', color = colors_Rr[i])

    plt.xticks(range_bins)
    plt.ylabel('mean intensity rain/clear')
    plt.xlabel('range bins')

    plt.legend()
    plt.show()

    plt.bar(range_bins[:-1], range_wise_num_vehicles_clear, label='clear', color='c')
    plt.bar(range_bins[:-1], range_wise_num_vehicles_sim_Rr[0], label = f'sim_Rr {Rr[0]}', color='b')
    plt.bar(range_bins[:-1], range_wise_num_vehicles_da, label = 'real rain', color= 'r')

    plt.xticks(range_bins)
    plt.ylabel('num vehicles')
    plt.xlabel('range bins')

    plt.legend()
    plt.show()

    return means_all_da/means_clear, means_all_da_all_pc / means_clear_all_pc

def plot_num_pts_Rr(Rr, extra_tag):
    colors_Rr=[]
    means_sim_Rr = []
    means_sim_all_pc_Rr = []

    save_summary_pkl = root_path / 'all_da_gt_Vehicle_summary.pkl'
    label = 'real_rain'
    means_all_da, std_all_da, means_all_da_all_pc, medians_all_da_all_pc, _ = range_wise_num_pts_plot(save_summary_pkl, label)

    label = 'clear'
    save_summary_pkl = root_path / 'train_clear_gt_Vehicle_summary.pkl'
    means_clear, std_clear, means_clear_all_pc, medians_clear_all_pc,_  = range_wise_num_pts_plot(save_summary_pkl, label)

    for r in Rr:
        label = f'sim_rain_{r}'
        save_summary_pkl = root_path / f'train_sim_rain_{r}_{extra_tag}_gt_Vehicle_summary.pkl'
        means_sim, std_sim, means_sim_all_pc, medians_sim_all_pc, color = range_wise_num_pts_plot(save_summary_pkl, label)
        means_sim_Rr.append(means_sim)
        means_sim_all_pc_Rr.append(means_sim_all_pc)
        colors_Rr.append(color)


    plt.legend()
    plt.show()

    plt.plot(range_bins[:-1], means_all_da_all_pc / means_clear_all_pc, ':', label='all_pc real', color='r')
    plt.plot(range_bins[:-1], means_all_da / means_clear, label='Vehicle real', color='r')
    for i in range(len(Rr)):
        plt.plot(range_bins[:-1], means_sim_Rr[i] / means_clear, label=f'Vehicle sim {Rr[i]}', color = colors_Rr[i])
        plt.plot(range_bins[:-1], means_sim_all_pc_Rr[i] / means_clear_all_pc, ':', label=f'all_pc sim {Rr[i]}', color = colors_Rr[i])

    plt.ylim([-1, 2])
    plt.xticks(range_bins)
    plt.ylabel('sparsity coeff = prob of keeping a point = mean num pts rain/clear') #'difference in mean intensity (clear - real rain)'
    plt.xlabel('range bins')


    plt.legend()
    plt.show()


    return means_all_da/means_clear, means_all_da_all_pc / means_clear_all_pc

def plot_intensities(Rr, extra_tag):
    label = 'sim_rain'
    save_summary_pkl = root_path / f'train_sim_rain_{Rr}_{extra_tag}_gt_Vehicle_summary.pkl'
    means_sim, std_sim, means_sim_all_pc, medians_sim_all_pc, range_wise_num_vehicles_sim = range_wise_intensity_plot(save_summary_pkl, label, color='b')

    save_summary_pkl = root_path / 'all_da_gt_Vehicle_summary.pkl'
    label = 'real_rain'
    means_all_da, std_all_da, means_all_da_all_pc, medians_all_da_all_pc, range_wise_num_vehicles_da = range_wise_intensity_plot(save_summary_pkl, label, color = 'r')

    label = 'clear'
    save_summary_pkl = root_path / 'train_clear_gt_Vehicle_summary.pkl'
    means_clear, std_clear, means_clear_all_pc, medians_clear_all_pc, range_wise_num_vehicles_clear  = range_wise_intensity_plot(save_summary_pkl, label, color = 'c')

    plt.legend()
    plt.show()

    plt.plot(range_bins[:-1], means_all_da/means_clear, label= 'Vehicle real', color = 'r')
    plt.plot(range_bins[:-1], means_sim / means_clear, label='Vehicle sim', color = 'b')

    plt.plot(range_bins[:-1], means_all_da_all_pc / means_clear_all_pc, ':', label='all_pc real', color = 'r')
    plt.plot(range_bins[:-1], means_sim_all_pc / means_clear_all_pc, ':', label='all_pc sim', color = 'b')

    plt.xticks(range_bins)
    plt.ylabel('mean intensity rain/clear')
    plt.xlabel('range bins')

    plt.legend()
    plt.show()

    plt.bar(range_bins[:-1], range_wise_num_vehicles_clear, label='clear', color='c')
    plt.bar(range_bins[:-1], range_wise_num_vehicles_sim, label = 'sim', color='b')
    plt.bar(range_bins[:-1], range_wise_num_vehicles_da, label = 'real rain', color= 'r')

    plt.xticks(range_bins)
    plt.ylabel('num vehicles')
    plt.xlabel('range bins')

    plt.legend()
    plt.show()

    return means_all_da/means_clear, means_all_da_all_pc / means_clear_all_pc


def plot_num_pts_gt(Rr, extra_tag):
    label = 'sim_rain'
    save_summary_pkl = root_path / f'train_sim_rain_{Rr}_{extra_tag}_gt_Vehicle_summary.pkl'
    means_sim, std_sim, means_sim_all_pc, medians_sim_all_pc = range_wise_num_pts_plot(save_summary_pkl, label, color='b')

    save_summary_pkl = root_path / 'all_da_gt_Vehicle_summary.pkl'
    label = 'real_rain'
    means_all_da, std_all_da, means_all_da_all_pc, medians_all_da_all_pc  = range_wise_num_pts_plot(save_summary_pkl, label, color = 'r')

    label = 'clear'
    save_summary_pkl = root_path / 'train_clear_gt_Vehicle_summary.pkl'
    means_clear, std_clear, means_clear_all_pc, medians_clear_all_pc  = range_wise_num_pts_plot(save_summary_pkl, label, color = 'c')

    plt.legend()
    plt.show()

    plt.plot(range_bins[:-1], means_all_da/means_clear, label= 'Vehicle real', color = 'r')
    plt.plot(range_bins[:-1], means_sim / means_clear, label='Vehicle sim', color = 'b')

    plt.plot(range_bins[:-1], means_all_da_all_pc / means_clear_all_pc, ':', label='all_pc real', color = 'r')
    plt.plot(range_bins[:-1], means_sim_all_pc / means_clear_all_pc, ':', label='all_pc sim', color = 'b')

    plt.ylim([-1, 2])
    plt.xticks(range_bins)
    plt.ylabel('sparsity coeff = prob of keeping a point = mean num pts rain/clear') #'difference in mean intensity (clear - real rain)'
    plt.xlabel('range bins')

    plt.legend()
    plt.show()
    return means_all_da / means_clear, means_all_da_all_pc / means_clear_all_pc

def main():
    # split = 'train_clear'
    # save_data(split)
    #
    # split = 'all_da'
    # save_data(split)

    split = 'train_sim_rain'
    extra_tag = 'rmax150_prob'
    rain_rates = [0.5] #[0.5, 5, 50]
    # for Rr in rain_rates:
    #     sim_rain_save_data(split, Rr, extra_tag=extra_tag, sim_rain=True)

    # s_i, s_i_all  = plot_intensities(Rr, extra_tag)
    # s_n, s_n_all = plot_num_pts_gt(Rr, extra_tag)

    plot_intensities_Rr(rain_rates, extra_tag)
    plot_num_pts_Rr(rain_rates, extra_tag)

    # alpha = {'range_bins': range_bins,
    #          'scale_intensity_vehicle': s_i,
    #          'prob_keep_vehicle': s_n,
    #          'scale_intensity': s_i_all,
    #          'prob_keep': s_n_all
    #          }

    # with open('rain_pattern.pkl', 'wb') as f:
    #     pickle.dump(alpha, f)

main()