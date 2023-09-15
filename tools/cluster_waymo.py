# import argparse

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from third_party.OpenPCDet.tools.approx_bbox_utils import *
from third_party.OpenPCDet.tools.cluster_utils import *
from third_party.OpenPCDet.pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from third_party.OpenPCDet.pcdet.ops.iou3d_nms import iou3d_nms_utils
from third_party.OpenPCDet.tools.tracker import PubTracker as Tracker
from third_party.OpenPCDet.tools.estimate_ground import estimate_ground
from third_party.OpenPCDet.pcdet.utils import box_utils
import multiprocessing as mp
from functools import partial
from matplotlib.lines import Line2D

# import open3d as o3d
import os
import torch
import glob
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V

np.random.seed(100)

def inv_pose(pose):
    pose_inv = np.zeros((4,4))
    pose_inv[-1,-1] = 1
    pose_inv[:3,:3] = pose[:3,:3].T
    pose_inv[:3,-1] = - pose_inv[:3,:3] @ pose[:3,-1]
    return pose_inv

class WaymoDataset():
    def __init__(self):
        self.root_path = Path('/home/barza/DepthContrast/data/waymo')
        self.processed_data_tag='waymo_processed_data_10_short'
        self.split = 'train_short'
        self.data_path = self.root_path / self.processed_data_tag
        self.infos_pkl_path = self.root_path / f'{self.processed_data_tag}_infos_{self.split}.pkl'
        self.label_root_path = self.root_path / (self.processed_data_tag + '_clustered')
        self.save_infos_pkl_path = self.root_path / f'{self.processed_data_tag}_infos_{self.split}_approx_boxes.pkl'
        self.class_names = ['Vehicle', 'Pedestrian', 'Cyclist']
        self.iou_eval_thresh_list = [0.03, 0.2, 0.5] #0.03, 
        self.eval_metrics = ['iou3d', 'overlap3d', 'overlapbev']


        self.infos_dict = {} # seq_name: [frame infos]
        self.include_waymo_data() # read tfrecords in sample_seq_list and then find its pkl in waymo_processed_data_10 and include the pkl infos in waymo infos

    def include_waymo_data(self):
        infos=[]
        with open(self.infos_pkl_path, 'rb') as f:
            infos = pickle.load(f) # loads all infos

        for info in infos:
            sequence_name = info['point_cloud']['lidar_sequence']
            if sequence_name not in self.infos_dict:
                self.infos_dict[sequence_name] = []
                #print(sequence_name)
            self.infos_dict[sequence_name].append(info)
    
    def get_lidar(self, seq_name, sample_idx):
        lidar_file = self.data_path / seq_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        point_features[:,3] = np.tanh(point_features[:, 3]) * 255.0
        return point_features[:,:4] #only get xyzi
    
    def get_cluster_labels(self, seq_name, sample_idx):
        label_file = self.label_root_path / seq_name / ('%04d.npy' % sample_idx)
        labels = np.fromfile(label_file, dtype=np.float16)
        return labels
    
    def get_ground_mask(self, seq_name, sample_idx):
        path = self.label_root_path / seq_name / 'ground' /('%04d.npy' % sample_idx)
        ground_mask = np.fromfile(path, dtype=np.bool_)
        return ground_mask
    
    def get_rejection_tag(self, seq_name, sample_idx):
        path = self.label_root_path / seq_name / 'rejection_tag' /('%04d.npy' % sample_idx)
        pt_wise_rejection_tag = np.fromfile(path, dtype=np.uint8)
        return pt_wise_rejection_tag
    
    def save_updated_infos(self):
        infos_list = []
        for seq_name, seq_infos in self.infos_dict.items():
            infos_list += seq_infos
        with open(self.save_infos_pkl_path, 'wb') as f:
            pickle.dump(infos_list,f)

    def aggregate_pcd_in_world(self, seq_name): 
        infos = self.infos_dict[seq_name]

        pc_lens = []
        xyz_world = np.empty((0,3))
        intensities = np.empty((0,1))
        for info in infos:
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']

            #print(sample_idx)
            pose_world_from_vehicle = info['pose']

            #points in current vehicle frame
            xyzi = self.get_lidar(seq_name, sample_idx)
            num_points=xyzi.shape[0]
            pc_lens.append(num_points)

            # Transform the points from the vehicle frame to the world frame.
            xyz = np.concatenate([xyzi[:,:3], np.ones([num_points, 1])], axis=-1) #(N, xyz1)
            xyz = np.matmul(pose_world_from_vehicle, xyz.T).T[:,:3] #(N, xyz)

            xyz_world = np.vstack([xyz_world, xyz])
            intensities = np.vstack([intensities, xyzi[:,-1][...,np.newaxis]])
        
        xyzi_world = np.concatenate([xyz_world, intensities], axis=-1)

        return xyzi_world, pc_lens
    
    def aggregate_pcd_in_frame_i(self, seq_name, i=-1):
        infos = self.infos_dict[seq_name]

        xyzi_world, pc_lens = self.aggregate_pcd_in_world(seq_name)
        
        xyz_world = xyzi_world[:,:3]
        # Inv(pose_world_from_vehicle)
        # last frame pose of vehicle wrt world
        pose_world_from_vehicle = infos[i]['pose']
        pose_vehicle_from_world = inv_pose(pose_world_from_vehicle)

        # diff=(pose_vehicle_from_world - np.linalg.inv(pose_world_from_vehicle)).sum()
        # Transform the points from the world frame to the last frame in the sequence
        xyz_world =  np.concatenate([xyz_world, np.ones([xyz_world.shape[0], 1])], axis=-1)
        xyz_frame_i = np.matmul(pose_vehicle_from_world, xyz_world.T).T[:,:3] #(N, xyz)


        # Append intensity
        xyzi_frame_i = np.concatenate([xyz_frame_i, xyzi_world[:,-1][..., np.newaxis]], axis=-1)

        # Visualize aggregated pcs of the seq in last frame
        #V.draw_scenes(xyz_last_vehicle)

        # Visualize aggregated pcs of the seq in last frame without ground plane
        #V.draw_scenes(xyz_last_vehicle[:,:3][non_ground_mask])

        return xyzi_frame_i, pc_lens
    
    def estimate_ground_seq(self, seq_name):
        save_seq_path = self.label_root_path / seq_name / 'ground'
        os.makedirs(save_seq_path.__str__(), exist_ok=True)
        infos = self.infos_dict[seq_name]
        non_ground_mask = np.empty((0,1), dtype=bool)
        for info in infos:
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']

            #points in current vehicle frame
            xyzi = self.get_lidar(seq_name, sample_idx)
            
            ground_mask = estimate_ground(xyzi)
            save_path = save_seq_path / ('%04d.npy' % sample_idx)
            ground_mask.tofile(save_path)
        
        print('Estimating ground Done!')

def transform_pc_to_world(xyz, pose_world_from_vehicle):
    xyz = np.concatenate([xyz, np.ones([xyz.shape[0], 1])], axis=-1) #(N, xyz1)
    xyz = np.matmul(pose_world_from_vehicle, xyz.T).T[:,:3] #(N, xyz)

    return xyz

def transform_pc_to_frame_i(xyz, pose_world_from_vehicle, pose_world_from_frame_i):
    xyz_world = transform_pc_to_world(xyz, pose_world_from_vehicle)
    pose_frame_i_from_world = inv_pose(pose_world_from_frame_i)

    # diff=(pose_vehicle_from_world - np.linalg.inv(pose_world_from_vehicle)).sum()
    # Transform the points from the world frame to the last frame in the sequence
    xyz_world =  np.concatenate([xyz_world, np.ones([xyz_world.shape[0], 1])], axis=-1)
    xyz_frame_i = np.matmul(pose_frame_i_from_world, xyz_world.T).T[:,:3] #(N, xyz)

    return xyz_frame_i

def fill_in_clusters_with_knn(labels, xyz, show_plots):
    # labels and xyz are of nonground pts
    unlabeled_pts_mask = labels == -1
    labeled_pts_mask = labels > -1
    unlabeled_pc = xyz[unlabeled_pts_mask]
    labeled_pc = xyz[labeled_pts_mask]

    labeled_pcd = o3d.geometry.PointCloud()
    labeled_pcd.points = o3d.utility.Vector3dVector(labeled_pc)

    labeled_tree = o3d.geometry.KDTreeFlann(labeled_pcd)
    for i in range(unlabeled_pc.shape[0]):
        pt = unlabeled_pc[i]
        #Find its neighbors with distance less than 0.2
        [_, idx, _] = labeled_tree.search_radius_vector_3d(pt, 0.4)
        if len(idx):
            nearest_labels = labels[labeled_pts_mask][np.asarray(idx)]
            label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
            labels[unlabeled_pts_mask][i] = label_of_majority

    if show_plots:
        # Filled in clusters
        visualize_pcd_clusters(xyz, labels.reshape((-1,1)))

    return labels #of nonground pts

def fill_in_clusters_with_bbox_knn(labels, pc, show_plots=False):
    boxes = np.empty((0,8))
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_pc = pc[labels==label, :3]
        # Note: sending full_pc gets max point in the bev box to be in cluster. This could overestimate the box height
        # if there is a tree over the car 

        # if cluster_pc[:,2].max() - cluster_pc[:,2].min() < 0.8:
        #     continue
        #Fit box to cluster, ignoring outlier ground points
        min_height = np.max([cluster_pc[:,2].min()+0.2,cluster_pc[:,2].min()-0.2])
        top_part_cluster = cluster_pc[cluster_pc[:,2]>min_height]
        
        if top_part_cluster.shape[0] == 0:
            top_part_cluster=cluster_pc
        box, _, _ = fit_box(top_part_cluster, fit_method='min_zx_area_fit') #, full_pc = pc[:,:3]
        #shift box back down
        #box[2] -= 0.2
        box[5] += 0.3 #increase height
        #append label
        box = np.concatenate((box, [label]), axis = -1)
        boxes = np.vstack([boxes, box])
    
    #fitted boxes
    print(f'4th Step Fit boxes on aggregate PC Done. Boxes: {boxes.shape[0]}')

    if show_plots:
        V.draw_scenes(pc[:,:3], gt_boxes=boxes)

    # try:
    #     box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #     torch.from_numpy(pc[:, 0:3]).unsqueeze(dim=0).float().cuda(),
    #     torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
    #     ).long().squeeze(dim=0).cpu().numpy() #(npoints)

    #     for i in range(boxes.shape[0]):
    #         label = boxes[i, -1]
    #         labels[box_idxs_of_pts == i] = label
    # except:

     # labels and xyz are of nonground pts
    unlabeled_pts_mask = labels == -1
    labeled_pts_mask = labels > -1
    unlabeled_pc = pc[unlabeled_pts_mask]
    labeled_pc = pc[labeled_pts_mask]
    labels_for_unlabeled_pc = labels[unlabeled_pts_mask]
    labels_for_labeled_pc = labels[labeled_pts_mask]

    # bev_corners = box_utils.boxes_to_corners_3d(boxes[:, 0:7])[:, :12] #get_box_corners(cxyz, lwh, heading)
    # flag = box_utils.in_hull(unlabeled_pc[:,:2], bev_corners)
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    torch.from_numpy(unlabeled_pc[:,:3]), torch.from_numpy(boxes[:, 0:7])).numpy()  # (nboxes, n unlabelled points mask) 1 if box i contains pt j, 0 otherwise
    
    pt_indices_mask_in_one_box = point_indices.sum(axis=0) == 1
    pt_indices_mask_in_multiple_boxes = point_indices.sum(axis=0) > 1
    for i in range(boxes.shape[0]):
        label = boxes[i, -1]
        labels_for_unlabeled_pc[(point_indices[i]>0) & pt_indices_mask_in_one_box] = label
    
    #Resolve points in multiple boxes with knn
    num_unlabeled_pts_in_multiple_boxes = pt_indices_mask_in_multiple_boxes.sum() #unlabeled_pc[pt_indices_mask_in_multiple_boxes].shape[0]
    if num_unlabeled_pts_in_multiple_boxes:
        labeled_pcd = o3d.geometry.PointCloud()
        labeled_pcd.points = o3d.utility.Vector3dVector(labeled_pc)

        labeled_tree = o3d.geometry.KDTreeFlann(labeled_pcd)
        conflicted_pts = unlabeled_pc[pt_indices_mask_in_multiple_boxes]
        labels_for_conflicted_pts = labels_for_unlabeled_pc[pt_indices_mask_in_multiple_boxes]
        for i in range(num_unlabeled_pts_in_multiple_boxes):
            conflicted_pt = conflicted_pts[i]
            #Find its neighbors with distance less than 0.2
            [_, idx, _] = labeled_tree.search_knn_vector_3d(conflicted_pt, 5)
            if len(idx):
                nearest_labels = labels_for_labeled_pc[np.asarray(idx)]
                label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
                labels_for_conflicted_pts[i] = label_of_majority
        
        labels_for_unlabeled_pc[pt_indices_mask_in_multiple_boxes] = labels_for_conflicted_pts

    labels[unlabeled_pts_mask] = labels_for_unlabeled_pc

    print(f'5th Step Clusters filling Done.')

    if show_plots:
        # Filled in clusters
        visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)), boxes=boxes, mode='')
        visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)))

    return labels


def fill_in_clusters_with_bbox(labels, pc, show_plots=False):
    boxes = np.empty((0,8))
    for label in np.unique(labels):
        if label == -1:
            continue
        cluster_pc = pc[labels==label, :3]
        # Note: sending full_pc gets max point in the bev box to be in cluster. This could overestimate the box height
        # if there is a tree over the car 

        if cluster_pc[:,2].max() - cluster_pc[:,2].min() < 0.8:
            continue
        #Remove outlier ground points in the cluster
        
        min_height = np.max([cluster_pc[:,2].min()+0.3,cluster_pc[:,2].min()-0.3])
        top_part_cluster = cluster_pc[cluster_pc[:,2]>min_height]
        
        if top_part_cluster.shape[0] == 0:
            top_part_cluster=cluster_pc
        box, _, _ = fit_box(top_part_cluster, fit_method='closeness_to_edge') #, full_pc = pc[:,:3]
        #shift box up
        #box[2] += 0.3
        #append label
        box = np.concatenate((box, [label]), axis = -1)
        boxes = np.vstack([boxes, box])
    
    #fitted boxes
    print(f'4th Step Fit boxes on aggregate PC Done. Boxes: {boxes.shape[0]}')

    if show_plots:
        V.draw_scenes(pc[:,:3], gt_boxes=boxes)

    # try:
    #     box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    #     torch.from_numpy(pc[:, 0:3]).unsqueeze(dim=0).float().cuda(),
    #     torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
    #     ).long().squeeze(dim=0).cpu().numpy() #(npoints)

    #     for i in range(boxes.shape[0]):
    #         label = boxes[i, -1]
    #         labels[box_idxs_of_pts == i] = label
    # except:
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
    torch.from_numpy(pc[:, 0:3]), torch.from_numpy(boxes[:, 0:7])).numpy()  # (nboxes, npoints)
    for i in range(boxes.shape[0]):
        label = boxes[i, -1]
        labels[point_indices[i]>0] = label

        
    print(f'5th Step Clusters filling Done.')

    if show_plots:
        # Filled in clusters
        visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)))

    return labels

def cluster_in_aggregated(seq_name, dataset, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    infos = dataset.infos_dict[seq_name]

    print(f'Clustering of sequence: {seq_name} started!')
    xyzi_last_vehicle, pc_lens = dataset.aggregate_pcd_in_frame_i(seq_name, -1)

    ground_mask_all = np.empty((0), dtype=np.bool_)
    for info in infos:
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
        ground_mask_all = np.concatenate([ground_mask_all, ground_mask])
    
    xyz = xyzi_last_vehicle[:,:3]
    above_plane_mask = np.logical_not(ground_mask_all)
    

    labels = cluster(xyz, above_plane_mask, eps=0.2)
    print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
    
    if show_plots:
        visualize_pcd_clusters(xyz, labels.reshape((-1,1)))

    # Filter big volumes/smears of moving objects
    new_labels, rejection_tag  = filter_labels(xyz, labels,
                                    max_volume=80, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = ground_mask_all)
    
    if show_plots:
        for key, val in REJECT.items():
            rejected_labels = np.where(rejection_tag == REJECT[key])[0]
            if len(rejected_labels):
                print(f'rejected_labels: {rejected_labels}')
                print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
                visualize_selected_labels(xyz, labels.flatten(), rejected_labels)
    
    # save_labels_path = save_seq_path / 'all.npy'
    # # save_labels(new_labels, save_labels_path.__str__())
    # labels = load_labels(save_labels_path.__str__())
    
    labels = remove_outliers_cluster(xyz, new_labels.flatten())
    labels = get_continuous_labels(labels)
    print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')

    if show_plots:
        visualize_pcd_clusters(xyz, labels.reshape((-1,1)))
    

    #save_labels_path = save_seq_path / 'all.npy'
    #save_labels(labels, save_labels_path.__str__())
    #labels = load_labels(save_labels_path.__str__())

    labels_above_plane = fill_in_clusters_with_bbox_knn(labels[above_plane_mask], xyz[above_plane_mask], show_plots=show_plots)
    labels[above_plane_mask] = labels_above_plane

    if show_plots:
        visualize_pcd_clusters(xyz, labels.reshape((-1,1)))

    # Save cluster to frame idx map
    cluster2frame_id_dict = {}
    pc_start = 0
    for frame_idx, info in enumerate(infos):
        for cluster_id in np.unique(labels[pc_start:pc_start+pc_lens[frame_idx]]):
            if cluster_id == -1:
                continue
            if cluster_id not in cluster2frame_id_dict:
                cluster2frame_id_dict[cluster_id] = []

            cluster2frame_id_dict[cluster_id].append(frame_idx)

        pc_start += pc_lens[frame_idx]

    save_path = save_seq_path / 'cluster2frame_id_dict.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(cluster2frame_id_dict, f)
    
    max_cluster_id_dict = {'max_cluster_id_aggregated' : labels.max()}
    save_path = save_seq_path / 'max_cluster_id.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(max_cluster_id_dict, f)

    # Separate labels pc wise and save
    infos = dataset.infos_dict[seq_name]
    i=0
    labels = labels.astype(np.float16)
    for info, pc_len in zip(infos, pc_lens):
        sample_idx = info['point_cloud']['sample_idx']
        save_label_path = save_seq_path / ('%04d.npy' % sample_idx)

        label_this_pc = labels[i:i+pc_len]
        i+=pc_len

        label_this_pc.tofile(save_label_path.__str__())

def cluster_only(seq_name, dataset, num_frames_to_aggr = 3, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    os.makedirs((save_seq_path / 'rejection_tag').__str__(), exist_ok=True)
    print(f'Clustering of sequence: {seq_name} started!')
    infos = dataset.infos_dict[seq_name]
    start_info_indices = np.arange(0, len(infos), num_frames_to_aggr)

    max_label = -1
    for start_idx in start_info_indices:
        
        aggr_infos = infos[start_idx:start_idx+num_frames_to_aggr]
        aggr_pcs_in_world = np.zeros((0,3))
        aggr_ground_mask = np.zeros(0, dtype=np.bool_)
        pc_lens = [] 
        for i, info in enumerate(aggr_infos):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            print(f'sample idx: {sample_idx}')
            
            xyzi = dataset.get_lidar(seq_name, sample_idx)
            ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
            xyzi[:,:3] = transform_pc_to_world(xyzi[:,:3], info['pose'])
            pc_lens.append(xyzi.shape[0])
            
            aggr_ground_mask = np.hstack([aggr_ground_mask, ground_mask])
            aggr_pcs_in_world = np.vstack([aggr_pcs_in_world, xyzi[:,:3]])

        # Get new labels
        labels = cluster(aggr_pcs_in_world, np.logical_not(aggr_ground_mask), eps=0.2)
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        if show_plots:
            print('showing new clustered labels')
            visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        new_labels, label_wise_rejection_tag  = filter_labels(aggr_pcs_in_world, labels,
                                    max_volume=None, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = aggr_ground_mask)
        
        pt_wise_rejection_tag = np.zeros(aggr_pcs_in_world.shape[0], dtype=np.uint8) #zero means not rejected
        for i in np.unique(labels):
            if i == -1:
                continue
            pt_wise_rejection_tag[labels == i] = label_wise_rejection_tag[int(i)]


        # if show_plots:
        #     for key, val in REJECT.items():
        #         rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
        #         if len(rejected_labels):
        #             print(f'rejected_labels: {rejected_labels}')
        #             print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
        #             visualize_selected_labels(aggr_pcs_in_world, labels.flatten(), rejected_labels)
        
        labels = remove_outliers_cluster(aggr_pcs_in_world, new_labels.flatten())
        labels = get_continuous_labels(labels)
        print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')
        if show_plots:
            print('showing new filtered labels')
            visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        labels[labels>-1] = labels[labels>-1] + max_label + 1
        max_label = np.max(labels)

    
        if show_plots:
            print('showing final labels')
            visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        i=0
        labels = labels.astype(np.float16)
        pt_wise_rejection_tag[labels>-1] = 0
        pt_wise_rejection_tag.astype(np.uint8)
        for info, pc_len in zip(aggr_infos, pc_lens):
            sample_idx = info['point_cloud']['sample_idx']
            print(f'Saving sample: {sample_idx}')
            save_path = save_seq_path / ('%04d.npy' % sample_idx)

            label_this_pc = labels[i:i+pc_len]
            label_this_pc.tofile(save_path.__str__())

            # Save rejection tag for each pt
            save_path = save_seq_path / 'rejection_tag'/ ('%04d.npy' % sample_idx)
            rej_tag_this_pc = pt_wise_rejection_tag[i:i+pc_len]
            rej_tag_this_pc.tofile(save_path.__str__())

            i+=pc_len




def cluster_tracking(seq_name, dataset, num_frames_to_aggr = 3, initial_guess=False, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    os.makedirs((save_seq_path / 'rejection_tag').__str__(), exist_ok=True)
    print(f'Clustering of sequence: {seq_name} started!')
    
    if initial_guess:
        #get clusters from the aggregated pcs of whole seq
        path = save_seq_path / 'max_cluster_id.pkl'
        with open(path, 'rb') as f:
            max_cluster_id_dict = pickle.load(f)
        max_label = max_cluster_id_dict['max_cluster_id_aggregated']
    else:
        max_label = -1

    # path = save_seq_path / 'cluster2frame_id_dict.pkl'
    # with open(path, 'rb') as f:
    #     cluster2frame_id_dict = pickle.load(f)

    infos = dataset.infos_dict[seq_name]
    start_info_indices = np.arange(0, len(infos)-num_frames_to_aggr+1)

    for start_idx in start_info_indices:
        
        aggr_infos = infos[start_idx:start_idx+num_frames_to_aggr]
        aggr_pcs_in_world = np.zeros((0,3))
        aggr_ground_mask = np.zeros(0, dtype=np.bool_)
        old_labels = np.zeros(0)
        pc_lens = [] 
        for i, info in enumerate(aggr_infos):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            print(f'sample idx: {sample_idx}')
            
            xyzi = dataset.get_lidar(seq_name, sample_idx)
            if initial_guess:
                labels = dataset.get_cluster_labels(seq_name, sample_idx)
            else:
                if start_idx == start_info_indices[0] or i == len(aggr_infos)-1:
                    labels = -1. * np.ones(xyzi.shape[0], dtype=np.float16)
                else:
                    labels = dataset.get_cluster_labels(seq_name, sample_idx)
            ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
            xyzi[:,:3] = transform_pc_to_world(xyzi[:,:3], info['pose'])
            pc_lens.append(xyzi.shape[0])
            
            
            aggr_ground_mask = np.hstack([aggr_ground_mask, ground_mask])
            if initial_guess and start_idx != start_info_indices[0] and i == len(aggr_infos)-1:
                
                old_labeled_pcd = o3d.geometry.PointCloud()
                old_labeled_pcd.points = o3d.utility.Vector3dVector(aggr_pcs_in_world[old_labels>-1])

                old_labeled_tree = o3d.geometry.KDTreeFlann(old_labeled_pcd)
                new_labels = -1* np.ones_like(labels)
                for l in np.unique(labels):
                    if l == -1:
                        continue
                    points_new_lbl = xyzi[labels == l]
                    for i in range(points_new_lbl.shape[0]):
                        pt = points_new_lbl[i,:3]
                        #Find its neighbors with distance less than 0.2
                        [_, idx, _] = old_labeled_tree.search_radius_vector_3d(pt, 0.2)
                        if len(idx):
                            nearest_labels = old_labels[old_labels>-1][np.asarray(idx)]
                            label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
                            new_labels[labels == l] = label_of_majority
                            break
                
                labels = new_labels
            aggr_pcs_in_world = np.vstack([aggr_pcs_in_world, xyzi[:,:3]])
            old_labels = np.hstack([old_labels, labels])

        # Get new labels
        labels = cluster(aggr_pcs_in_world, np.logical_not(aggr_ground_mask), eps=0.2)
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        # if show_plots:
        #     print('showing old labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
        #     print('showing new clustered labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        new_labels, label_wise_rejection_tag  = filter_labels(aggr_pcs_in_world, labels,
                                    max_volume=None, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = aggr_ground_mask)
        
        pt_wise_rejection_tag = np.zeros(aggr_pcs_in_world.shape[0], dtype=np.uint8) #zero means not rejected
        for i in np.unique(labels):
            if i == -1:
                continue
            pt_wise_rejection_tag[labels == i] = label_wise_rejection_tag[int(i)]


        # if show_plots:
        #     for key, val in REJECT.items():
        #         rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
        #         if len(rejected_labels):
        #             print(f'rejected_labels: {rejected_labels}')
        #             print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
        #             visualize_selected_labels(aggr_pcs_in_world, labels.flatten(), rejected_labels)
        
        labels = remove_outliers_cluster(aggr_pcs_in_world, new_labels.flatten())
        labels = get_continuous_labels(labels)
        print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')
        # if show_plots:
        #     #visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
        #     print('showing new filtered labels')
        #     visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))

        lbls_only_in_new_labels = []
        label_new2old_dict = {}

        for i in np.unique(labels):
            if i == -1:
                continue
            old_lbls_for_lbl_i = old_labels[labels==i]
            if old_lbls_for_lbl_i.max() == -1:
                lbls_only_in_new_labels.append(i)
            else:
                obj_old_labels = old_lbls_for_lbl_i[old_lbls_for_lbl_i>-1]
                label_new2old_dict[i] = {'majority_old_label': -1,
                                         'old_labels': [],
                                         'old_labels_count': []}
                # old_labels_count = []
                for old_lbl in np.unique(obj_old_labels):
                    label_new2old_dict[i]['old_labels'].append(old_lbl)
                    label_new2old_dict[i]['old_labels_count'].append((obj_old_labels == old_lbl).sum()) #old_labels == old_lbl
                
                label_new2old_dict[i]['majority_old_label'] = label_new2old_dict[i]['old_labels'][np.argmax(label_new2old_dict[i]['old_labels_count'])]

        # Start fusing aggregated view and new frame view labels
        new_labels = old_labels.copy()
        
        #Add labels only in new labels
        if len(lbls_only_in_new_labels):
            label_mapping = {lbl:i+max_label+1 for i, lbl in enumerate(sorted(lbls_only_in_new_labels))}
            for lbl, new_lbl in label_mapping.items():
                new_labels[labels == lbl] = new_lbl
                #cluster2frame_id_dict[new_lbl] = [frame_idx]
            max_label = np.max(new_labels)
            #save max cluster id
            if initial_guess:
                path = save_seq_path / 'max_cluster_id.pkl'
                max_cluster_id_dict['max_cluster_id'] = max_label
                with open(path, 'wb') as f:
                    pickle.dump(max_cluster_id_dict,f)
            
            # # save new cluster2frame_id_dict
            # save_path = save_seq_path / 'cluster2frame_id_dict.pkl'
            # with open(save_path, 'wb') as f:
            #     pickle.dump(cluster2frame_id_dict, f)

        # if show_plots:
        #     visualize_pcd_clusters(aggr_pcs_in_world, new_labels.reshape((-1,1)))
        # Keep old labels if old2new label connection exists
        for new_lbl, value in label_new2old_dict.items():
            new_labels[labels == new_lbl] = value['majority_old_label']

        if show_plots:
            # print('showing old labels')
            # visualize_pcd_clusters(aggr_pcs_in_world, old_labels.reshape((-1,1)))
            # print('showing new_filtered labels')
            # visualize_pcd_clusters(aggr_pcs_in_world, labels.reshape((-1,1)))
            print('showing final labels')
            visualize_pcd_clusters(aggr_pcs_in_world, new_labels.reshape((-1,1)))

        i=0
        new_labels = new_labels.astype(np.float16)
        pt_wise_rejection_tag[new_labels>-1] = 0
        pt_wise_rejection_tag.astype(np.uint8)
        for info, pc_len in zip(aggr_infos, pc_lens):
            sample_idx = info['point_cloud']['sample_idx']
            print(f'Saving sample: {sample_idx}')
            save_path = save_seq_path / ('%04d.npy' % sample_idx)

            label_this_pc = new_labels[i:i+pc_len]
            label_this_pc.tofile(save_path.__str__())

            # Save rejection tag for each pt
            save_path = save_seq_path / 'rejection_tag'/ ('%04d.npy' % sample_idx)
            rej_tag_this_pc = pt_wise_rejection_tag[i:i+pc_len]
            rej_tag_this_pc.tofile(save_path.__str__())

            i+=pc_len

def cluster_seq_each_pc(seq_name, dataset, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    os.makedirs((save_seq_path / 'rejection_tag').__str__(), exist_ok=True)

    print(f'Clustering of sequence: {seq_name} started!')
    path = save_seq_path / 'max_cluster_id.pkl'
    with open(path, 'rb') as f:
        max_cluster_id_dict = pickle.load(f)
    max_label = max_cluster_id_dict['max_cluster_id_aggregated']

    path = save_seq_path / 'cluster2frame_id_dict.pkl'
    with open(path, 'rb') as f:
        cluster2frame_id_dict = pickle.load(f)

    infos = dataset.infos_dict[seq_name]
    for frame_idx, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        gt_boxes = info['annos']['gt_boxes_lidar']
        
        xyzi = dataset.get_lidar(seq_name, sample_idx)
        labels_from_aggregated_clustering = dataset.get_cluster_labels(seq_name, sample_idx)
        ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
        

        # cluster points that are not ground
        xyz = xyzi[:,:3]
        labels = cluster(xyz, np.logical_not(ground_mask))
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
        if show_plots:
            visualize_pcd_clusters(xyz, labels_from_aggregated_clustering.reshape((-1,1)))
            visualize_pcd_clusters(xyz, labels.reshape((-1,1)), boxes=gt_boxes, mode='')
        

        new_labels, label_wise_rejection_tag  = filter_labels(xyz, labels,
                                    max_volume=460, min_volume=0.1, 
                                    max_height_for_lowest_point=1, 
                                    min_height_for_highest_point=0.5,
                                    ground_mask = ground_mask)
        
        pt_wise_rejection_tag = np.zeros(xyz.shape[0], dtype=np.uint8) #zero means not rejected
        for i in np.unique(labels):
            if i == -1:
                continue
            pt_wise_rejection_tag[labels == i] = label_wise_rejection_tag[int(i)]


        # if show_plots:
        #     for key, val in REJECT.items():
        #         rejected_labels = np.where(label_wise_rejection_tag == REJECT[key])[0]
        #         if len(rejected_labels):
        #             print(f'rejected_labels: {rejected_labels}')
        #             print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
        #             visualize_selected_labels(xyz, labels.flatten(), rejected_labels)
        
        labels = get_continuous_labels(new_labels)
        print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')
        if show_plots:
            visualize_pcd_clusters(xyz, labels_from_aggregated_clustering.reshape((-1,1)))
            visualize_pcd_clusters(xyz, labels.reshape((-1,1)), boxes=gt_boxes, mode='')
        labels_only_in_frame_view = []
        labels_frame2aggr_map = {}
        labels_only_in_aggr_view = []
        
        for i in np.unique(labels):
            if i == -1:
                continue
            pt_indices_for_new_label = (labels==i).nonzero()[0]
            labels_in_aggr_view = labels_from_aggregated_clustering[pt_indices_for_new_label]
            if labels_in_aggr_view.max() == -1:
                labels_only_in_frame_view.append(i)
            else:
                obj_labels_in_aggr_view = labels_in_aggr_view[labels_in_aggr_view>-1]
                label_of_majority_aggr = np.bincount(obj_labels_in_aggr_view.astype(int)).argmax()
                labels_frame2aggr_map[i] = label_of_majority_aggr
        
        for i in np.unique(labels_from_aggregated_clustering):
            if i == -1:
                continue
            pt_indices_aggr_label = (labels_from_aggregated_clustering == i).nonzero()[0]
            labels_in_frame_view = labels[pt_indices_aggr_label]
            if labels_in_frame_view.max() == -1:
                labels_only_in_aggr_view.append(i)
        
        # Start fusing aggregated view and new frame view labels
        new_labels = -1 * np.ones(labels.shape[0])

        #Add labels_only_in_frame_view
        if len(labels_only_in_frame_view):
            label_mapping_frame_view = {old_label:i+max_label+1 for i, old_label in enumerate(sorted(labels_only_in_frame_view))}
            for old_lbl, new_lbl in label_mapping_frame_view.items():
                new_labels[labels == old_lbl] = new_lbl
                cluster2frame_id_dict[new_lbl] = [frame_idx]
            max_label = np.max(new_labels)
            #save max cluster id
            path = save_seq_path / 'max_cluster_id.pkl'
            max_cluster_id_dict['max_cluster_id'] = max_label
            with open(path, 'wb') as f:
                pickle.dump(max_cluster_id_dict,f)
            
            # save new cluster2frame_id_dict
            save_path = save_seq_path / 'cluster2frame_id_dict.pkl'
            with open(save_path, 'wb') as f:
                pickle.dump(cluster2frame_id_dict, f)


        for frame_view_lbl, aggr_view_lbl in labels_frame2aggr_map.items():
            pt_indices_this_label = (labels == frame_view_lbl) | (labels_from_aggregated_clustering == aggr_view_lbl)
            new_labels[pt_indices_this_label] = aggr_view_lbl
        
        for aggr_view_lbl in labels_only_in_aggr_view:
            new_labels[labels_from_aggregated_clustering == aggr_view_lbl] = aggr_view_lbl
        
        if show_plots:
            visualize_pcd_clusters(xyz, new_labels.reshape((-1,1)))
    
        # Save labels with new clusters
        save_label_path = save_seq_path / ('%04d.npy' % sample_idx)
        save_labels(new_labels, save_label_path.__str__())

        # Save rejection tag for each pt
        save_path = save_seq_path / 'rejection_tag'/ ('%04d.npy' % sample_idx)
        pt_wise_rejection_tag.astype(np.uint8)
        pt_wise_rejection_tag.tofile(save_path.__str__())

def merge_overlapping_boxes(seq_name, dataset, iou_thresh, method, show_plots=False):
    aggr_pcs_in_world = np.zeros((0,3))
    aggr_ground_mask = np.zeros(0, dtype=np.bool_)
    aggr_labels = np.zeros(0)
    pc_lens = [] 
    aggr_boxes_world_frame = np.zeros((0, 8)) #cxyz, lwh, heading, label

    #Load approx boxes
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f)

    # aggregate infos
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        print(f'sample idx: {sample_idx}')
        
        xyzi = dataset.get_lidar(seq_name, sample_idx)
        xyzi[:,:3] = transform_pc_to_world(xyzi[:,:3], info['pose'])
        pc_lens.append(xyzi.shape[0])
        
        ground_mask = dataset.get_ground_mask(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)

        aggr_ground_mask = np.hstack([aggr_ground_mask, ground_mask])
        aggr_pcs_in_world = np.vstack([aggr_pcs_in_world, xyzi[:,:3]])
        aggr_labels =  np.hstack([aggr_labels, labels])

        det_boxes_in_v = info[f'approx_boxes_{method}']
        cluster_labels_boxes = info['cluster_labels_boxes']
        # for i in range(det_boxes.shape[0]):
        #     box_label = cluster_labels_boxes[i]
        #     if box_label not in unique_cluster_boxes:
        #         unique_cluster_boxes[box_label] = [det_boxes[i]]
        #     else:
        #         unique_cluster_boxes[box_label].append(det_boxes[i])
        pose_v_to_w = info['pose']
        det_boxes_in_w = transform_box(det_boxes_in_v, pose_v_to_w)
        det_boxes_in_w = np.hstack([det_boxes_in_w, cluster_labels_boxes.reshape(-1,1)])
        aggr_boxes_world_frame = np.vstack([aggr_boxes_world_frame, det_boxes_in_w])
    
    
    iou3d_det_det, _, _ = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(aggr_boxes_world_frame[:, 0:7]).float().cuda(), 
                                                         torch.from_numpy(aggr_boxes_world_frame[:, 0:7]).float().cuda())
    
    iou3d = iou3d_det_det.cpu().numpy()

    # max_scores = (iou3d_det_gt.max(dim=0)[0]).cpu().numpy() # for each gt col, find max iou3d across all dets/rows

    map_= {}
    aggr_boxes_unique_labels = np.unique(aggr_boxes_world_frame[:,-1])
    for box_lbl in aggr_boxes_unique_labels:
        boxes_this_lbl_mask = aggr_boxes_world_frame[:,-1] == box_lbl
        boxes_other_lbl_mask = np.logical_not(boxes_this_lbl_mask)

        iou3d_this_box_lbl = iou3d[boxes_this_lbl_mask][:, boxes_other_lbl_mask]
        indices = (iou3d_this_box_lbl > iou_thresh).nonzero()
        if indices[0].shape[0]:
            map_[box_lbl] = {'overlapping_box_cluster_labels': [],
                         'overlapping_box_ious': [],
                         'unique_labels': []}
            rows, cols = indices
            matching_box_indices = boxes_other_lbl_mask.nonzero()[0][cols]
            matching_box_labels = aggr_boxes_world_frame[matching_box_indices][:,-1]
            map_[box_lbl]['overlapping_box_cluster_labels'] = matching_box_labels
            map_[box_lbl]['unique_labels'] = np.unique(matching_box_labels)
            map_[box_lbl]['overlapping_box_ious'] = iou3d_this_box_lbl[rows, cols]
    
    final_map_=[]
    ids_done = set()
    box_lbls_with_multiple_matches = []
    for box_lbl, val in map_.items():
        ids_done.add(box_lbl)
        if len(val['unique_labels']) == 1:
            matching_lbl = val['unique_labels'][0]
            if matching_lbl not in ids_done and len(map_[matching_lbl]['unique_labels']) == 1:
                final_map_.append([box_lbl, matching_lbl])
                ids_done.add(matching_lbl)
        else:
            matching_lbls = val['unique_labels']
            for l in matching_lbls:
                if l not in ids_done:
                    matches = [box_lbl]
                    matches += matching_lbls.tolist()
                    box_lbls_with_multiple_matches.append(matches)
                    break

            for l in matching_lbls:
                ids_done.add(l)
        
        
    
    print(len(final_map_))
    print(len(box_lbls_with_multiple_matches))
    f = np.concatenate(final_map_)
    mm =  np.concatenate(box_lbls_with_multiple_matches)

    if show_plots:
        visualize_selected_labels(aggr_pcs_in_world, aggr_labels, np.concatenate([f, mm]))    
    # visualize_selected_labels(aggr_pcs_in_world, aggr_labels, f)
    # visualize_selected_labels(aggr_pcs_in_world, aggr_labels, mm)
    # print(len(box_lbls_with_multiple_matches))
    # for i in box_lbls_with_multiple_matches:
    #     visualize_selected_labels(aggr_pcs_in_world, aggr_labels, i)

    for m in final_map_:
        lbl = min(m)
        aggr_labels[aggr_labels==m[0]] = lbl
        aggr_labels[aggr_labels==m[1]] = lbl

    for m in box_lbls_with_multiple_matches:
        lbl = min(m)
        for i in m:
            aggr_labels[aggr_labels==i] = lbl

    aggr_labels = get_continuous_labels(aggr_labels)
    if show_plots:
        visualize_pcd_clusters(aggr_pcs_in_world, aggr_labels.reshape(-1,1))
    
    save_seq_path = dataset.label_root_path / seq_name
    i=0
    aggr_labels = aggr_labels.astype(np.float16)
    for info, pc_len in zip(infos, pc_lens):
        sample_idx = info['point_cloud']['sample_idx']
        save_label_path = save_seq_path / ('%04d.npy' % sample_idx)
        print(f'Saving labels: {sample_idx}')

        label_this_pc = aggr_labels[i:i+pc_len]
        i+=pc_len

        label_this_pc.tofile(save_label_path.__str__())




def fit_approx_boxes_seq(seq_name, dataset, show_plots=False, method = 'closeness_to_edge'):
    #Load approx boxes
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'

    try:
        with open(approx_boxes_path, 'rb') as f:
            infos = pickle.load(f)
    except:
        infos= dataset.infos_dict[seq_name]

    

    print(f'Fitting boxes for sequence: {seq_name}')
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']

        #points in current vehicle frame
        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)

        approx_boxes_this_pc = np.empty((0, 18)) #cxyz, lwh, heading, bev_corners.flatten(), frame_info_idx, num_points in this box, label
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_pc = pc[labels==label, :]
            if cluster_pc.shape[0] < 10:
                continue
            box, corners, _ = fit_box(cluster_pc, fit_method=method)
            full_box = np.zeros((1, approx_boxes_this_pc.shape[-1]))
            full_box[0,:7] = box
            full_box[0,7:15] = corners.flatten()
            full_box[0,15] = i # info index
            full_box[0,16] = cluster_pc.shape[0] # num_points
            full_box[0,17] = label
            approx_boxes_this_pc = np.vstack([approx_boxes_this_pc, full_box])
            # [cxy[0], cxy[1], cz, l, w, h, rz, corner0_x, corner0_y, ..., corner3_x, corner3_y,
            # info index, num cluster pts, label]
            # corner0-3 are BEV box corners in lidar frame
        
        # Fitting boxes done for this pc
        #assert np.unique(labels).shape[0] - 1 == approx_boxes_this_pc.shape[0]
        info[f'approx_boxes_{method}'] = approx_boxes_this_pc.astype(np.float32)
        info['cluster_labels_boxes'] = approx_boxes_this_pc[:, -1]


        if show_plots:
            gt_boxes = info['annos']['gt_boxes_lidar']
            show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, 'unrefined_approx_boxes')
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=approx_boxes_this_pc[:,:7], ref_labels=None, ref_scores=None, 
                                color_feature=None, draw_origin=True)
            
    print(f'Fitting boxes Done.')
    save_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)

def refine_boxes_seq(seq_name, dataset, method, show_plots=False):
    
    #Load approx boxes
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f)
    
    approx_boxes = np.empty((0, 18))
    num_boxes_per_pc = np.zeros(len(infos), dtype=int)
    for i, info in enumerate(infos):
        num_boxes_per_pc[i] = info[f'approx_boxes_{method}'].shape[0]
        approx_boxes= np.vstack([approx_boxes, info[f'approx_boxes_{method}']])
    
    #Refine boxes cxyz, lwh, heading, bev_corners.flatten(), label
    refined_boxes = refine_boxes(approx_boxes, approx_boxes_labels=approx_boxes[:,-1])
    print(f'Refining boxes Done.')

    if show_plots:
        ind = 0
        for i, info in enumerate(infos):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            gt_boxes = info['annos']['gt_boxes_lidar']

            #points in current vehicle frame
            pc = dataset.get_lidar(seq_name, sample_idx)
            labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()
            num_boxes_this_pc = int(num_boxes_per_pc[i])
            approx_boxes_this_pc = approx_boxes[ind:ind+num_boxes_this_pc]
            refined_boxes_this_pc = refined_boxes[ind:ind+num_boxes_this_pc]
            ind += num_boxes_this_pc

            gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
        
            for j in range(gt_boxes.shape[0]):
                corners = get_box_corners(gt_boxes[j, :3], gt_boxes[j, 3:6], gt_boxes[j, 6])
                gt_boxes_corners[j, :] = corners.flatten()

            gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])
            
            savefig_path = dataset.label_root_path / seq_name/ ('%04d.png' % sample_idx)
            savefig_path = savefig_path.__str__() #None
            # show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, f'approx_boxes_{method}', \
            #                refined_boxes_this_pc, f'refined_boxes_{method}', gt_boxes, 'gt_boxes',\
            #                 savefig_path=savefig_path)
            # V.draw_scenes(pc, gt_boxes=approx_boxes_this_pc[:,:7], 
            #                     ref_boxes=refined_boxes_this_pc[:,:7]) #gt_boxes=blue, ref_boxes=green
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=refined_boxes_this_pc[:,:7])
    
    #save sequence boxes
    ind = 0
    for i, info in enumerate(infos):
        num_boxes = num_boxes_per_pc[i]
        boxes_this_pc = refined_boxes[ind:ind+num_boxes, :]
        info[f'refined_boxes_{method}'] = boxes_this_pc.astype(np.float32)
        # info['cluster_labels_boxes'] = refined_boxes[ind:ind+num_boxes, -1]
        ind += num_boxes
    
    save_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)


def transform_box(box, pose):
    """Transforms 3d upright boxes from one frame to another.
    Args:
    box: [..., N, 7] boxes.
    from_frame_pose: [...,4, 4] origin frame poses.
    to_frame_pose: [...,4, 4] target frame poses.
    Returns:
    Transformed boxes of shape [..., N, 7] with the same type as box.
    """
    transform = pose 
    heading = box[..., 6] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0]) # heading of obj x/front axis wrt world x axis = rz (angle b/w obj x and ego veh x) + angle between ego veh x and world x axis
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2) # box center wrt ego vehicle frame -> transform wrt world frame

    return np.concatenate([center, box[..., 3:6], heading[..., np.newaxis]], axis=-1)

def convert_detection_to_global_box(det_key, infos, dataset, cluster2frame_id_dict=None):
    for info in infos:
        pose_v_to_w = info['pose']
        box3d_in_v = info[det_key]
        box3d_in_w = transform_box(box3d_in_v, pose_v_to_w)
        cluster_labels = info['cluster_labels_refined_boxes']

        num_box = len(box3d_in_w)

        anno_list =[] # detection list for this pc
        corners3d = box_utils.boxes_to_corners_3d(box3d_in_w[:, :7]) #(num boxes, 8 corners, 3 xyz)
        for i in range(num_box):
            cluster_id = cluster_labels[i]
            bev_corners = get_box_corners(box3d_in_w[i, :3], box3d_in_w[i, 3:6], box3d_in_w[i, -1]) # (4 corners, 2 xy)
            anno = {
                'translation': box3d_in_w[i, :3],
                'lwh': box3d_in_w[i, 3:6],
                'heading': box3d_in_w[i, -1],
                'corners3d': corners3d[i, :, :],
                'bev_corners': bev_corners,
                'box_id': i,
                'cluster_id': cluster_labels[i],

            }

            # Only track boxes ocurring in single frames only
            if cluster2frame_id_dict is not None and len(cluster2frame_id_dict[cluster_id]) < 3:
                anno_list.append(anno)
            
            if cluster2frame_id_dict is None:
                anno_list.append(anno)
        
        info[f'{det_key}_to_track_in_w'] = anno_list #(M, 18)
    
    return infos

def track(infos, dataset, direction='forward', det_key='approx_boxes'):
    last_time_stamp = 1e-6 * infos[0]['metadata']['timestamp_micros'] #first frame time

    tracker = Tracker(max_age=3, max_dist=[3, 5, 8], matcher='greedy')
    
    print(f'Tracking started: {direction}')
    for i, info in enumerate(infos):
        cur_timestamp = 1e-6 * info['metadata']['timestamp_micros']
        time_lag = cur_timestamp - last_time_stamp if direction == 'forward' else last_time_stamp - cur_timestamp
        last_time_stamp = cur_timestamp

        
        ############### For visualization ##################
        seq_name = info['point_cloud']['lidar_sequence']
        sample_idx = info['point_cloud']['sample_idx']
        print(f'sample_idx: {sample_idx}')
        curr_pc = dataset.get_lidar(seq_name, sample_idx)
        curr_pc[:,:3] = transform_pc_to_world(curr_pc[:,:3], info['pose'])
        if i > 0:
            last_pc = dataset.get_lidar(infos[i-1]['point_cloud']['lidar_sequence'], 
                                        infos[i-1]['point_cloud']['sample_idx'])
            last_pc[:,:3] = transform_pc_to_world(last_pc[:,:3], infos[i-1]['pose'])
        else:
            last_pc = curr_pc
        visualize = {'curr_pc': curr_pc, 'last_pc': last_pc}
        ####################################################


        outputs = tracker.step_centertrack(info[f'{det_key}_to_track_in_w'], time_lag, visualize)
        tracking_ids = []
        box_ids = []
        active = [] 
        cluster_ids = []
        for item in outputs:
            if item['active'] == 0: # skip tracks not visible in curr frame/not matched with curr frame i.e. only output tracks visible in curr frame
                continue 
            
            box_ids.append(item['box_id']) # det index from 500 curr detections
            tracking_ids.append(item['tracking_id']) 
            active.append(item['active']) #temporal consistency score = num times this track id is matched consecutively = active - 1
            cluster_ids.append(item['cluster_id'])
        info[f'tracking_result_{direction}'] = {'box_ids': box_ids, 'tracking_ids': tracking_ids, 'active': active,
                                                'cluster_ids': cluster_ids}


    return infos

def track_boxes_seq(seq_name, dataset, method, track_single_occurance_clusters=False):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos
    
    cluster2frame_id_dict = None
    if track_single_occurance_clusters:
        path = dataset.label_root_path / seq_name / 'cluster2frame_id_dict.pkl'
        with open(path, 'rb') as f:
            cluster2frame_id_dict = pickle.load(f) #seq_infos
    
    # path = dataset.label_root_path / seq_name / 'max_cluster_id.pkl'
    # with open(path, 'rb') as f:
    #     max_cluster_id = pickle.load(f) #seq_infos

    det_key = f'refined_boxes_{method}'
    infos = convert_detection_to_global_box(det_key, infos, dataset, cluster2frame_id_dict=cluster2frame_id_dict)
    infos = track(infos, dataset, direction='forward', det_key=det_key)
    infos = track(infos[::-1], dataset, direction='backward', det_key=det_key)

    with open(approx_boxes_path, 'wb') as f:
        pickle.dump(infos[::-1], f)

def get_all_labels(dataset, infos):
    labels = np.empty(0)
    for info in infos:
        pc_info = info['point_cloud']
        seq_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        #points in current vehicle frame
        labels_this_pc = dataset.get_cluster_labels(seq_name, sample_idx)
        labels = np.hstack([labels, labels_this_pc])
    
    return labels.flatten()

def get_tracked_label_this_pc(info, dataset):
    pc_info = info['point_cloud']
    seq_name = pc_info['lidar_sequence']
    sample_idx = pc_info['sample_idx']
    approx_boxes = info[f'approx_boxes_{method}']
    approx_boxes_tracked = approx_boxes[info['tracking_result_forward']['box_ids']]
    approx_boxes_tracked_cluster_ids = info['tracking_result_forward']['cluster_ids']
    tracking_ids = info['tracking_result_forward']['tracking_ids']

    #points in current vehicle frame
    old_labels_this_pc = dataset.get_cluster_labels(seq_name, sample_idx)
    
    tracking_labels_this_pc = -1 * np.ones((old_labels_this_pc.shape[0]))
    for i in range(len(approx_boxes_tracked_cluster_ids)):
        old_box_label = approx_boxes_tracked_cluster_ids[i]
        new_box_label = tracking_ids[i]
        tracking_labels_this_pc[old_labels_this_pc == old_box_label] = new_box_label

    return tracking_labels_this_pc

def visualize_tracks(seq_name, dataset):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos
    
    xyzi_world, pc_lens = dataset.aggregate_pcd_in_world(seq_name)
    tracked_labels_all_pc = np.empty((0,1))
    for i, info in enumerate(infos):
        tracking_labels_this_pc = get_tracked_label_this_pc(info, dataset)
        tracked_labels_all_pc = np.vstack([tracked_labels_all_pc, tracking_labels_this_pc[...,np.newaxis]])

    visualize_pcd_clusters(xyzi_world[:,:3], tracked_labels_all_pc)

def visualize_tracked_boxes(seq_name, dataset, method):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos
    
    for i, info in enumerate(infos):
        # gt_boxes = info['annos']['gt_boxes_lidar']
        # approx_boxes = info[f'approx_boxes_{method}']
        det_boxes = info[f'refined_boxes_{method}']

        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']


        ############### For visualization ##################
        if i > 0:
            last_sample_idx = infos[i-1]['point_cloud']['sample_idx']
            last_pc = dataset.get_lidar(seq_name, last_sample_idx)        
            last_pc[:,:3] = transform_pc_to_world(last_pc[:,:3], infos[i-1]['pose'])
            last_pc_labels = dataset.get_cluster_labels(seq_name,last_sample_idx).flatten()
            last_pc_labelled = last_pc[last_pc_labels>-1]
            print(f'Visualizing Sample idx: {last_sample_idx}-{sample_idx}')

        

            pc = dataset.get_lidar(seq_name, sample_idx)
            pc[:,:3] = transform_pc_to_world(pc[:,:3], info['pose'])
            labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()
            pc_labelled = pc[labels > -1]


            tracked_box_ids = info['tracking_result_forward']['box_ids']
            non_tracked_box_ids = np.in1d(tracked_box_ids, np.arange(det_boxes.shape[0]), invert=True).nonzero()[0]
            tracked_cluster_ids = info['tracking_result_forward']['cluster_ids']
            tracking_ids = info['tracking_result_forward']['tracking_ids']
            #det_boxes_tracked = det_boxes[tracked_box_ids]

            # visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)))
            
            # gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
            
            # for i in range(gt_boxes.shape[0]):
            #     corners = get_box_corners(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
            #     gt_boxes_corners[i, :] = corners.flatten()

            # gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])

            gray = [153/255, 153/255, 153/255]
            
            # Show point clouds curr and last in world frame
            ax.scatter(pc_last[:,0], pc_last[:,1], s=1, color=np.tile(gray,(pc_last.shape[0], 1)), label="pc_last")#np.tile(gray,(pc_last.shape[0], 1))
            ax.scatter(pc_curr[:,0], pc_curr[:,1], s=1, color='cyan', label="pc_curr")#np.tile(gray,(pc_curr.shape[0], 1))
            fig=plt.figure(figsize=(20,20))
            ax = fig.add_subplot(111)
            ax.set_xlabel('x')
            ax.set_ylabel('y')
            ax.scatter(pc_labelled[:,0], pc_labelled[:,1], s=2)
            ax.arrow(0,0,2,0, facecolor='red', linewidth=1, width=0.2) #x-axis
            ax.arrow(0,0,0,2, facecolor='green', linewidth=1, width=0.2) #y-axis
            handles =[]

            det_bev_corners = det_boxes[:, 7:15].reshape((-1,4,2))
            handles.append(Line2D([0], [0], label='det_boxes', color='--k'))
            for i in non_tracked_box_ids:
                draw2DRectangle(ax, det_bev_corners[i].T, color='--k')
            
            for i in tracked_box_ids:
                draw2DRectangle(ax, det_bev_corners[i].T, color='k')
                ax.text(det_boxes[i, 0], det_boxes[i, 1],  f'T {tracking_ids[i]}, C {int(tracked_cluster_ids[i])}', color='black', fontsize = 10, bbox=dict(facecolor='yellow', alpha=0.5))

            ax.legend(handles=handles, fontsize='large', loc='upper right')
            ax.grid()
            # if savefig_path is not None:
            #     plt.savefig(savefig_path)
            # else:
            plt.show()



def get_approx_boxes_tracked(info, method):
    approx_boxes = info[f'approx_boxes_{method}']
    approx_boxes_tracked = approx_boxes[info['tracking_results_forward']['box_ids']]
    tracking_ids = info['tracking_results_forward']['tracking_ids']
    
    # replace old labels with new tracked box labels
    approx_boxes_tracked[:,-1] = tracking_ids

    return approx_boxes_tracked


def refine_tracked_boxes(seq_name, dataset, method, show_plots=False):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos

    approx_boxes_tracked_all_pc = np.empty((0, 18))
    num_boxes_per_pc = np.zeros(len(infos), np.int32)
    for i, info in enumerate(infos):
        approx_boxes_tracked = get_approx_boxes_tracked(info)
        num_boxes_per_pc[i] = approx_boxes_tracked.shape[0]
        approx_boxes_tracked_all_pc = np.vstack([approx_boxes_tracked_all_pc, approx_boxes_tracked])
    
    refined_boxes = refine_boxes(approx_boxes_tracked_all_pc, approx_boxes_labels=approx_boxes_tracked_all_pc[:,-1])

    #save sequence boxes
    ind = 0
    for i, info in enumerate(infos):
        num_boxes = num_boxes_per_pc[i]
        # replace old approx boxes with new tracked approx boxes with tracked labels
        info[f'approx_boxes_{method}'] = approx_boxes_tracked_all_pc[ind:ind+num_boxes, :]
        info[f'refined_boxes_{method}'] = refined_boxes[ind:ind+num_boxes, :]
        ind += num_boxes
    
    save_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)

def remove_outliers_cluster(xyz, labels):
    for i in np.unique(labels):
        if i == -1:
            continue
        cluster_indices = labels==i
        cluster_pc = xyz[cluster_indices]
        cluster_labels = labels[cluster_indices]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(cluster_pc)

        # set as background if point in cluster has less than 4 neighbors within 0.4 m distance
        tree = o3d.geometry.KDTreeFlann(pcd)
        for i in range(cluster_pc.shape[0]):
            [_, idx, _] = tree.search_radius_vector_3d(cluster_pc[i], 0.4)
            if len(idx) < 2:
                cluster_labels[i] = -1
        
        labels[cluster_indices] = cluster_labels
    
    return labels
             

def cluster_seq(seq_name, dataset, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    cluster_files = [] #glob.glob(f'{save_seq_path.__str__()}/*.npy') #[]
    #save_labels_path = save_seq_path / 'all.npy'

    print(f'Clustering of sequence: {seq_name} started!')
    if len(cluster_files) < len(dataset.infos_dict[seq_name]):
        # Aggregate TODO: Use Patchwork to estimate ground
        xyzi_last_vehicle, pc_lens = dataset.aggregate_pcd_in_frame_i(seq_name, -1)
        non_ground_mask = dataset.aggregate_above_plane_mask(seq_name)
        
        print(f'0th Step Aggregation Done.')
        
        # Only cluster non-ground points
        labels = cluster(xyzi_last_vehicle[:,:3], non_ground_mask) 
        # save_labels(labels, save_labels_path.__str__())
        # labels = load_labels(save_labels_path.__str__())
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')

        if show_plots:
            visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))
        
        
        labels, _ = filter_labels(xyzi_last_vehicle[:,:3], labels, labels.max()+1, max_volume=120, estimate_dist_to_plane=False)
        labels = get_continuous_labels(labels)
        print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')

        if show_plots:
            visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))


        # Separate labels pc wise and filter labels
        infos = dataset.infos_dict[seq_name]
        i=0
        num_obj_labels = int(labels.max()+1)
        rejection_tags = np.empty((num_obj_labels, 0))
        for info, pc_len in zip(infos, pc_lens):
            label_this_pc = labels[i:i+pc_len]
            #ground_mask_this_pc = ~non_ground_mask[i:i+pc_len]
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            #print(f'sample idx: {sample_idx}')
            
            #points in current vehicle frame
            this_pc = dataset.get_lidar(seq_name, sample_idx)
            # V.draw_scenes(this_pc[ground_mask_this_pc])
            label_this_pc, rejection_tag = filter_labels(this_pc, label_this_pc, num_obj_labels, max_volume=80, min_volume=0.15) #, ground_mask_this_pc
            rejection_tags = np.hstack([rejection_tags, rejection_tag])
            
            # if sample_idx == 170:
            #     for key,val in REJECT.items():
            #         rejected_labels = np.where(rejection_tag == REJECT[key])[0]
            #         print(f'rejected_labels: {rejected_labels}')
            #         print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
            #         visualize_selected_labels(this_pc, labels[i:i+pc_len], rejected_labels)
            
            labels[i:i+pc_len] = label_this_pc #set -1's to filtered labels
            # if sample_idx >= 160:
            #    visualize_pcd_clusters(this_pc[:,:3], labels[i:i+pc_len].reshape((-1,1)))

            i+=pc_len


        # Keep labels continuous
        labels = get_continuous_labels(labels)
        # print(f'labels found after filtering: {np.unique(labels).shape[0]}')
        print(f'3rd Step Individual PC Filtering Done. Labels: {np.unique(labels).shape[0]}')

        if show_plots:
            visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))      


        labels = fill_in_clusters(labels, xyzi_last_vehicle, show_plots)
        #save_labels(labels, save_labels_path.__str__())

        # Separate labels pc wise and save
        infos = dataset.infos_dict[seq_name]
        i=0
        labels = labels.astype(np.float16)
        for info, pc_len in zip(infos, pc_lens):
            pc_info = info['point_cloud']
            sample_idx = pc_info['sample_idx']
            save_label_path = save_seq_path / ('%04d.npy' % sample_idx)

            label_this_pc = labels[i:i+pc_len]
            i+=pc_len

            label_this_pc.tofile(save_label_path.__str__())

    else:
        print(f'Skipping {seq_name}: Already exists with {len(cluster_files)} files in {save_seq_path}')
    
def cluster_all(dataset, show_plots=False):
    # mp.set_start_method('spawn')
    #Root dir to save labels
    os.makedirs(dataset.label_root_path.__str__(), exist_ok=True)
    num_workers = mp.cpu_count() - 2
    cluster_single_seq = partial(cluster_seq, dataset=dataset, show_plots=show_plots)

    seq_name_list = [seq_name for seq_name in dataset.infos_dict]
    with mp.Pool(num_workers) as p:
        list(tqdm(p.imap(cluster_single_seq, seq_name_list), total=len(seq_name_list)))

def show_bev_boxes(pc, boxes1, label1, boxes2=None, label2=None, boxes3=None, label3=None, savefig_path=None, show_rot=False, iou3d=None):
    fig=plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(pc[:,0], pc[:,1], s=2)
    ax.arrow(0,0,2,0, facecolor='red', linewidth=1, width=0.5) #x-axis
    ax.arrow(0,0,0,2, facecolor='green', linewidth=1, width=0.5) #y-axis
    handles =[]

    bev_corners1 = boxes1[:, 7:15].reshape((-1,4,2))
    handles.append(Line2D([0], [0], label=label1, color='k'))
    for i in range(bev_corners1.shape[0]):
        draw2DRectangle(ax, bev_corners1[i].T, color='k')
        if show_rot:
            ax.text(boxes1[i, 0], boxes1[i, 1],  "{:.2f}".format(np.rad2deg(boxes1[i, 6])), color='black', fontsize = 10, bbox=dict(facecolor='yellow', alpha=0.5))

    if boxes2 is not None:
        handles.append(Line2D([0], [0], label=label2, color='m'))
        bev_corners2 = boxes2[:, 7:15].reshape((-1,4,2))
        for i in range(bev_corners2.shape[0]):   
            draw2DRectangle(ax, bev_corners2[i].T, color='m')
            if iou3d is not None:
                ax.text(boxes2[i, 0]+0.3, boxes2[i, 1]+0.3,  "{:.2f}".format(iou3d[i]), color='black', fontsize = 10, bbox=dict(facecolor='green', alpha=0.5))

    
    if boxes3 is not None:
        bev_corners3 = boxes3[:, 7:15].reshape((-1,4,2))
        handles.append(Line2D([0], [0], label=label3, color='g'))
        for i in range(bev_corners3.shape[0]):   
            draw2DRectangle(ax, bev_corners3[i].T, color='g')
            if show_rot:
                ax.text(boxes3[i, 0]+0.3, boxes3[i, 1]+0.3,  "{:.2f}".format(np.rad2deg(boxes3[i, 6])), color='black', fontsize = 10, bbox=dict(facecolor='green', alpha=0.5))


    ax.legend(handles=handles, fontsize='large', loc='upper right')
    ax.grid()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    else:
        plt.show()

def fit_boxes_all(dataset, show_plots=False):
    num_workers = mp.cpu_count() - 2
    fit_single_seq = partial(fit_approx_boxes_seq, dataset=dataset, show_plots=show_plots)

    seq_name_list = [seq_name for seq_name in dataset.infos_dict]
    with mp.Pool(num_workers) as p:
        list(tqdm(p.imap(fit_single_seq, seq_name_list), total=len(seq_name_list)))

    refine_single_seq = partial(refine_boxes_seq, dataset=dataset, show_plots=show_plots)

    with mp.Pool(num_workers) as p:
        list(tqdm(p.imap(refine_single_seq, seq_name_list), total=len(seq_name_list)))

def get_boxes_in_range_mask(boxes, range):
    mask = (boxes[:,0] < range[0][1]) & (boxes[:,0] > range[0][0]) &\
           (boxes[:,1] < range[1][1]) & (boxes[:,1] > range[1][0])
    
    return mask
def filter_boxes_range_class(gt_boxes, det_boxes, dataset, gt_names, num_points_in_gt, only_close_range, only_class_names):
    min_num_pts_mask = num_points_in_gt > 5
    gt_box_mask = min_num_pts_mask
    dt_box_mask = np.ones(det_boxes.shape[0], np.bool_)
    if only_close_range:
        gt_range_mask = get_boxes_in_range_mask(gt_boxes, range=[[-40, 40],[-30, 30]])
        det_range_mask = get_boxes_in_range_mask(det_boxes, range=[[-40, 40],[-30, 30]])
        dt_box_mask = det_range_mask
        gt_box_mask = gt_box_mask & gt_range_mask
    if only_class_names:
        gt_class_mask = np.array([n in dataset.class_names for n in gt_names], dtype=np.bool_)
        gt_box_mask = gt_box_mask & gt_class_mask
       
    
    gt_boxes = gt_boxes[gt_box_mask]
    det_boxes = det_boxes[dt_box_mask]
    
    return gt_boxes, det_boxes
def eval_sequence(seq_name, dataset, method, only_close_range=False, only_class_names=False):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'

    eval_dict = {}
    for cur_thresh in dataset.iou_eval_thresh_list:
        for metric in dataset.eval_metrics:
            eval_dict[f'tp_{metric}_{cur_thresh}'] = 0
            eval_dict[f'fp_{metric}_{cur_thresh}'] = 0
            eval_dict[f'fn_{metric}_{cur_thresh}'] = 0
    
    print(f'Evaluating seq: {seq_name}')
    with open(approx_boxes_path, 'rb') as f:
        approx_infos = pickle.load(f)
    for info in approx_infos:
        gt_boxes = info['annos']['gt_boxes_lidar']
        det_boxes = info[f'refined_boxes_{method}']
        sample_idx = info['point_cloud']['sample_idx']
        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()

        

        if only_close_range or only_class_names:
            gt_boxes, det_boxes = filter_boxes_range_class(gt_boxes, det_boxes, dataset, info['annos']['name'], info['annos']['num_points_in_gt'], only_close_range, only_class_names)
        else:
            gt_boxes = gt_boxes[info['annos']['num_points_in_gt'] > 10]

        num_gt = gt_boxes.shape[0]
        num_det = det_boxes.shape[0]
        gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
        
        for i in range(gt_boxes.shape[0]):
            corners = get_box_corners(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
            gt_boxes_corners[i, :] = corners.flatten()

        gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])



        iou3d_det_gt, overlapsbev_over_bev_det, overlaps3d_over_vol_det = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(det_boxes[:, 0:7]).float().cuda(), 
                                                       torch.from_numpy(gt_boxes[:, 0:7]).float().cuda())
        
        # print(f'sample_idx: {sample_idx}')
        for cur_thresh in dataset.iou_eval_thresh_list:
            for metric in dataset.eval_metrics:
                if metric == 'iou3d':
                    max_scores = (iou3d_det_gt.max(dim=0)[0]).cpu().numpy() # for each gt col, find max iou3d across all dets/rows
                elif metric == 'overlap3d':
                    max_scores = np.nanmax(overlaps3d_over_vol_det.cpu().numpy(), axis = 0)
                elif metric == 'overlapbev':
                    max_scores = np.nanmax(overlapsbev_over_bev_det.cpu().numpy(), axis = 0)
                
                tp_mask = max_scores >= cur_thresh
                tp_this_pc = tp_mask.sum() 
                tp_gt_indices = tp_mask.nonzero()[0]
                fn_gt_indices = np.logical_not(tp_mask).nonzero()[0]
                fn_iou = max_scores[fn_gt_indices]
                fn_this_pc = num_gt - tp_this_pc
                fp_this_pc = num_det - tp_this_pc
                
                print(f'sample_idx: {sample_idx}, {metric}_{cur_thresh}, \ttp: {tp_this_pc}, \tnum_gt: {num_gt}, \tfn: {fn_this_pc} \tfp:  {fp_this_pc}')

                # show_bev_boxes(pc[labels>-1], det_boxes, 'det_refined_boxes', 
                #                gt_boxes[fn_gt_indices], f'fn_gt_{metric}_{cur_thresh}', 
                #                gt_boxes[tp_gt_indices], 'tp_gt_boxes', show_rot=False,
                #                iou3d=fn_iou)

                
                eval_dict[f'tp_{metric}_{cur_thresh}'] += tp_this_pc
                eval_dict[f'fp_{metric}_{cur_thresh}'] += fp_this_pc
                eval_dict[f'fn_{metric}_{cur_thresh}'] += fn_this_pc

    print(f'Final results for this sequence: {seq_name}')
    for cur_thresh in dataset.iou_eval_thresh_list:
        for metric in dataset.eval_metrics:
            tp = eval_dict[f'tp_{metric}_{cur_thresh}']
            fp = eval_dict[f'fp_{metric}_{cur_thresh}']
            fn = eval_dict[f'fn_{metric}_{cur_thresh}']
            print(f'{metric}_{cur_thresh}: tp: {tp}, fn: {fn}, fp: {fp}, recall: {tp/(tp+fn)}')
                   

    return eval_dict

def eval_all(dataset, only_close_range=False, only_class_names=False):
    # num_workers = mp.cpu_count() - 2
    eval_single_seq = partial(eval_sequence, dataset=dataset, only_close_range=only_close_range, only_class_names=only_class_names)

    # seq_name_list = [seq_name for seq_name in dataset.infos_dict]
    # with mp.Pool(num_workers) as p:
    #     results = list(tqdm(p.imap(eval_single_seq, seq_name_list), total=len(seq_name_list)))
    results=[]
    for seq_name in tqdm(dataset.infos_dict):
        results.append(eval_single_seq(seq_name))

    print('Evaluation Done!')

    results_all_dict = {}
    for cur_thresh in dataset.iou_eval_thresh_list:
        for metric in dataset.eval_metrics:
            results_all_dict[f'tp_{metric}_{cur_thresh}'] = 0
            results_all_dict[f'fp_{metric}_{cur_thresh}'] = 0
            results_all_dict[f'fn_{metric}_{cur_thresh}'] = 0

    for result in results:
        for cur_thresh in dataset.iou_eval_thresh_list:
            for metric in dataset.eval_metrics:
                results_all_dict[f'tp_{metric}_{cur_thresh}'] += result[f'tp_{metric}_{cur_thresh}']
                results_all_dict[f'fp_{metric}_{cur_thresh}'] += result[f'fp_{metric}_{cur_thresh}']
                results_all_dict[f'fn_{metric}_{cur_thresh}'] += result[f'fn_{metric}_{cur_thresh}']

    print(f'Evaluation Results for only_close_range: {only_close_range}, only_class_names: {only_class_names}')
    
    for cur_thresh in dataset.iou_eval_thresh_list:
        for metric in dataset.eval_metrics:
            precision = results_all_dict[f'tp_{metric}_{cur_thresh}'] / (results_all_dict[f'tp_{metric}_{cur_thresh}'] + results_all_dict[f'fp_{metric}_{cur_thresh}'])
            recall = results_all_dict[f'tp_{metric}_{cur_thresh}'] / (results_all_dict[f'tp_{metric}_{cur_thresh}'] + results_all_dict[f'fn_{metric}_{cur_thresh}'])
            print(f'Precision_{metric}_{cur_thresh}: {precision}')
            print(f'Recall_{metric}_{cur_thresh}: {recall}')
            print(results_all_dict[f'tp_{metric}_{cur_thresh}'])
            print(results_all_dict[f'fp_{metric}_{cur_thresh}'])
            print(results_all_dict[f'fn_{metric}_{cur_thresh}'])

def visualize_aggregate_pcd_clusters_in_world(seq_name, dataset, infos=None):
    if infos is None:
        infos = dataset.infos_dict[seq_name]
    xyzi_world, _ = dataset.aggregate_pcd_in_world(seq_name)
    labels = get_all_labels(dataset, infos)
    visualize_pcd_clusters(xyzi_world[:,:3], labels.reshape((-1,1)))

def visualize_aggregate_pcd_clusters_in_frame_i(seq_name, dataset, approx_infos, i=-1):
    xyzi_last_vehicle, _ = dataset.aggregate_pcd_in_frame_i(seq_name, i)
    labels = get_all_labels(dataset, approx_infos)
    visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))

def visualize_seq(seq_name, dataset, method):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        approx_infos = pickle.load(f)

    print(f'Visualizing Sequence: {seq_name}')

    #print(f'Visualizing Clusters in last frame')
    #visualize_aggregate_pcd_clusters_in_frame_i(seq_name, dataset, approx_infos, i=-1)
    
    for i, info in enumerate(approx_infos):
        gt_boxes = info['annos']['gt_boxes_lidar']
        approx_boxes = info[f'approx_boxes_{method}']
        det_boxes = info[f'refined_boxes_{method}']

        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']

        print(f'Visualizing Sample idx: {sample_idx}')

        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()
        visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)))
        
        gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
        
        for i in range(gt_boxes.shape[0]):
            corners = get_box_corners(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
            gt_boxes_corners[i, :] = corners.flatten()

        gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])

        #show_bev_boxes(pc[labels>-1], approx_boxes, f'approx_boxes_{method}', det_boxes, f'refined_boxes_{method}', gt_boxes, 'gt_boxes', show_rot=False)

        #V.draw_scenes(pc, gt_boxes=approx_boxes[:,:7], ref_boxes=det_boxes[:,:7])

        V.draw_scenes(pc, gt_boxes=gt_boxes[:,:7], ref_boxes=det_boxes[:,:7]) #gt_boxes=blue, ref_boxes=green

def visualize_all(dataset):
    for seq_name in dataset.infos_dict:
        visualize_seq(seq_name, dataset)

def eval_clustering(seq_name, dataset):
    infos = dataset.infos_dict[seq_name]
    for frame_idx, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        gt_boxes = info['annos']['gt_boxes_lidar']
        gt_boxes = gt_boxes[info['annos']['num_points_in_gt'] > 5]
        num_gt = gt_boxes.shape[0]
        
        xyzi = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)
        rejection_tags = dataset.get_rejection_tag(seq_name, sample_idx)

        box_wise_point_mask = roiaware_pool3d_utils.points_in_boxes_cpu(
            torch.from_numpy(xyzi[:,:3]), torch.from_numpy(gt_boxes[:, 0:7])).numpy() #(m boxes, n points)
        
        tp = 0
        fn = 0
        fn_dict = {0: 0} # count num gt rejected due to different reasons, 0is the tag for clusters not found by dbscan

        for key, val in REJECT.items():
            fn_dict[val] = 0
        for i in range(num_gt):
            pt_indices = (box_wise_point_mask[i] > 0).nonzero()[0]
            if (labels[pt_indices] > -1).any():
                tp += 1
            else:
                fn += 1
                tag_of_majority_pts = np.bincount(rejection_tags[pt_indices].astype(int)).argmax()
                fn_dict[tag_of_majority_pts] += 1
                
        print(seq_name)
        print(f'sample_idx: {sample_idx}, \ttp: {tp}, fn: {fn}, recall: {tp/(num_gt)}')
        fn_dict_str=f'dbscan_fn: {fn_dict[0]}, '
        for key, val in REJECT.items():
            count = fn_dict[val]
            fn_dict_str += f'{key}: {count} ,'
        print(fn_dict_str)
    

def main():
    dataset = WaymoDataset()

    seq_name = 'segment-10023947602400723454_1120_000_1140_000_with_camera_labels' #Bad
    #dataset.estimate_ground_seq(seq_name)
    # cluster_tracking(seq_name, dataset, initial_guess=False, show_plots=False)
    # fit_approx_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False) #fit using closeness or min max?
    # merge_overlapping_boxes(seq_name, dataset, iou_thresh=0.5,method='naive_min_max', show_p   lots=False) # for some clusters it does not work, maybe try just using fitted boxes?

    # TODO: Some clusters have different/colourful ids although they belong to the same object. If you want all points of the same object to have one cluster id.
    # boxes of different cluster ids but high iou-> make all their cluster points the same id and then refit naive min max and refine
    fit_approx_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False) #fit using closeness or min max?
    refine_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False)
    fit_approx_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=False) #fit using closeness or min max?
    refine_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=False)

    #visualize_aggregate_pcd_clusters_in_world(seq_name, dataset)
    #eval_sequence(seq_name, dataset, only_close_range=False, only_class_names=False)
    visualize_seq(seq_name, dataset, method='naive_min_max')
    
    #cluster_in_aggregated(seq_name, dataset, show_plots=False) #TODO: floating volumes filter signs and small volumes filter fire hydrant
    #cluster_tracking(seq_name, dataset, initial_guess=True, show_plots=False) Above step is not needed if initial guess is false

    #cluster_only(seq_name, dataset, show_plots=True)
    # fit_approx_boxes_seq(seq_name, dataset, show_plots=False, method='naive_min_max') #fit using closeness or min max?
    # refine_boxes_seq(seq_name, dataset, method='naive_min_max', show_plots=False)
    # fit_approx_boxes_seq(seq_name, dataset, show_plots=False, method='closeness_to_edge') #fit using closeness or min max?
    # refine_boxes_seq(seq_name, dataset, method='closeness_to_edge', show_plots=False)

    #track_boxes_seq(seq_name, dataset, track_single_occurance_clusters = False)
    #visualize_tracks(seq_name, dataset)
    #fit_approx_boxes_seq(seq_name, dataset, show_plots=False) #fit using closeness or min max?
    #refine_boxes_seq(seq_name, dataset, show_plots=False)
    #visualize_aggregate_pcd_clusters_in_world(seq_name, dataset)
    #eval_sequence(seq_name, dataset)
    #visualize_seq(seq_name, dataset)

    #cluster_seq_each_pc(seq_name, dataset, show_plots=False)
    
    #from gtboxes containing atleast 5 pts
    #get al points labeled and see how many boxes have these points -> Count tp and fn and calculate recall, count fn due to different rejections tags, not clustered
    # eval_clustering(seq_name, dataset) 

    #
    #     segment-10023947602400723454_1120_000_1140_000_with_camera_labels
    # sample_idx: 50, 180, 120         tp: 59, fn: 5, recall: 0.921875
    # dbscan_fn: 2, too_few_pts: 0 ,vol_too_big: 2 ,vol_too_small: 0 ,floating: 1 ,below_ground
    # TODO: save approx and refined boxes as float32

    #fit_approx_boxes_seq(seq_name, dataset, show_plots=False) #fit using closeness or min max?
    #refine_boxes_seq(seq_name, dataset, show_plots=False)
    #visualize_seq(seq_name, dataset)

    #track_boxes_seq(seq_name, dataset, track_single_occurance_clusters = True)
    




    #seq_name = 'segment-10061305430875486848_1080_000_1100_000_with_camera_labels' #Good
    #cluster_seq(seq_name, dataset=dataset, show_plots=True)
    # fit_approx_boxes(seq_name, dataset, show_plots=False)
    # refine_boxes_seq(seq_name, dataset, show_plots=True)
    #visualize_seq(seq_name, dataset)


    # cluster_all(dataset, show_plots=False)
    # fit_boxes_all(dataset, show_plots=False)

    # eval_all(dataset)
    # eval_all(dataset, only_class_names=True)
    # eval_all(dataset, only_close_range=True)
    # eval_all(dataset, only_close_range=True, only_class_names=True)

    ########################## Tracking###############
    #cluster_seq_each_pc(seq_name, dataset=dataset, show_plots=False)
    #fit_approx_boxes_seq(seq_name, dataset, show_plots=False)
    #track_boxes_seq(seq_name, dataset)
    #visualize_tracks(seq_name, dataset)

    #visualize_all(dataset)
    


if __name__ == '__main__':
    main()
    # ry best is where we find max height -> get ry in current frame
    # filter objects with occurance in less than 3 frames
    # parallelizing
    #TODO: patchwork ground estimation-> remove floating boxes accurately if 1m above closest ground point
    # lidomAug
    # tracking
    # tune filter with other sequences!!!!!



