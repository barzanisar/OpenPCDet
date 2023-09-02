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
        self.iou_eval_thresh_list = [0.5]
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
    

def cluster_seq_each_pc(seq_name, dataset, show_plots=False):
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    cluster_files = [] #glob.glob(f'{save_seq_path.__str__()}/*.npy') #[]
    #save_labels_path = save_seq_path / 'all.npy'

    print(f'Clustering of sequence: {seq_name} started!')
    path = save_seq_path / 'max_cluster_id.pkl'
    with open(path, 'rb') as f:
        max_cluster_id_dict = pickle.load(f)
    max_label = max_cluster_id_dict['max_cluster_id_aggregated']

    path = save_seq_path / 'cluster2frame_id_dict.pkl'
    with open(path, 'rb') as f:
        cluster2frame_id_dict = pickle.load(f)

    if len(cluster_files) < len(dataset.infos_dict[seq_name]):
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
            

            new_labels, rejection_tag  = filter_labels(xyz, labels,
                                      max_volume=160, min_volume=0.1, 
                                      max_height_for_lowest_point=1, 
                                      min_height_for_highest_point=0.5,
                                      ground_mask = ground_mask)
            # if show_plots:
            #     for key, val in REJECT.items():
            #         rejected_labels = np.where(rejection_tag == REJECT[key])[0]
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
                label_mapping_frame_view = {old_label:i+max_label for i, old_label in enumerate(sorted(labels_only_in_frame_view))}
                for old_lbl, new_lbl in label_mapping_frame_view.items():
                    new_labels[labels == old_lbl] = new_lbl
                    cluster2frame_id_dict[new_lbl] = [frame_idx]
                max_label = np.max(labels_only_in_frame_view) + max_label
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



            # flag_pts_labelled_in_frame_view_but_not_in_agg_view = (labels>-1) & (labels_from_aggregated_clustering == -1)

            # if flag_pts_labelled_in_frame_view_but_not_in_agg_view.sum():
            #     new_labels = -1 * np.ones(labels.shape[0])
            #     new_labels[flag_pts_labelled_in_frame_view_but_not_in_agg_view] = labels[flag_pts_labelled_in_frame_view_but_not_in_agg_view]

            #     if show_plots:
            #         # visualize_pcd_clusters(xyz, labels.reshape((-1,1)), boxes=gt_boxes, mode='')
            #         visualize_pcd_clusters(xyz, new_labels.reshape((-1,1)))

            #     # Points labelled in this frame view but unlabelled in aggregated view
            #     # Assign them a new cluster id -> max+1
            #     # (Assign them to cluster label in aggregated view if 90% of them fit in the aggregated view boxes. 
            #     # If they don't fit in any boxes, assign them a new cluster label i.e. max + 1)
            #     #labels[flag_pts_labelled_in_frame_view_but_not_in_agg_view==0] = -1
                
            #     # fill in labels_from_aggregated_clustering with pts from new_labels found close to the labels_from_aggregated_clustering
            #     flag_pts_labelled_aggr =  labels_from_aggregated_clustering>-1
            #     pc_labelled_aggr = xyz[flag_pts_labelled_aggr]
            #     labels_for_pc_labelled_aggr = labels_from_aggregated_clustering[flag_pts_labelled_aggr]
                
            #     pcd_labelled_aggr = o3d.geometry.PointCloud()
            #     pcd_labelled_aggr.points = o3d.utility.Vector3dVector(pc_labelled_aggr)
            #     tree = o3d.geometry.KDTreeFlann(pcd_labelled_aggr)
                
            #     for i in np.unique(new_labels):
            #         if i == -1:
            #             continue
            #         cluster_pt_indices = (new_labels==i).nonzero()[0]
            #         for new_pt_idx in cluster_pt_indices:
            #             [_, idx, _] = tree.search_radius_vector_3d(xyz[new_pt_idx], 0.25)
            #             if len(idx):
            #                 nearest_labels = labels_for_pc_labelled_aggr[np.asarray(idx)]
            #                 label_of_majority = np.bincount(nearest_labels.astype(int)).argmax()
            #                 #Transfer aggregated view cluster label to these unlabelled pts in aggregated view
            #                 labels_from_aggregated_clustering[cluster_pt_indices] = label_of_majority

            #                 # so that new_labels only contains completely new clusters
            #                 new_labels[cluster_pt_indices] = -1
            #                 break
                        
            #     if show_plots:
            #         visualize_pcd_clusters(xyz, new_labels.reshape((-1,1)))
            #         visualize_pcd_clusters(xyz, labels_from_aggregated_clustering.reshape((-1,1)), boxes=gt_boxes, mode='')
                
            #     labels = get_continuous_labels(new_labels)
            #     labels[labels>-1] += max_label
            #     max_label = labels.max()

                #update the cluster2frameid by adding new clusters_id: this frame idx
            

             # Save labels with new clusters
            # flag_pts_labelled_in_frame_view_but_not_in_agg_view = (labels>-1) & (labels_from_aggregated_clustering == -1)
            # labels_from_aggregated_clustering[flag_pts_labelled_in_frame_view_but_not_in_agg_view] = labels[flag_pts_labelled_in_frame_view_but_not_in_agg_view]
            # save_label_path = save_seq_path / ('%04d.npy' % sample_idx)
            # save_labels(labels_from_aggregated_clustering, save_label_path.__str__())    

def fit_approx_boxes_seq(seq_name, dataset, show_plots=False):
    infos= dataset.infos_dict[seq_name]

    print(f'Fitting boxes for sequence: {seq_name}')
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']

        #points in current vehicle frame
        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx)

        approx_boxes_this_pc = np.empty((0, 18))
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_pc = pc[labels==label, :]
            box, corners, _ = fit_box(cluster_pc, fit_method='closeness_to_edge')
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
        info['approx_boxes'] = approx_boxes_this_pc

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
    heading = box[..., -1] + np.arctan2(transform[..., 1, 0], transform[..., 0,
                                                                    0]) # heading of obj x/front axis wrt world x axis = rz (angle b/w obj x and ego veh x) + angle between ego veh x and world x axis
    center = np.einsum('...ij,...nj->...ni', transform[..., 0:3, 0:3],
                    box[..., 0:3]) + np.expand_dims(
                        transform[..., 0:3, 3], axis=-2) # box center wrt ego vehicle frame -> transform wrt world frame

    return np.concatenate([center, box[..., 3:6], heading[..., np.newaxis]], axis=-1)

def convert_detection_to_global_box(det_key, infos):
    for info in infos:
        pose_v_to_w = info['pose']
        box3d_in_v = info[det_key]
        box3d_in_w = transform_box(box3d_in_v, pose_v_to_w)

        num_box = len(box3d_in_w)

        anno_list =[] # detection list for this pc
        for i in range(num_box):
            anno = {
                'translation': box3d_in_w[i, :3],
                'box_id': i 
            }

            anno_list.append(anno)
        info[f'{det_key}_in_w'] = anno_list #(M, 18)
    
    return infos

def track(infos, direction='forward', det_key='approx_boxes'):
    last_time_stamp = 1e-6 * infos[0]['metadata']['timestamp_micros']

    tracker = Tracker(max_age=3, max_dist=1, matcher='greedy')
    
    for info in infos:
        cur_timestamp = 1e-6 * info['metadata']['timestamp_micros']
        time_lag = cur_timestamp - last_time_stamp if direction == 'forward' else last_time_stamp - cur_timestamp
        last_time_stamp = cur_timestamp

        outputs = tracker.step_centertrack(info[f'{det_key}_in_w'], time_lag)
        tracking_ids = []
        box_ids = []
        active = [] 
        for item in outputs:
            if item['active'] == 0: # skip tracks not visible in curr frame/not matched with curr frame i.e. only output tracks visible in curr frame
                continue 
            
            box_ids.append(item['box_id']) # det index from 500 curr detections
            tracking_ids.append(item['tracking_id']) 
            active.append(item['active']) #temporal consistency score = num times this track id is matched consecutively = active - 1
        
        info[f'tracking_result_{direction}'] = {'box_ids': box_ids, 'tracking_ids': tracking_ids, 'active': active}

    return infos

def track_boxes_seq(seq_name, dataset):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos

    det_key = 'approx_boxes'
    infos = convert_detection_to_global_box(det_key, infos)
    infos = track(infos, direction='forward', det_key=det_key)
    infos = track(infos[::-1], direction='backward', det_key=det_key)

    with open(approx_boxes_path, 'wb') as f:
        pickle.dump(infos, f)

def get_all_labels(dataset, infos):
    labels = np.empty((0,1))
    for info in infos:
        pc_info = info['point_cloud']
        seq_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']

        #points in current vehicle frame
        labels_this_pc = dataset.get_cluster_labels(seq_name, sample_idx)
        labels = np.vstack([labels, labels_this_pc])
    
    return labels.flatten()

def get_tracked_label_this_pc(info, dataset):
    pc_info = info['point_cloud']
    seq_name = pc_info['lidar_sequence']
    sample_idx = pc_info['sample_idx']
    approx_boxes = info['approx_boxes']
    approx_boxes_tracked = approx_boxes[info['tracking_result_forward']['box_ids']]
    tracking_ids = info['tracking_result_forward']['tracking_ids']

    #points in current vehicle frame
    old_labels_this_pc = dataset.get_cluster_labels(seq_name, sample_idx)
    
    tracking_labels_this_pc = -1 * np.ones((old_labels_this_pc.shape[0]))
    for i in range(approx_boxes_tracked.shape[0]):
        old_box_label = approx_boxes_tracked[i, 17]
        new_box_label = tracking_ids[i]
        tracking_labels_this_pc[old_labels_this_pc == old_box_label] = new_box_label

    return tracking_labels_this_pc

def visualize_tracks(seq_name, dataset):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f) #seq_infos
    
    xyzi_world, pc_lens = dataset.aggregate_pcd_in_world(seq_name)
    tracked_labels_all_pc = np.empty((0,1))
    for info in infos:
        tracking_labels_this_pc = get_tracked_label_this_pc(info, dataset)
        tracked_labels_all_pc = np.vstack([tracked_labels_all_pc, tracking_labels_this_pc[...,np.newaxis]])

    visualize_pcd_clusters(xyzi_world[:,:3], tracked_labels_all_pc)
        
def get_approx_boxes_tracked(info):
    approx_boxes = info['approx_boxes']
    approx_boxes_tracked = approx_boxes[info['tracking_results_forward']['box_ids']]
    tracking_ids = info['tracking_results_forward']['tracking_ids']
    
    # replace old labels with new tracked box labels
    approx_boxes_tracked[:,-1] = tracking_ids

    return approx_boxes_tracked


def refine_tracked_boxes(seq_name, dataset, show_plots=False):
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
        info['approx_boxes'] = approx_boxes_tracked_all_pc[ind:ind+num_boxes, :]
        info['refined_boxes'] = refined_boxes[ind:ind+num_boxes, :]
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
    

    # labels = cluster(xyz, above_plane_mask, eps=0.2)
    # print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')
    
    # # if show_plots:
    # #     visualize_pcd_clusters(xyz, labels.reshape((-1,1)))

    # # Filter big volumes/smears of moving objects
    # new_labels, rejection_tag  = filter_labels(xyz, labels,
    #                                 max_volume=80, min_volume=0.1, 
    #                                 max_height_for_lowest_point=1, 
    #                                 min_height_for_highest_point=0.5,
    #                                 ground_mask = ground_mask_all)
    
    # if show_plots:
    #     for key, val in REJECT.items():
    #         rejected_labels = np.where(rejection_tag == REJECT[key])[0]
    #         if len(rejected_labels):
    #             print(f'rejected_labels: {rejected_labels}')
    #             print(f'Showing {rejected_labels.shape[0]} rejected labels due to: {key}')
    #             visualize_selected_labels(xyz, labels.flatten(), rejected_labels)
    
    # # save_labels_path = save_seq_path / 'all.npy'
    # # # save_labels(new_labels, save_labels_path.__str__())
    # # labels = load_labels(save_labels_path.__str__())
    
    # labels = remove_outliers_cluster(xyz, labels.flatten())
    # labels = get_continuous_labels(labels)
    # print(f'2nd Step Filtering Done. Labels: {np.unique(labels).shape[0]}')

    # if show_plots:
    #     visualize_pcd_clusters(xyz, labels.reshape((-1,1)))
    

    save_labels_path = save_seq_path / 'all.npy'
    #save_labels(labels, save_labels_path.__str__())
    labels = load_labels(save_labels_path.__str__())

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

def show_bev_boxes(pc, boxes1, label1, boxes2=None, label2=None, boxes3=None, label3=None, savefig_path=None):
    fig=plt.figure(figsize=(20,20))
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(pc[:,0], pc[:,1], s=2)
    ax.arrow(0,0,2,0, facecolor='red', linewidth=1, width=0.5) #x-axis
    ax.arrow(0,0,0,2, facecolor='green', linewidth=1, width=0.5) #y-axis

    bev_corners1 = boxes1[:, 7:15].reshape((-1,4,2))
    draw2DRectangle(ax, bev_corners1[0].T, color='k', label=label1)
    for i in range(bev_corners1.shape[0]):
        draw2DRectangle(ax, bev_corners1[i].T, color='k')

    if boxes2 is not None:
        bev_corners2 = boxes2[:, 7:15].reshape((-1,4,2))
        draw2DRectangle(ax, bev_corners2[0].T, color='m', label=label2)
        for i in range(bev_corners2.shape[0]):   
            draw2DRectangle(ax, bev_corners2[i].T, color='m')
    
    if boxes3 is not None:
        bev_corners3 = boxes3[:, 7:15].reshape((-1,4,2))
        draw2DRectangle(ax, bev_corners3[0].T, color='g', label=label3)
        for i in range(bev_corners3.shape[0]):   
            draw2DRectangle(ax, bev_corners3[i].T, color='g')

    ax.grid()
    ax.legend()
    if savefig_path is not None:
        plt.savefig(savefig_path)
    else:
        plt.show()


def refine_boxes_seq(seq_name, dataset, show_plots=False):
    
    #Load approx boxes
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        infos = pickle.load(f)
    
    approx_boxes = np.empty((0, 18))
    num_boxes_per_pc = np.zeros(len(infos), dtype=int)
    for i, info in enumerate(infos):
        num_boxes_per_pc[i] = info['approx_boxes'].shape[0]
        approx_boxes= np.vstack([approx_boxes, info['approx_boxes']])
    
    #Refine boxes
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
        
            for i in range(gt_boxes.shape[0]):
                corners = get_box_corners(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
                gt_boxes_corners[i, :] = corners.flatten()

            gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])
            
            savefig_path = dataset.label_root_path / seq_name/ ('%04d.png' % sample_idx)
            savefig_path = savefig_path.__str__() #None
            # show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, 'approx_boxes', \
            #                refined_boxes_this_pc, 'refined_boxes', gt_boxes, 'gt_boxes',\
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
        info['refined_boxes'] = boxes_this_pc
        ind += num_boxes
    
    save_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(save_path, 'wb') as f:
        pickle.dump(infos, f)

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
def eval_sequence(seq_name, dataset, only_close_range=True, only_class_names=True):
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
        det_boxes = info['refined_boxes']
        sample_idx = info['point_cloud']['sample_idx']
        num_gt = gt_boxes.shape[0]
        num_det = det_boxes.shape[0]

        if only_close_range or only_class_names:
            gt_boxes, det_boxes = filter_boxes_range_class(gt_boxes, det_boxes, dataset, info['annos']['name'], info['annos']['num_points_in_gt'], only_close_range, only_class_names)
        else:
            gt_boxes = gt_boxes[info['annos']['num_points_in_gt'] > 5]

        iou3d_det_gt, overlapsbev_over_bev_det, overlaps3d_over_vol_det = iou3d_nms_utils.boxes_iou3d_gpu(torch.from_numpy(det_boxes[:, 0:7]).float().cuda(), 
                                                       torch.from_numpy(gt_boxes[:, 0:7]).float().cuda())
        
        # print(f'sample_idx: {sample_idx}')
        for cur_thresh in dataset.iou_eval_thresh_list:
            for metric in dataset.eval_metrics:
                if metric == 'iou3d':
                    tp_this_pc = (iou3d_det_gt.max(dim=0)[0] > cur_thresh).sum().item()
                elif metric == 'overlap3d':
                    tp_this_pc = (overlaps3d_over_vol_det.max(dim=0)[0] > cur_thresh).sum().item()
                elif metric == 'overlapbev':
                    tp_this_pc = (overlapsbev_over_bev_det.max(dim=0)[0] > cur_thresh).sum().item()
                
                fn_this_pc = num_gt - tp_this_pc
                fp_this_pc = num_det - tp_this_pc

                print(f'sample_idx: {sample_idx}, {metric}, \ttp_{cur_thresh}: {tp_this_pc}, \tnum_gt: {num_gt}, \tfn_{cur_thresh}: {fn_this_pc} \tfp_{cur_thresh}:  {fp_this_pc}')
                
                eval_dict[f'tp_{metric}_{cur_thresh}'] += tp_this_pc
                eval_dict[f'fp_{metric}_{cur_thresh}'] += fp_this_pc
                eval_dict[f'fn_{metric}_{cur_thresh}'] += fn_this_pc
            

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

def visualize_aggregate_pcd_clusters_in_world(seq_name, dataset, approx_infos, i=-1):
    xyzi_world, _ = dataset.aggregate_pcd_in_world(seq_name)
    labels = get_all_labels(seq_name, dataset, approx_infos)
    visualize_pcd_clusters(xyzi_world[:,:3], labels.reshape((-1,1)))

def visualize_aggregate_pcd_clusters_in_frame_i(seq_name, dataset, approx_infos, i=-1):
    xyzi_last_vehicle, _ = dataset.aggregate_pcd_in_frame_i(seq_name, i)
    labels = get_all_labels(seq_name, dataset, approx_infos)
    visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))

def visualize_seq(seq_name, dataset):
    approx_boxes_path = dataset.label_root_path / seq_name / 'approx_boxes.pkl'
    with open(approx_boxes_path, 'rb') as f:
        approx_infos = pickle.load(f)

    print(f'Visualizing Sequence: {seq_name}')

    print(f'Visualizing Clusters in last frame')
    visualize_aggregate_pcd_clusters_in_frame_i(seq_name, dataset, approx_infos, i=-1)
    
    for info in approx_infos:
        gt_boxes = info['annos']['gt_boxes_lidar']
        approx_boxes = info['approx_boxes']
        det_boxes = info['refined_boxes']

        pc_info = info['point_cloud']
        sample_idx = pc_info['sample_idx']
        print(f'Sample idx: {sample_idx}')

        pc = dataset.get_lidar(seq_name, sample_idx)
        labels = dataset.get_cluster_labels(seq_name, sample_idx).flatten()
        
        gt_boxes_corners = np.zeros((gt_boxes.shape[0], 8))
        
        for i in range(gt_boxes.shape[0]):
            corners = get_box_corners(gt_boxes[i, :3], gt_boxes[i, 3:6], gt_boxes[i, 6])
            gt_boxes_corners[i, :] = corners.flatten()

        gt_boxes = np.hstack([gt_boxes, gt_boxes_corners])

        show_bev_boxes(pc[labels>-1], approx_boxes, 'approx_boxes', det_boxes, 'refined_boxes', gt_boxes, 'gt_boxes')

        #V.draw_scenes(pc, gt_boxes=approx_boxes[:,:7], ref_boxes=det_boxes[:,:7])

        V.draw_scenes(pc, gt_boxes=gt_boxes[:,:7], ref_boxes=det_boxes[:,:7]) #gt_boxes=blue, ref_boxes=green

def visualize_all(dataset):
    for seq_name in dataset.infos_dict:
        visualize_seq(seq_name, dataset)

def main():
    dataset = WaymoDataset()

    seq_name = 'segment-10023947602400723454_1120_000_1140_000_with_camera_labels' #Bad
    #dataset.estimate_ground_seq(seq_name)
    #cluster_in_aggregated(seq_name, dataset, show_plots=False) #TODO: floating volumes filter signs and small volumes filter fire hydrant
    cluster_seq_each_pc(seq_name, dataset, show_plots=True)

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



