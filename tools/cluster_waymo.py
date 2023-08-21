# import argparse

import pickle
from pathlib import Path

import numpy as np
from tqdm import tqdm
from third_party.OpenPCDet.tools.approx_bbox_utils import *
from third_party.OpenPCDet.tools.cluster_utils import *
from third_party.OpenPCDet.pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils


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
                print(sequence_name)
            self.infos_dict[sequence_name].append(info)
    
    def get_lidar(self, sequence_name, sample_idx):
        lidar_file = self.data_path / sequence_name / ('%04d.npy' % sample_idx)
        point_features = np.load(lidar_file)  # (N, 7): [x, y, z, intensity, elongation, NLZ_flag]
        point_features[:,3] = np.tanh(point_features[:, 3]) * 255.0
        return point_features[:,:4] #only get xyzi
    
    def get_cluster_labels(self, sequence_name, sample_idx):
        label_file = self.label_root_path / sequence_name / ('%04d.npy' % sample_idx)
        labels = np.fromfile(label_file, dtype=np.float16)
        return labels.reshape((-1,1))
    def save_updated_infos(self):
        infos_list = []
        for seq_name, seq_infos in self.infos_dict.items():
            infos_list += seq_infos
        with open(self.save_infos_pkl_path, 'wb') as f:
            pickle.dump(infos_list,f)
        
    def aggregate_pcd(self, sequence_name):
        infos = self.infos_dict[sequence_name]

        pc_lens = []
        xyz_world = np.empty((0,3))
        intensities = np.empty((0,1))
        non_ground_mask = np.empty((0,1), dtype=bool)
        for info in infos:
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']

            print(sample_idx)
            pose_world_from_vehicle = info['pose']

            #points in current vehicle frame
            xyzi = self.get_lidar(sequence_name, sample_idx)
            xyz=xyzi[:,:3]
            intensity=xyzi[:,-1][...,np.newaxis]
            num_points=xyz.shape[0]
            pc_lens.append(num_points)

            #Remove ground
            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(xyz[:,:3])
            
            #Before ground removal
            #V.draw_scenes(xyz)

            above_plane_mask = np.ones(xyz.shape[0], dtype=bool)
            for i in range(10):
                plane = estimate_plane(xyz, max_hs=0.05, ptc_range=((-70, 70), (-30, 30)))
                if plane is not None:
                    above_plane_mask_this = above_plane(
                        xyzi[:,:3], plane,
                        offset=0.1,
                        only_range=None)
                    above_plane_mask *= above_plane_mask_this
                
                    xyz = xyzi[:,:3][above_plane_mask]
            
            xyz = xyzi[:,:3] # get orginal pc back
            non_ground_mask=np.vstack([non_ground_mask, above_plane_mask[..., np.newaxis]])
            
            #cluster each pc
            # labels_this = clusters_hdbscan(xyz[above_plane_mask], n_clusters=100)[...,np.newaxis]
            # labels = np.ones((xyz.shape[0], 1)) * -1
            # labels[above_plane_mask] = labels_this #(N all points, 1)
            # visualize_pcd_clusters(xyz, labels)

            #V.draw_scenes(xyz[above_plane_mask])
            # Another method of removing ground
            # _, inliers = pcd.segment_plane(distance_threshold=0.25, ransac_n=3, num_iterations=200)
            # mask = np.ones((xyz.shape[0], 1), dtype=bool)
            # mask[above_plane_mask] = False # set ground pts to false
            # non_ground_mask=np.vstack([non_ground_mask, mask])

            # Transform the points from the vehicle frame to the world frame.
            xyz = np.concatenate([xyz, np.ones([num_points, 1])], axis=-1) #(N, xyz1)
            xyz = np.matmul(pose_world_from_vehicle, xyz.T).T[:,:3] #(N, xyz)

            xyz_world = np.vstack([xyz_world, xyz])
            intensities = np.vstack([intensities, intensity])
        
        non_ground_mask = non_ground_mask.flatten()
        # Inv(pose_world_from_vehicle)
        pose_vehicle_from_world = inv_pose(pose_world_from_vehicle)

        # diff=(pose_vehicle_from_world - np.linalg.inv(pose_world_from_vehicle)).sum()
        # Transform the points from the world frame to the last frame in the sequence
        xyz_world =  np.concatenate([xyz_world, np.ones([xyz_world.shape[0], 1])], axis=-1)
        xyz_last_vehicle = np.matmul(pose_vehicle_from_world, xyz_world.T).T[:,:3] #(N, xyz)


        # Append intensity
        xyzi_last_vehicle = np.concatenate([xyz_last_vehicle, intensities], axis=-1)

        # Visualize aggregated pcs of the seq in last frame
        #V.draw_scenes(xyz_last_vehicle)

        # Visualize aggregated pcs of the seq in last frame without ground plane
        #V.draw_scenes(xyz_last_vehicle[:,:3][non_ground_mask])

        return xyzi_last_vehicle, non_ground_mask, pc_lens

def fill_in_clusters(labels, pc, show_plots=False):
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
        
        min_height = np.max([cluster_pc[:,2].min()+0.5,cluster_pc[:,2].min()-0.5])
        top_part_cluster = cluster_pc[cluster_pc[:,2]>min_height]
        
        if top_part_cluster.shape[0] == 0:
            top_part_cluster=cluster_pc
        box, _, _ = fit_box(top_part_cluster, fit_method='closeness_to_edge') #, full_pc = pc[:,:3]
        #shift box up
        box[2] += 0.3
        #append label
        box = np.concatenate((box, [label]), axis = -1)
        boxes = np.vstack([boxes, box])
    
    #fitted boxes
    print(f'4th Step Fit boxes on aggregate PC Done. Boxes: {boxes.shape[0]}')

    if show_plots:
        V.draw_scenes(pc[:,:3], gt_boxes=boxes)

    box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(
    torch.from_numpy(pc[:, 0:3]).unsqueeze(dim=0).float().cuda(),
    torch.from_numpy(boxes[:, 0:7]).unsqueeze(dim=0).float().cuda()
    ).long().squeeze(dim=0).cpu().numpy()

    for i in range(boxes.shape[0]):
        label = boxes[i, -1]
        labels[box_idxs_of_pts == i] = label
    
    print(f'5th Step Clusters filling Done.')

    if show_plots:
        # Filled in clusters
        visualize_pcd_clusters(pc[:,:3], labels.reshape((-1,1)))

    return labels

def cluster_all(dataset, show_plots=False):
    # Cluster
    #Root dir to save labels
    os.makedirs(dataset.label_root_path.__str__(), exist_ok=True)

    # for seq_name in dataset.infos_dict:
    seq_name = 'segment-10061305430875486848_1080_000_1100_000_with_camera_labels' #TODO: remove
    save_seq_path = dataset.label_root_path / seq_name
    os.makedirs(save_seq_path.__str__(), exist_ok=True)
    cluster_files = []#glob.glob(f'{save_seq_path.__str__()}/*.npy') #TODO
    #save_labels_path = save_seq_path / 'all.npy'

    print(f'Clustering of sequence: {seq_name} started!')
    if len(cluster_files) < len(dataset.infos_dict[seq_name]):
        # Aggregate TODO: Use Patchwork to estimate ground
        xyzi_last_vehicle, non_ground_mask, pc_lens = dataset.aggregate_pcd(seq_name)
        
        # Only cluster non-ground points
        labels = cluster(xyzi_last_vehicle[:,:3], non_ground_mask) 
        assert (labels.max() < np.finfo('float16').max), 'max segment id overflow float16 number' 
        # save_labels(labels, save_labels_path.__str__())
        # labels = load_labels(save_labels_path.__str__())
        print(f'1st Step Clustering Done. Labels found: {np.unique(labels).shape[0]}')

        if show_plots:
            visualize_pcd_clusters(xyzi_last_vehicle[:,:3], labels.reshape((-1,1)))
        labels, _ = filter_labels(xyzi_last_vehicle[:,:3], labels, labels.max()+1, estimate_dist_to_plane=False)
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
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            print(f'sample idx: {sample_idx}')
            
            #points in current vehicle frame
            this_pc = dataset.get_lidar(sequence_name, sample_idx)
            # V.draw_scenes(this_pc[ground_mask_this_pc])
            label_this_pc, rejection_tag = filter_labels(this_pc, label_this_pc, num_obj_labels, max_volume=60) #, ground_mask_this_pc
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
        print(f'SKipping {seq_name}: Already exists with {len(cluster_files)} files in {save_seq_path}')

def show_bev_boxes(pc, boxes1, label1, boxes2=None, label2=None):
    fig=plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.scatter(pc[:,0], pc[:,1], s=5)
    bev_corners1 = boxes1[:, 7:15].reshape((-1,4,2))
    draw2DRectangle(ax, bev_corners1[0].T, color='k', label=label1)

    for i in range(bev_corners1.shape[0]):
        draw2DRectangle(ax, bev_corners1[i].T, color='k')

    if boxes2 is not None:
        bev_corners2 = boxes2[:, 7:15].reshape((-1,4,2))
        draw2DRectangle(ax, bev_corners2[0].T, color='m', label=label2)
        for i in range(bev_corners2.shape[0]):   
            draw2DRectangle(ax, bev_corners2[i].T, color='m')

    ax.grid()
    ax.legend()
    plt.show()

def fit_boxes(dataset, show_plots=False):
    # Fit boxes 
    # for seq_name, infos in dataset.infos_dict.items():
        # Approx Bboxes for all pcs this seq
    seq_name = 'segment-10061305430875486848_1080_000_1100_000_with_camera_labels'
    infos= dataset.infos_dict[seq_name]

    print(f'Fitting boxes for sequence: {seq_name}')
    approx_boxes = np.empty((0, 17))
    num_boxes_per_pc = np.zeros(len(infos), dtype=int)
    poses_inv= []
    for i, info in enumerate(infos):
        pc_info = info['point_cloud']
        sequence_name = pc_info['lidar_sequence']
        sample_idx = pc_info['sample_idx']
        pose_w_v = info['pose']
        poses_inv.append(inv_pose(pose_w_v))

        #points in current vehicle frame
        pc = dataset.get_lidar(sequence_name, sample_idx)
        labels = dataset.get_cluster_labels(sequence_name, sample_idx).flatten()

        approx_boxes_this_pc = np.empty((0, 17))
        for label in np.unique(labels):
            if label == -1:
                continue
            cluster_pc = pc[labels==label, :]
            box, corners, _ = fit_box(cluster_pc, fit_method='closeness_to_edge')
            full_box = np.zeros((1, approx_boxes_this_pc.shape[-1]))
            full_box[0,:7] = box
            full_box[0,7:15] = corners.flatten()
            full_box[0,15] = i # info index
            full_box[0,16] = label
            approx_boxes_this_pc = np.vstack([approx_boxes_this_pc, full_box])
            #[cxy[0], cxy[1], cz, l, w, h, rz, corner0_x, corner0_y, ..., corner3_x, corner3_y  area, label]
            #corner0-3 are BEV box corners in lidar frame

        num_boxes_per_pc[i] += int(approx_boxes_this_pc.shape[0])
        approx_boxes = np.vstack([approx_boxes, approx_boxes_this_pc])

        if False: #show_plots:
            gt_boxes = info['annos']['gt_boxes_lidar']
            show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, 'unrefined_approx_boxes')
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=approx_boxes_this_pc[:,:7], ref_labels=None, ref_scores=None, 
                                color_feature=None, draw_origin=True)
            
    print(f'Fitting boxes Done.')

    approx_boxes = np.array(approx_boxes) #(all pcs boxes, 16)
    labels = approx_boxes[:,-1]
    refined_boxes = np.zeros((approx_boxes.shape[0],15))
    poses_inv = np.array(poses_inv) # (M infos, 4, 4)

    for label in np.unique(labels):
        boxes_this_label = approx_boxes[labels==label, :]
        max_l, max_w = np.percentile(boxes_this_label[:, 3], 95), np.percentile(boxes_this_label[:, 4], 95) #TODO: after patchwork, make this max
        
        # to avoid partial boxes floating in air
        max_h = boxes_this_label[:, 5].max()
        cz_in_v=None
        # ind = np.argmax(max_h)
        # cxyz_max_h_in_v2 = boxes_this_label[ind, :3]
        # info_ind = int(boxes_this_label[ind, 15])
        # pose_w_vm = infos[info_ind]['pose']
        # cxyz_max_h_in_w = pose_w_vm @ np.concatenate((cxyz_max_h_in_v2, [1])).reshape((4, -1))
        # poses_v_w = poses_inv[boxes_this_label[:, 15].astype(int), :, :].reshape((-1, 4, 4)) # (M boxes this label, 4, 4)
        # cxyz_max_h_in_v = poses_v_w @ cxyz_max_h_in_w
        # cxyz_max_h_in_v = cxyz_max_h_in_v.reshape((-1, 4))
        # cz_in_v = cxyz_max_h_in_v[:, -2]


        boxes_this_label = refine_boxes(boxes_this_label, max_l, max_w, max_h=max_h, cz_max_h=cz_in_v)
        refined_boxes[labels==label,:]= boxes_this_label
    
    refined_boxes = np.array(refined_boxes)

    print(f'Refining boxes Done.')

    if show_plots:
        ind = 0
        for i, info in enumerate(infos):
            pc_info = info['point_cloud']
            sequence_name = pc_info['lidar_sequence']
            sample_idx = pc_info['sample_idx']
            gt_boxes = info['annos']['gt_boxes_lidar']

            #points in current vehicle frame
            pc = dataset.get_lidar(sequence_name, sample_idx)
            labels = dataset.get_cluster_labels(sequence_name, sample_idx).flatten()
            num_boxes_this_pc = int(num_boxes_per_pc[i])
            approx_boxes_this_pc = approx_boxes[ind:ind+num_boxes_this_pc]
            refined_boxes_this_pc = refined_boxes[ind:ind+num_boxes_this_pc]
            ind += num_boxes_this_pc
            show_bev_boxes(pc[labels>-1], approx_boxes_this_pc, 'approx_boxes', refined_boxes_this_pc, 'refined_boxes')
            # V.draw_scenes(pc, gt_boxes=approx_boxes_this_pc[:,:7], 
            #                     ref_boxes=refined_boxes_this_pc[:,:7]) #gt_boxes=blue, ref_boxes=green
            V.draw_scenes(pc, gt_boxes=gt_boxes, 
                                ref_boxes=refined_boxes_this_pc[:,:7])
    
    ind = 0
    for i, info in enumerate(infos):
        num_boxes = num_boxes_per_pc[i]
        boxes_this_pc = refined_boxes[ind:ind+num_boxes, :]

        #save boxes this pc
        info['approx_boxes'] = boxes_this_pc
        ind += num_boxes

    dataset.save_updated_infos()
def main():
    dataset = WaymoDataset()

    cluster_all(dataset, show_plots=False)
    fit_boxes(dataset, show_plots=True)

if __name__ == '__main__':
    main()
    # ry best is where we find max height -> get ry in current frame
    # parallelizing
    #TODO: patchwork ground estimation-> remove floating boxes if 1m above closest ground point
    # lidomAug
    # tracking

