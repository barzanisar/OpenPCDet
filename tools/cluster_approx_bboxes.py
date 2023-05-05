# import argparse

import numpy as np
# from tqdm import tqdm

import numpy as np
import numpy.linalg as LA
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import pickle
# from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti
import matplotlib.pyplot as plt
from visual_utils.pcd_preprocess import *
from pathlib import Path

np.random.seed(100)
ROOT_PATH = Path('/home/barza/OpenPCDet/data/kitti')
CLUSTER_HERE = True
FILTER_GROUND_CLUSTERS= True
FILTER_WALL_CLUSTERS= True
FILTER_POLES= True
approx_boxes_PCA = False# boxes with PCA
SHOW_PLOTS = False
info_path = ROOT_PATH / 'kitti_infos_train_95_0.pkl'



# def generate_prediction_dicts(frame_info, approx_boxes):
        # """
        # Args:
        # Returns:

        # """
        # def get_calib(idx):
        #     calib_file = ROOT_PATH / 'training' / 'calib' / ('%s.txt' % idx)
        #     assert calib_file.exists()
        #     return calibration_kitti.Calibration(calib_file)

        # def get_template_prediction(num_samples):
        #     ret_dict = {
        #         'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
        #         'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
        #         'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
        #         'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
        #         'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        #     }
        #     return ret_dict

        # def generate_single_sample_dict(batch_index, box_dict):
        #     pred_scores = box_dict['pred_scores'].cpu().numpy()
        #     pred_boxes = box_dict['pred_boxes'].cpu().numpy()
        #     pred_labels = box_dict['pred_labels'].cpu().numpy()
        #     pred_dict = get_template_prediction(pred_scores.shape[0])
        #     if pred_scores.shape[0] == 0:
        #         return pred_dict

        #     calib = batch_dict['calib'][batch_index]
        #     image_shape = batch_dict['image_shape'][batch_index].cpu().numpy()
        #     pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        #     pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
        #         pred_boxes_camera, calib, image_shape=image_shape
        #     )

        #     pred_dict['name'] = np.array(['Object'])[pred_labels - 1]
        #     pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        #     pred_dict['bbox'] = pred_boxes_img
        #     pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        #     pred_dict['location'] = pred_boxes_camera[:, 0:3]
        #     pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        #     pred_dict['score'] = pred_scores
        #     pred_dict['boxes_lidar'] = pred_boxes

        #     return pred_dict

       
        # frame_id = info['point_cloud']['lidar_idx']
        # calib = get_calib(frame_id)
        # num_boxes = approx_boxes.shape[0]
        # annos = {
        #     'name': np.array(['Object']*num_boxes),
        #     'truncated': np.zeros(num_boxes),
        #     'occluded': np.zeros(num_boxes), 
        #     'alpha': np.zeros(num_boxes),
        #     'bbox': np.zeros([num_boxes, 4]), 
        #     'dimensions': np.zeros([num_boxes, 3]),
        #     'location': np.zeros([num_boxes, 3]), 
        #     'rotation_y': np.zeros(num_boxes),
        #     'score': np.zeros(num_boxes),
        #     'difficulty': np.ones(num_boxes),
        #     'index': np.array(list(range(num_objects))),
        #     'gt_boxes_lidar': approx_boxes,
        #     'num_points_in_gt': 

        # }


        

        # return new_info


def draw2DRectangle(x1, y1, x2, y2):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    plt.plot([x1, x2], [y1, y1], color='b') # -->
    plt.plot([x2, x2], [y1, y2], color='b') # | (up)
    plt.plot([x2, x1], [y2, y2], color='b') # <--
    plt.plot([x1, x1], [y2, y1], color='b') # | (down)

def filter_ground(indices, pc):
    object = pc[indices, :]
    z = object[:,2]
    h = z.max() - z.min()
    # if height is not much, make this cluster background
    if h < 0.4:
        pc[indices, -1] = -1
    if FILTER_POLES and h > 2.5:
        pc[indices, -1] = -1
    return pc

def filter_walls(indices, pc):
    object = pc[indices, :]
    x = object[:,0]
    y = object[:,1]
    l = x.max() - x.min()
    w = y.max() - y.min()
    
    if l > 6 or w > 6:
        pc[indices, -1] = -1
        object_clustered, num_clusters_found = clusterize_pcd(object[:,:-1], 5, dist_thresh=0.1, eps=0.3)
        #visualize_pcd_clusters(object_clustered)
        if num_clusters_found > 1:
            cluster_ids = object_clustered[:, -1] 
            for cluster_id in np.unique(cluster_ids):
                if cluster_id == -1:
                    continue
                mini_obj_indices = cluster_ids == cluster_id 
                if FILTER_GROUND_CLUSTERS:
                    object_clustered = filter_ground(mini_obj_indices, object_clustered)
                if FILTER_WALL_CLUSTERS:
                    mini_object = object_clustered[mini_obj_indices, :]
                    x = mini_object[:,0]
                    y = mini_object[:,1]
                    l = x.max() - x.min()
                    w = y.max() - y.min()
                    if l > 6 or w > 6:
                        object_clustered[mini_obj_indices, -1] = -1
            
            #visualize_pcd_clusters(object_clustered)
            cluster_ids = object_clustered[:, -1]
            if cluster_ids.max() > -1:
                object_clustered[cluster_ids > -1,-1] += pc[:,-1].max()
                pc[indices, :] = object_clustered
                return pc
            
    return pc

def cluster(pc):
    pc, num_clusters_found = clusterize_pcd(pc, 100, dist_thresh=0.1, eps=0.5) # reduce min samples
    cluster_ids = pc[:,-1]
    assert cluster_ids.max() < np.iinfo(np.int16).max
    if len(info['annos']['name']) > 0:
        assert num_clusters_found > 1

    # Filter unnecessary clusters: ground, walls
    for cluster_id in np.unique(cluster_ids):
        if cluster_id == -1:
            continue
        indices = cluster_ids == cluster_id
        cluster_pc = pc[indices, :]
        if FILTER_GROUND_CLUSTERS:
            pc = filter_ground(indices, pc)
        if FILTER_WALL_CLUSTERS:
            pc= filter_walls(indices, pc)
 
    visualize_pcd_clusters(pc)
    return pc


def approximate_boxes(pc, gt_boxes):
    # Approx Bboxes
    cluster_ids = pc[:,-1]
    num_clusters = len(np.unique(cluster_ids))
    approx_boxes = []

    for cluster_id in np.unique(cluster_ids):
        if cluster_id == -1:
            continue
        indices = cluster_ids == cluster_id
        cluster_pc = pc[indices, :].T # 3xN

        gt_points_np = cluster_pc # 3xN

        num_gt_pts = gt_points_np.shape[1]
        if num_gt_pts > 5:
            #find cov of xy coords
            cov = np.cov(gt_points_np[0:2, :]) #xy covariance
            eval, evec = LA.eig(cov)

            #sort evals and evecs from largest to smallest
            idx = eval.argsort()[::-1]   
            eval = eval[idx]
            evec = evec[:,idx]
            
            # Check if evecs are orthogonal
            # print(np.rad2deg(np.arccos(np.dot(evec[:,0], evec[:,1]))))
            # print(np.allclose(LA.inv(evec), evec.T))

            # center gt points
            means = np.mean(gt_points_np, axis=1)
            centered_pts = gt_points_np - means[:,np.newaxis]
            
            ### Rotate the points i.e. align the eigen vector to the cartesian basis
            rot = lambda theta: np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
            theta = np.arctan(evec[1,0]/evec[0,0]) # radians
            aligned_pts = np.matmul(rot(-theta), centered_pts[:2,:]) 

            # min/max bbox for axis aligned points
            xmin, xmax, ymin, ymax = np.min(aligned_pts[0, :]), np.max(aligned_pts[0, :]), np.min(aligned_pts[1, :]), np.max(aligned_pts[1, :])
            
            # Rotate points to align with the body frame i.e. e.vectors (back to their original orientation)
            realigned_pts =  np.matmul(rot(theta), aligned_pts) 

            # Translate points back to their original position
            realigned_pts += means[:2, np.newaxis]

            #Find axis-aligned min/max bbox corners
            rectCoords = lambda x1, y1, x2, y2: np.array([[x1, x2, x2, x1],
                                    [y1, y1, y2, y2]])
            rectangleCoordinates = rectCoords(xmin, ymin, xmax, ymax)

            # Rotate and translate the axis-aligned min/max bbox corners to align with body frame i.e. e.vectors
            rectangleCoordinates = np.matmul(rot(theta), rectangleCoordinates) #np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
            rectangleCoordinates += means[:2, np.newaxis]


            # Find box dims
            if approx_boxes_PCA:
                #axis-aligned box dims
                dx = xmax-xmin
                dy = ymax-ymin
                # Find box center
                xc = 0.5*(rectangleCoordinates[0,2] + rectangleCoordinates[0,0])
                yc = 0.5*(rectangleCoordinates[1,2] + rectangleCoordinates[1,0])
                
            else:
                #min/max naive bbox
                xc = 0.5*(np.max(gt_points_np[0,:]) + np.min(gt_points_np[0,:]))
                yc = 0.5*(np.max(gt_points_np[1,:]) + np.min(gt_points_np[1,:]))
                dx = np.max(gt_points_np[0,:]) - np.min(gt_points_np[0,:])
                dy = np.max(gt_points_np[1,:]) - np.min(gt_points_np[1,:])

            zc = 0.5*(np.max(gt_points_np[2,:]) + np.min(gt_points_np[2,:]))
            dz = np.max(gt_points_np[2,:]) - np.min(gt_points_np[2,:])#zmax-zmin
            
            # Find yaw between body fixed frame (e.vectors) and lidar frame
            if approx_boxes_PCA:
                heading = np.arctan2(evec[1,0], evec[0,0])
            else:
                heading = 0 

            box = [xc, yc, zc, dx, dy, dz, heading]
            est_vol = dx*dy*dz
            print('est_vol ', est_vol)
            if est_vol > 20:
                continue
            if dx > 6 or dy > 6 or dz > 2:
                continue
            #plot histogram of zc
            approx_boxes.append(box)
            

            if SHOW_PLOTS:

                plt.scatter(realigned_pts[0, :], realigned_pts[1, :], label='realigned')
                #plt.scatter(gt_points_np[0, :], gt_points_np[1, :], label='gt points')

                # four sides of the axis-aligned bbox
                plt.plot(rectangleCoordinates[0, 0:2], rectangleCoordinates[1, 0:2], color='g') # | (up)
                plt.plot(rectangleCoordinates[0, 1:3], rectangleCoordinates[1, 1:3], color='g') # -->
                plt.plot(rectangleCoordinates[0, 2:], rectangleCoordinates[1, 2:], color='g')    # | (down)
                plt.plot([rectangleCoordinates[0, 3], rectangleCoordinates[0, 0]], [rectangleCoordinates[1, 3], rectangleCoordinates[1, 0]], color='g')    # <--
                # plot the eigen vactors scaled by their eigen values
                plt.plot([xc, xc + eval[0] * evec[0, 0]],  [yc, yc + eval[0] * evec[1, 0]], label="e.vec1", color='r')
                plt.plot([xc, xc + eval[1] * evec[0, 1]],  [yc, yc + eval[1] * evec[1, 1]], label="e.vec2", color='g')
                plt.xlabel('x')
                plt.ylabel('y')
                # min/max bbox
                draw2DRectangle(np.min(gt_points_np[0,:]), np.min(gt_points_np[1,:]), np.max(gt_points_np[0,:]), np.max(gt_points_np[1,:]))
                title = "Est Yaw: {:.2f}, Est Yaw2: {:.2f}".format(np.arctan2(evec[1,0], evec[0,0])*180/np.pi, 
                                                                                    np.arctan(evec[1,0]/ evec[0,0])*180/np.pi)
                plt.title(title)
                plt.show()


    approx_boxes = np.array(approx_boxes)
    for box in approx_boxes:
        print('final est vol: ', box[3]*box[4]*box[5]) 
    V.draw_scenes(pc[:,:-1], gt_boxes=gt_boxes, 
                        ref_boxes=approx_boxes, ref_labels=None, ref_scores=None, 
                        color_feature=None, draw_origin=True, 
                        point_features=None)
    return approx_boxes


with open(info_path, 'rb') as f:
    infos = pickle.load(f)

for info in infos:
    sample_idx = info['point_cloud']['lidar_idx']
    cluster_file = ROOT_PATH / 'training_clustered' / 'velodyne' / ('%s.bin' % sample_idx)
    lidar_file = ROOT_PATH / 'training' / 'velodyne' / ('%s.bin' % sample_idx)
    pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
    # if cluster_file.exists() and not CLUSTER_HERE:
    #     cluster_ids = np.fromfile(str(cluster_file), dtype=np.int16).reshape(-1, 1)
    #     cluster_ids = cluster_ids.astype(np.float32)
    #     pc = np.hstack((pc, cluster_ids))
    # else:
    pc = cluster(pc)
    boxes = approximate_boxes(pc, info['annos']['gt_boxes_lidar'])
    b=1
    #new_info = generate_prediction_dicts(info, boxes)
   
    

    

    