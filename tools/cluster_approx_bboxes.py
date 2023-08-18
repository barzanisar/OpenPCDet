# import argparse

import numpy as np
from tqdm import tqdm

import numpy as np
import numpy.linalg as LA
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import pickle
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
from pcdet.utils import box_utils, calibration_kitti
import matplotlib.pyplot as plt
from visual_utils.pcd_preprocess import *
from pathlib import Path
from sklearn.linear_model import RANSACRegressor
import torch
from sklearn import cluster

np.random.seed(100)
ROOT_PATH = Path('/home/barza/OpenPCDet/data/kitti')
FOV_POINTS_ONLY = False
SHOW_PLOTS = True
info_path = ROOT_PATH / 'kitti_infos_train_360FOV.pkl'
new_info_path = ROOT_PATH / 'kitti_infos_train_360FOV_close.pkl'
METHOD='closeness'#, 'min_max' 'PCA'
USE_SKLEARN_CLUSTER=False

def get_calib(idx):
    calib_file = ROOT_PATH / 'training' / 'calib' / ('%s.txt' % idx)
    assert calib_file.exists()
    return calibration_kitti.Calibration(calib_file)

def generate_prediction_dicts(info, approx_boxes, pc):
        """
        Args:
        Returns:

        """
        def get_fov_flag(pts_rect, img_shape, calib):
            """
            Args:
                pts_rect:
                img_shape:
                calib:

            Returns:

            """
            pts_img, pts_rect_depth = calib.rect_to_img(pts_rect)
            val_flag_1 = np.logical_and(pts_img[:, 0] >= 0, pts_img[:, 0] < img_shape[1])
            val_flag_2 = np.logical_and(pts_img[:, 1] >= 0, pts_img[:, 1] < img_shape[0])
            val_flag_merge = np.logical_and(val_flag_1, val_flag_2)
            pts_valid_flag = np.logical_and(val_flag_merge, pts_rect_depth >= 0)

            return pts_valid_flag

        num_boxes = approx_boxes.shape[0]

        frame_id = info['point_cloud']['lidar_idx']
        calib = get_calib(frame_id)
        image_shape = info['image']['image_shape']
        pred_boxes_camera = box_utils.boxes3d_lidar_to_kitti_camera(approx_boxes, calib)
        pred_boxes_img = box_utils.boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )

        annos = {
            'name': np.array(['Object']*num_boxes),
            'truncated': np.zeros(num_boxes),
            'occluded': np.zeros(num_boxes), 
            'alpha':  -np.arctan2(-approx_boxes[:, 1], approx_boxes[:, 0]) + pred_boxes_camera[:, 6],
            'bbox': pred_boxes_img, 
            'dimensions': pred_boxes_camera[:, 3:6],
            'location': pred_boxes_camera[:, 0:3], 
            'rotation_y':pred_boxes_camera[:, 6],
            'score': -1*np.ones(num_boxes),
            'difficulty': np.ones(num_boxes),
            'gt_boxes_lidar': approx_boxes,
            'num_points_in_gt': -np.ones(num_boxes, dtype=np.int32)
 
        }  # all moderate

        pts_rect = calib.lidar_to_rect(pc[:, 0:3])

        if FOV_POINTS_ONLY:
            fov_flag = get_fov_flag(pts_rect, info['image']['image_shape'], calib)
            pts_fov = pc[fov_flag]
        else:
            pts_fov = pc
        corners_lidar = box_utils.boxes_to_corners_3d(approx_boxes)

        for k in range(num_boxes):
            flag = box_utils.in_hull(pts_fov[:, 0:3], corners_lidar[k])
            annos['num_points_in_gt'][k] = flag.sum()


        info['annos'] = annos
        

        return info


def drawMinMaxRectangle(x1, y1, x2, y2):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    plt.plot([x1, x2], [y1, y1], color='b', label='min/max box') # -->
    plt.plot([x2, x2], [y1, y2], color='b') # | (up)
    plt.plot([x2, x1], [y2, y2], color='b') # <--
    plt.plot([x1, x1], [y2, y1], color='b') # | (down)

def draw2DRectangle(rectangleCoordinates, color, label):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    plt.plot(rectangleCoordinates[0, 0:2], rectangleCoordinates[1, 0:2], color=color, label=label) # | (up)
    plt.plot(rectangleCoordinates[0, 1:3], rectangleCoordinates[1, 1:3], color=color) # -->
    plt.plot(rectangleCoordinates[0, 2:], rectangleCoordinates[1, 2:], color=color)    # | (down)
    plt.plot([rectangleCoordinates[0, 3], rectangleCoordinates[0, 0]], [rectangleCoordinates[1, 3], rectangleCoordinates[1, 0]], color=color)    # <--

def cluster_pc(pc, min_num=20, dist_thresh=0.05, eps=0.5):

    if USE_SKLEARN_CLUSTER:
        labels = np.zeros(pc.shape[0], dtype=int) - 1
        labels = cluster.DBSCAN(
            eps=eps,
            min_samples=min_num,
            n_jobs=-1).fit(pc).labels_
        
        pc = np.concatenate((pc, labels.reshape(-1, 1)), axis = 1)
    else:
        pc, num_clusters_found = clusterize_pcd(pc, min_num, dist_thresh=dist_thresh, eps=eps) # reduce min samples
    cluster_ids = pc[:,-1]
    assert cluster_ids.max() < np.iinfo(np.int16).max

    if SHOW_PLOTS:
        visualize_pcd_clusters(pc)
    return pc[:,:4], cluster_ids.astype(int)

def min_max_box(cluster_pc):
    #min/max naive bbox
    xc = 0.5*(np.max(cluster_pc[0,:]) + np.min(cluster_pc[0,:]))
    yc = 0.5*(np.max(cluster_pc[1,:]) + np.min(cluster_pc[1,:]))
    dx = np.max(cluster_pc[0,:]) - np.min(cluster_pc[0,:])
    dy = np.max(cluster_pc[1,:]) - np.min(cluster_pc[1,:])

    zc = 0.5*(np.max(cluster_pc[2,:]) + np.min(cluster_pc[2,:]))
    dz = np.max(cluster_pc[2,:]) - np.min(cluster_pc[2,:])#zmax-zmin
    heading = 0 
    box = [xc, yc, zc, dx, dy, dz, heading]
    return box

def PCA_box(cluster_pc):
    #https://logicatcore.github.io/scratchpad/lidar/sensor-fusion/jupyter/2021/04/20/3D-Oriented-Bounding-Box.html
    # cluster_pc 3xN points
    #find cov of xy coords
    cov = np.cov(cluster_pc[0:2, :]) #xy covariance
    eval, evec = LA.eig(cov)

    #sort eigen vals and eigen vecs from largest to smallest
    idx = eval.argsort()[::-1]   
    eval = eval[idx]
    evec = evec[:,idx]
    
    # Check if evecs are orthogonal
    # print(np.rad2deg(np.arccos(np.dot(evec[:,0], evec[:,1]))))
    # print(np.allclose(LA.inv(evec), evec.T))

    # center gt points
    means = np.mean(cluster_pc, axis=1)
    centered_pts = cluster_pc - means[:,np.newaxis]
    
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
        #axis-aligned box dims
    dx = xmax-xmin
    dy = ymax-ymin
    # Find box center
    xc = 0.5*(rectangleCoordinates[0,2] + rectangleCoordinates[0,0])
    yc = 0.5*(rectangleCoordinates[1,2] + rectangleCoordinates[1,0])
    zc = 0.5*(np.max(cluster_pc[2,:]) + np.min(cluster_pc[2,:]))
    dz = np.max(cluster_pc[2,:]) - np.min(cluster_pc[2,:])#zmax-zmin
    
    # Find yaw between body fixed frame (e.vectors) and lidar frame
    heading = np.arctan2(evec[1,0], evec[0,0])

    box = [xc, yc, zc, dx, dy, dz, heading]
    
    return box, rectangleCoordinates, evec, eval

def approximate_boxes_from_gt(pc, gt_boxes):
    point_indices = roiaware_pool3d_utils.points_in_boxes_cpu(
                torch.from_numpy(pc[:, 0:3]), torch.from_numpy(gt_boxes)
            ).numpy()  # (nboxes, npoints)
    gt_corners_lidar = box_utils.boxes_to_corners_3d(gt_boxes)

    err_pca_rot = []
    err_closeness_rot = []
    for i in range(gt_boxes.shape[0]):
        gt_points = pc[point_indices[i] > 0][:, 0:3].T #3xN
        if gt_points.shape[1] > 5:
            box_pca, rec_coords_pca, evec, eval=PCA_box(gt_points)
            corners, ry, area = closeness_rectangle(gt_points[[0, 1], :].T) # directly in LiDAR frame
            heading = ry
            dx = np.linalg.norm(corners[0] - corners[1])
            dy = np.linalg.norm(corners[0] - corners[-1])
            c = (corners[0] + corners[2]) / 2 # center in xz cam axis 

            xc = c[0]
            yc = c[1]
            zc = 0.5*(np.max(gt_points[2,:]) + np.min(gt_points[2,:]))
            dz = np.max(gt_points[2,:]) - np.min(gt_points[2,:])
            box_closeness = [xc, yc, zc, dx, dy, dz, heading]

            err_pca_rot.append((gt_boxes[i, -1]-box_pca[-1])**2)
            err_closeness_rot.append((gt_boxes[i, -1]-heading)**2)

            if SHOW_PLOTS:
                plt.scatter(gt_points[0, :], gt_points[1, :], color='g') 
                
                # Draw PCA box four sides of the axis-aligned bbox
                draw2DRectangle(rec_coords_pca, color='g', label='PCA')
                # plot the eigen vactors scaled by their eigen values
                plt.plot([box_pca[0], box_pca[0] + eval[0] * evec[0, 0]],  [box_pca[1], box_pca[1] + eval[0] * evec[1, 0]], label="e.vec1", color='r')
                plt.plot([box_pca[0], box_pca[0] + eval[1] * evec[0, 1]],  [box_pca[1], box_pca[1] + eval[1] * evec[1, 1]], label="e.vec2", color='g')
                
                # min/max bbox
                drawMinMaxRectangle(np.min(gt_points[0,:]), np.min(gt_points[1,:]), np.max(gt_points[0,:]), np.max(gt_points[1,:]))
                
                # Draw closeness rect
                draw2DRectangle(corners.T, color='r', label='closeness')

                
                
                # Draw closeness rect
                draw2DRectangle(gt_corners_lidar[i].T, color='c', label='Gt')

                
                plt.xlabel('x-lidar')
                plt.ylabel('y-lidar')
                
                title = "Gt Yaw: {:.2f}, Est Yaw PCA atan2: {:.2f}, Est Yaw PCA atan: {:.2f}, Est Yaw closeness: {:.2f}".format(gt_boxes[i, -1]*180/np.pi, np.arctan2(evec[1,0], evec[0,0])*180/np.pi, 
                                                                                    np.arctan(evec[1,0]/ evec[0,0])*180/np.pi, heading*180/np.pi)
                plt.title(title)
                plt.grid()
                plt.legend()
                plt.show()

    return err_pca_rot, err_closeness_rot

def approximate_boxes(pc, cluster_ids, gt_boxes, calib, method='closeness'):
    # Approx Bboxes
    approx_boxes = []

    for cluster_id in np.unique(cluster_ids):
        if cluster_id == -1:
            continue
        indices = cluster_ids == cluster_id
        cluster_pc = pc[indices, :].T # 3xN
        num_gt_pts = cluster_pc.shape[1]
        if num_gt_pts > 5:
            #if method == 'PCA':
            box_pca, rec_coords_pca, evec, eval=PCA_box(cluster_pc)

            corners, ry, area = closeness_rectangle(cluster_pc[[0, 1], :].T) # directly in LiDAR frame
            dx = np.linalg.norm(corners[0] - corners[1])
            dy = np.linalg.norm(corners[0] - corners[-1])
            c = (corners[0] + corners[2]) / 2 # center in xz cam axis 
            # midpoint_w = (corners[0] + corners[-1])/2
            # body_xvec = midpoint_w - c
            heading = ry #np.arctan2(body_xvec[1], body_xvec[0]) #same as ry
        
            xc = c[0]
            yc = c[1]
            zc = 0.5*(np.max(cluster_pc[2,:]) + np.min(cluster_pc[2,:]))
            dz = np.max(cluster_pc[2,:]) - np.min(cluster_pc[2,:])
            box_closeness = [xc, yc, zc, dx, dy, dz, heading]

            if method == 'PCA':
                box = box_pca
            elif method == 'closeness':
                box = box_closeness
            else: # min/max box
                box= min_max_box(cluster_pc)
            
            # filter based on big volume or very big len
            if (box[3]*box[4]*box[5]) > 20:
                continue
            if box[3] > 6 or box[4] > 6 or box[5] > 2:
                continue

            if SHOW_PLOTS:
                plt.scatter(cluster_pc[0, :], cluster_pc[1, :], color='g') 
                
                # Draw PCA box four sides of the axis-aligned bbox
                draw2DRectangle(rec_coords_pca, color='g', label='PCA')
                # plot the eigen vactors scaled by their eigen values
                plt.plot([box_pca[0], box_pca[0] + eval[0] * evec[0, 0]],  [box_pca[1], box_pca[1] + eval[0] * evec[1, 0]], label="e.vec1", color='r')
                plt.plot([box_pca[0], box_pca[0] + eval[1] * evec[0, 1]],  [box_pca[1], box_pca[1] + eval[1] * evec[1, 1]], label="e.vec2", color='g')
                
                # min/max bbox
                drawMinMaxRectangle(np.min(cluster_pc[0,:]), np.min(cluster_pc[1,:]), np.max(cluster_pc[0,:]), np.max(cluster_pc[1,:]))
                
                # Draw closeness rect
                draw2DRectangle(corners.T, color='r', label='closeness')
                
                plt.xlabel('x-lidar')
                plt.ylabel('y-lidar')
                
                title = "Est Yaw PCA atan2: {:.2f}, Est Yaw PCA atan: {:.2f}, Est Yaw closeness: {:.2f}".format(np.arctan2(evec[1,0], evec[0,0])*180/np.pi, 
                                                                                    np.arctan(evec[1,0]/ evec[0,0])*180/np.pi, heading*180/np.pi)
                plt.title(title)
                plt.grid()
                plt.legend()

            approx_boxes.append(box)


            
    if SHOW_PLOTS:
        plt.show()
    approx_boxes = np.array(approx_boxes)
    # for box in approx_boxes:
    #     print('final est vol: ', box[3]*box[4]*box[5])
    if SHOW_PLOTS:
        V.draw_scenes(pc[:,:-1], gt_boxes=gt_boxes, 
                            ref_boxes=approx_boxes, ref_labels=None, ref_scores=None, 
                            color_feature=None, draw_origin=True, 
                            point_features=None)
    return approx_boxes



def get_road_plane(idx):
    plane_file = ROOT_PATH / 'training' / 'planes' / ('%s.txt' % idx)
    if not plane_file.exists():
        return None

    with open(plane_file, 'r') as f:
        lines = f.readlines()
    lines = [float(i) for i in lines[3].split()]
    plane = np.asarray(lines)

    # Ensure normal is always facing up, this is in the rectified camera coordinate
    if plane[1] > 0:
        plane = -plane

    norm = np.linalg.norm(plane[0:3])
    plane = plane / norm
    return plane

def distance_to_plane(ptc, plane, directional=False):
    d = ptc @ plane[:3] + plane[3]
    if not directional:
        d = np.abs(d)
    d /= np.sqrt((plane[:3]**2).sum())
    return d

def above_plane(ptc, plane, offset=0.05, only_range=((-30, 30), (-30, 30))):
    mask = distance_to_plane(ptc, plane, directional=True) < offset
    if only_range is not None:
        range_mask = (ptc[:, 0] < only_range[0][1]) * (ptc[:, 0] > only_range[0][0]) * \
            (ptc[:, 1] < only_range[1][1]) * (ptc[:, 1] > only_range[1][0])
        mask *= range_mask
    return np.logical_not(mask) # returns mask of all non-plane points

def estimate_plane(origin_ptc, max_hs=-1.5, it=1, ptc_range=((-20, 70), (-20, 20))):
    mask = (origin_ptc[:, 2] < max_hs) & \
        (origin_ptc[:, 0] > ptc_range[0][0]) & \
        (origin_ptc[:, 0] < ptc_range[0][1]) & \
        (origin_ptc[:, 1] > ptc_range[1][0]) & \
        (origin_ptc[:, 1] < ptc_range[1][1])
    for _ in range(it):
        ptc = origin_ptc[mask]
        reg = RANSACRegressor().fit(ptc[:, [0, 1]], ptc[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = -1.0
        h = reg.estimator_.intercept_
        norm = np.linalg.norm(w)
        w /= norm
        h = h / norm
        result = np.array((w[0], w[1], w[2], h))
        result *= -1
        mask = np.logical_not(above_plane(
            origin_ptc[:, :3], result, offset=0.2)) # mask of all plane-points
    return result # minus [a,b,c,d] = -plane coeff

def is_valid_cluster(
        ptc,
        plane,
        min_points=10,
        max_volume=20,
        min_volume=0.5,
        max_min_height=1,
        min_max_height=0.5):
    if ptc.shape[0] < min_points:
        return False
    volume = np.prod(ptc.max(axis=0) - ptc.min(axis=0))
    if volume > max_volume or volume < min_volume: # volume too big or small
        return False
    distance_to_ground = distance_to_plane(ptc, plane, directional=True)
    if distance_to_ground.min() > max_min_height: #min distance to plane > 1m, object floating in air
        #V.draw_scenes(ptc[:,:4])
        return False
    if distance_to_ground.max() < min_max_height: #max distance to plane < 0.5, object underground or too small
        #V.draw_scenes(ptc[:,:4])
        return False
    h = ptc[:,2].max() - ptc[:,2].min()
    # if height is not much, make this cluster background
    if h < 0.4:
        return False
    return True

def filter_labels(
    ptc,
    labels
):
    labels = labels.copy()
    plane = estimate_plane(ptc, max_hs=-1.5, ptc_range=((-70, 70), (-50, 50)))
    for i in range(labels.max()+1):
        if not is_valid_cluster(ptc[labels == i, :3], plane):
            labels[labels == i] = -1
        
    label_mapping = sorted(list(set(labels)))
    label_mapping = {x:i for i, x in enumerate(label_mapping)}
    for i in range(len(labels)):
        labels[i] = label_mapping[labels[i]]
    return labels

def closeness_rectangle(cluster_ptc, delta=0.1, d0=1e-2):
    max_beta = -float('inf')
    choose_angle = None
    for angle in np.arange(0, 90+delta, delta): #from 0 to 90 deg, step 0.1
        angle = angle / 180. * np.pi # convert to rad
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ]) #rectangles orthogonal edge directions e1 = [np.cos(angle), np.sin(angle)], e2=[-np.sin(angle), np.cos(angle)]
        projection = cluster_ptc @ components.T #project points to the rectangle's edges Nx2 i.e. C1 = Nx1 and C2 = Nx2
        min_x, max_x = projection[:,0].min(), projection[:,0].max() #boundaries of projections along axis e1 i.e. min/max projection len
        min_y, max_y = projection[:,1].min(), projection[:,1].max() #boundaries of projections along axis e2 i.e. min/max projection len
        Dx = np.vstack((projection[:, 0] - min_x, max_x - projection[:, 0])).min(axis=0) # distance of all point projections to closest corner/boundary of e1
        Dy = np.vstack((projection[:, 1] - min_y, max_y - projection[:, 1])).min(axis=0) # distance of all point projections to closest corner/boundary of e2
        beta = np.vstack((Dx, Dy)).min(axis=0) #smallest distance between projected point and closest rectangle's edge
        beta = np.maximum(beta, d0) 
        beta = 1 / beta # closeness score
        beta = beta.sum()
        if beta > max_beta:
            max_beta = beta
            choose_angle = angle
    angle = choose_angle
    components = np.array([
        [np.cos(angle), np.sin(angle)],
        [-np.sin(angle), np.cos(angle)]
    ])
    projection = cluster_ptc @ components.T
    min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
    min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    if (max_x - min_x) < (max_y - min_y):
        angle = choose_angle + np.pi / 2
        components = np.array([
            [np.cos(angle), np.sin(angle)],
            [-np.sin(angle), np.cos(angle)]
        ])
        projection = cluster_ptc @ components.T
        min_x, max_x = projection[:, 0].min(), projection[:, 0].max()
        min_y, max_y = projection[:, 1].min(), projection[:, 1].max()

    area = (max_x - min_x) * (max_y - min_y)

    rval = np.array([
        [max_x, min_y],
        [min_x, min_y],
        [min_x, max_y],
        [max_x, max_y],
    ]) # corners of rectangle in e1 and e2 i.e. rectangle edges frame i.e e.vectors frame
    rval = rval @ components
    return rval, angle, area

def main():
    with open(info_path, 'rb') as f:
        infos = pickle.load(f)

    new_infos = []
    for i, info in tqdm(enumerate(infos)):
        sample_idx = info['point_cloud']['lidar_idx']

        #cluster_file = ROOT_PATH / 'training_clustered' / 'velodyne' / ('%s.bin' % sample_idx)
        lidar_file = ROOT_PATH / 'training' / 'velodyne' / ('%s.bin' % sample_idx)
        pc = np.fromfile(str(lidar_file), dtype=np.float32).reshape(-1, 4)
        calib = get_calib(sample_idx)
        # if cluster_file.exists() and not CLUSTER_HERE:
        #     cluster_ids = np.fromfile(str(cluster_file), dtype=np.int16).reshape(-1, 1)
        #     cluster_ids = cluster_ids.astype(np.float32)
        #     pc = np.hstack((pc, cluster_ids))
        # else:

        # Approx boxes on GT clusters
        # error_pca_rot = []
        # error_close_rot = []
        # err_pca, err_closeness = approximate_boxes_from_gt(pc, info['annos']['gt_boxes_lidar'])
        # error_pca_rot.append(err_pca)
        # error_close_rot.append(err_closeness)
        # if i > 100:
        #     break

        # Only extract above plane points and points within x=[-70, 70], y=[-40, 40], z=[-3, 1]
        plane = None #get_road_plane(sample_idx)
        if plane is None:
            plane = estimate_plane(pc[:, :3], max_hs=-1.5, ptc_range=[[-70, 70], [-20, 20]])
        above_plane_mask = above_plane(
            pc[:, :3], plane,
            offset=0.05,
            only_range=[[-70, 70], [-20, 20]])
        # range_mask = (pc[:, 0] <= 70) * \
        #     (pc[:, 0] > -70) * \
        #     (pc[:, 1] <= 40) * \
        #     (pc[:, 1] > -40)
        range_mask = (pc[:, 0] <= 40) * \
            (pc[:, 0] > -40) * \
            (pc[:, 1] <= 20) * \
            (pc[:, 1] > -20)
        final_mask = above_plane_mask * range_mask #only above ground points and points in range -40,40 for y and -70,70 for x are clustered
        
        # V.draw_scenes(new_pc[:,:4])
        new_pc = pc[final_mask]
        #V.draw_scenes(pc[:,:4])

        # Cluster above plane points
        new_pc, cluster_labels = cluster_pc(new_pc)

        # Filter clusters
        labels_filtered = filter_labels(
            new_pc, cluster_labels) # makes background as 0th cluster
        
        labels_filtered[labels_filtered==0] = -1
        if SHOW_PLOTS:
            visualize_pcd_clusters(np.hstack((new_pc,labels_filtered.reshape(-1, 1))))

        # Filter labels and boxes further based on volume
        boxes = approximate_boxes(new_pc, labels_filtered, info['annos']['gt_boxes_lidar'], calib, method=METHOD)

        # nms boxes
        if boxes.shape[0] > 0:
            new_info = generate_prediction_dicts(info, boxes, pc)
        new_infos.append(new_info)
    # print('pca err: ', np.mean(np.array(error_pca_rot).flatten()))
    # print('close_err: ', np.mean(np.array(error_close_rot).flatten()))

    # with open(new_info_path, 'wb') as f:    
    #     pickle.dump(new_infos, f)
    
main()

    

    