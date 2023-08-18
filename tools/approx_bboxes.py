import argparse

import numpy as np
#import torch
from tqdm import tqdm

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import build_dataset
from pcdet.models import load_data_to_gpu
import numpy as np
import numpy.linalg as LA
from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
import torch
import pickle
# from sklearn.manifold import TSNE
# from umap import UMAP
from pcdet.ops.roiaware_pool3d import roiaware_pool3d_utils
import matplotlib.pyplot as plt
# import seaborn as sns

def draw2DRectangle(x1, y1, x2, y2):
    # diagonal line
    # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
    # four sides of the rectangle
    plt.plot([x1, x2], [y1, y1], color='b') # -->
    plt.plot([x2, x2], [y1, y2], color='b') # | (up)
    plt.plot([x2, x1], [y2, y2], color='b') # <--
    plt.plot([x1, x1], [y2, y1], color='b') # | (down)

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--cfg_file', type=str, default=None, help='specify the config for training')

    args = parser.parse_args()

    cfg_from_yaml_file(args.cfg_file, cfg)

    np.random.seed(1024)
    return args, cfg


def main():
    args, cfg = parse_config()

    classes = ['back', 'car', 'ped', 'cyc']
  
    dataset_cfg=cfg.DATA_CONFIG
    dataset_train = build_dataset(dataset_cfg, cfg.CLASS_NAMES, root_path=None,
                  logger=None, training=False)
    dataset_val = build_dataset(dataset_cfg, cfg.CLASS_NAMES, root_path=None,
                  logger=None, training=False)
    
    axis_aligned = True
    show_plots = False

    hist_dict = {'car': None, 'ped': None, 'cyc': None}
    for key, val in hist_dict.items():
        hist_dict[key] = {'est_dx_dy_dz': [], 'num_pts': []}
    
    for idx, data_dict in tqdm(enumerate(dataset_train)):
        data_dict = dataset_train.collate_batch([data_dict])
        load_data_to_gpu(data_dict)
        
        # Find pointwise gt class
        gt_boxes = data_dict['gt_boxes'][0]
        points = data_dict['points'][:,1:4]

        num_obj = gt_boxes.shape[0]
        point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(dim=0), gt_boxes[:,:-1].unsqueeze(dim=0)).long().squeeze(dim=0).cpu().numpy()  # (npoints) -1 for backgrnd, 0 for 1st gt box, ..., num boxes-1

        approx_boxes = np.zeros(gt_boxes.shape)
        for i in range(num_obj):
            cls = int(gt_boxes[i, -1])
            gt_points = points[point_indices==i]
            gt_points_np = gt_points.cpu().numpy().T # 3xN

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
                aligned_pts = np.matmul(rot(-theta), centered_pts[:2,:]) #same as evec.T @ centered_pts[:2,:]

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
                if axis_aligned:
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
                if axis_aligned:
                    heading = np.arctan2(evec[1,0], evec[0,0])
                else:
                    heading = 0 


                approx_boxes[i,:3] = np.array([xc, yc, zc]) #means #center
                approx_boxes[i,3] = dx #len
                approx_boxes[i,4] = dy #width
                approx_boxes[i,5] = dz #height
                approx_boxes[i,6] = heading #yaw

                # hist_dict[classes[cls]]['est_vol'].append(dx*dz*dy) 
                # hist_dict[classes[cls]]['est_height'].append(dz)  
                # hist_dict[classes[cls]]['eval_ratio'].append(eval[0]/eval[1])
                # hist_dict[classes[cls]]['xyarea_height_ratio'].append((dx*dy)/(dx*dz))
                hist_dict[classes[cls]]['est_dx_dy_dz'].append([dx,dz,dy])  
                hist_dict[classes[cls]]['num_pts'].append(num_gt_pts) 
                

                if show_plots:
                    print('\n')
                    print('i: ', i)
                    print(f'Gt label: {classes[cls]} est_vol: {dx*dz*dy}, dz: {dz}, eval_x/eval_y: {eval[0]/eval[1]}')
                    print('est center: ', means)
                    print('est lens: ', dx, dy, dz)
                    print('gt_box: ', gt_boxes[i, :])
                    print('gt yaw deg: ', gt_boxes[i, -2]*180/np.pi)
                    print('est yaw deg: ', np.arctan2(evec[1,0], evec[0,0])*180/np.pi)
                    print('\n')

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
                    title = "Est Yaw: {:.2f}, Est Yaw2: {:.2f}, Gt Yaw: {:.2f}".format(np.arctan2(evec[1,0], evec[0,0])*180/np.pi, 
                                                                                        np.arctan(evec[1,0]/ evec[0,0])*180/np.pi, 
                                                                                        gt_boxes[i, -2]*180/np.pi)
                    plt.title(title)
                    plt.show()
       
        # V.draw_scenes(data_dict['points'][:,1:], gt_boxes=gt_boxes, 
        #                       ref_boxes=approx_boxes, ref_labels=None, ref_scores=None, 
        #                       color_feature=None, draw_origin=True, 
        #                       point_features=None)

    with open('bbox_hist.pkl', 'wb') as f:
        pickle.dump(hist_dict, f)


if __name__ == '__main__':
    main()
    # with open('bbox_hist.pkl', 'rb') as f:
    #     hist_dict = pickle.load(f)

    # for cls in ['car', 'ped', 'cyc']:
    #     plt.figure()
    #     plt.hist2d(hist_dict[cls]['est_height'], hist_dict[cls]['est_vol'], bins=(100, 100), cmap = plt.cm.jet)#
    #     plt.xlabel('est_height')
    #     plt.ylabel('est_vol')
    #     plt.title(cls)
    # plt.show()
    
    # for key in ['est_dx_dy_dz']:
    #     for cls in ['car', 'ped', 'cyc']:
    #         val = np.array(hist_dict[cls][key])
    #         # val = val[np.isfinite(val)]
    #         # print(val.max())
    #         #plt.figure()
    #         val = val[:,:2].max(axis=1)
    #         n, bins, patches = plt.hist(x=val,
    #                             bins=np.linspace(start=0, stop=val.max(), num=50 + 1, endpoint=True),
    #                             alpha=0.7, rwidth=0.85, label=f'{cls}_{key}')
    #         plt.grid(axis='y', alpha=0.75)
    #         plt.xlabel(key)
    #         plt.ylabel(f'Number of samples')
    #         plt.legend()
    #     plt.show()
    
        
    for key in ['num_pts']:
        for cls in ['car', 'ped', 'cyc']:
            val = np.array(hist_dict[cls][key])
            n, bins, patches = plt.hist(x=val,
                                bins=np.linspace(start=0, stop=val.max(), num=50 + 1, endpoint=True),
                                alpha=0.7, rwidth=0.85, label=f'{cls}_{key}')
            plt.grid(axis='y', alpha=0.75)
            plt.xlabel(key)
            plt.ylabel(f'Number of samples')
            plt.legend()
        plt.show()
