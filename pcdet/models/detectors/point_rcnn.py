from .detector3d_template import Detector3DTemplate

# import numpy as np
# import numpy.linalg as LA
# from lib.LiDAR_snow_sim.tools.visual_utils import open3d_vis_utils as V
# import torch
# import pickle
# from sklearn.manifold import TSNE
# from umap import UMAP
# from ...ops.roiaware_pool3d import roiaware_pool3d_utils
# import matplotlib.pyplot as plt
# import seaborn as sns

# def draw2DRectangle(x1, y1, x2, y2):
#     # diagonal line
#     # plt.plot([x1, x2], [y1, y2], linestyle='dashed')
#     # four sides of the rectangle
#     plt.plot([x1, x2], [y1, y1], color='b') # -->
#     plt.plot([x2, x2], [y1, y2], color='b') # | (up)
#     plt.plot([x2, x1], [y2, y2], color='b') # <--
#     plt.plot([x1, x1], [y2, y1], color='b') # | (down)
# def draw3DRectangle(ax, x1, y1, z1, x2, y2, z2):
#     # the Translate the datatwo sets of coordinates form the apposite diagonal points of a cuboid
#     ax.plot([x1, x2], [y1, y1], [z1, z1], color='b') # | (up)
#     ax.plot([x2, x2], [y1, y2], [z1, z1], color='b') # -->
#     ax.plot([x2, x1], [y2, y2], [z1, z1], color='b') # | (down)
#     ax.plot([x1, x1], [y2, y1], [z1, z1], color='b') # <--

#     ax.plot([x1, x2], [y1, y1], [z2, z2], color='b') # | (up)
#     ax.plot([x2, x2], [y1, y2], [z2, z2], color='b') # -->
#     ax.plot([x2, x1], [y2, y2], [z2, z2], color='b') # | (down)
#     ax.plot([x1, x1], [y2, y1], [z2, z2], color='b') # <--
    
#     ax.plot([x1, x1], [y1, y1], [z1, z2], color='b') # | (up)
#     ax.plot([x2, x2], [y2, y2], [z1, z2], color='b') # -->
#     ax.plot([x1, x1], [y2, y2], [z1, z2], color='b') # | (down)
#     ax.plot([x2, x2], [y1, y1], [z1, z2], color='b') # <--

# def visualize_tsne(pointwise_gt_cls, tsne):

#     class_names = ['background', 'car', 'ped', 'cyc']
#     num_classes = len(class_names)
#     # We choose a color palette with seaborn.
#     palette = np.array(sns.color_palette("hls", num_classes))

#     # scale and move the coordinates so they fit [0; 1] range
#     def scale_to_01_range(x):
#         # compute the distribution range
#         value_range = (np.max(x) - np.min(x))

#         # move the distribution so that it starts from zero
#         # by extracting the minimal value from all its values
#         starts_from_zero = x - np.min(x)

#         # make the distribution fit [0; 1] by dividing by its range
#         return starts_from_zero / value_range

#     # extract x and y coordinates representing the positions of the images on T-SNE plot
#     tx = tsne[:, 0]
#     ty = tsne[:, 1]

#     tx = scale_to_01_range(tx)
#     ty = scale_to_01_range(ty)

#     # initialize a matplotlib plot
#     fig = plt.figure()
#     ax = fig.add_subplot(111)

#     # for every class, we'll add a scatter plot separately
#     for idx, label in enumerate(class_names):
#         # extract the coordinates of the points of this class only
#         current_tx = tx[pointwise_gt_cls == idx]
#         current_ty = ty[pointwise_gt_cls == idx]

#         if idx == 0:
#             continue
#         # add a scatter plot with the corresponding color and label
#         ax.scatter(current_tx, current_ty, label=label, marker='x', linewidth=0.5)

#     # build a legend using the labels we set previously
#     ax.legend(loc='best')
#     plt.grid()
#     plt.xlabel("x1")
#     plt.ylabel("x2")
#     plt.title(f"TSNE on point features")
#     plt.show()

class PointRCNN(Detector3DTemplate):
    def __init__(self, model_cfg, num_class, dataset):
        super().__init__(model_cfg=model_cfg, num_class=num_class, dataset=dataset)
        self.module_list = self.build_networks()

    def forward(self, batch_dict):
        for cur_module in self.module_list:
            batch_dict = cur_module(batch_dict)

            # pc1_len = int(batch_dict['points'].shape[0]/2)
            # classes = ['back', 'car', 'ped', 'cyc']
            # if 'point_features' in batch_dict:
            #     # Find pointwise gt class
            #     gt_boxes = batch_dict['gt_boxes'][0]

            #     num_obj = gt_boxes.shape[0]
            #     point_indices = roiaware_pool3d_utils.points_in_boxes_gpu(batch_dict['points'][:pc1_len,1:4].unsqueeze(dim=0), gt_boxes[:,:-1].unsqueeze(dim=0)
            #     ).long().squeeze(dim=0).cpu().numpy()  # (npoints) -1 for backgrnd, 0 for 1st gt box, ..., num boxes-1

            #     pointwise_gt_cls = np.zeros(pc1_len)
            #     approx_boxes = np.zeros(gt_boxes.shape)#torch.zeros_like(gt_boxes)
            #     for i in range(num_obj):
            #         cls = int(gt_boxes[i, -1])
            #         gt_points = batch_dict['points'][:pc1_len,1:4][point_indices==i]
            #         gt_points_np = gt_points.cpu().numpy().T # 3xN
            #         #xyz_center_app = gt_points[:,:3].mean(dim=0)

            #         means = np.mean(gt_points_np, axis=1)
            #         #cov = np.cov(gt_points_np) #xyz covariance

            #         num_gt_pts = gt_points_np.shape[1]
            #         if num_gt_pts > 0:
            #             cov = np.cov(gt_points_np[0:2, :]) #xy covariance
            #             eval, evec = LA.eig(cov)
            #             idx = eval.argsort()[::-1]   
            #             eval = eval[idx]
            #             evec = evec[:,idx]
            #             print('i: ', i)
            #             print(eval, evec)
            #             # print(np.rad2deg(np.arccos(np.dot(evec[:,0], evec[:,1]))))
            #             centered_data = gt_points_np - means[:,np.newaxis]
            #             # print(np.allclose(LA.inv(evec), evec.T))
            #             ### Rotate the data i.e. align the eigen vector to the cartesian basis
            #             rot = lambda theta: np.array([[np.cos(np.deg2rad(theta)), -np.sin(np.deg2rad(theta))],
            #                     [np.sin(np.deg2rad(theta)),  np.cos(np.deg2rad(theta))]])
            #             theta = np.rad2deg(np.arctan(evec[1,0]/evec[0,0]))
            #             aligned_coords = np.matmul(rot(-theta), centered_data[:2,:]) 
            #             xmin, xmax, ymin, ymax = np.min(aligned_coords[0, :]), np.max(aligned_coords[0, :]), np.min(aligned_coords[1, :]), np.max(aligned_coords[1, :])

            #             realigned_coords =  np.matmul(rot(theta), aligned_coords) #np.matmul(evec, aligned_coords)
            #             realigned_coords += means[:2, np.newaxis]

            #             rectCoords = lambda x1, y1, x2, y2: np.array([[x1, x2, x2, x1],
            #                                     [y1, y1, y2, y2]])
            #             rectangleCoordinates = rectCoords(xmin, ymin, xmax, ymax)
            #             rectangleCoordinates = np.matmul(rot(theta), rectangleCoordinates) #np.matmul(evec, rectCoords(xmin, ymin, zmin, xmax, ymax, zmax))
            #             rectangleCoordinates += means[:2, np.newaxis]
            #             #plt.title("Re rotated and translated data")
            #             # plt.ylim(-6, 8)
            #             # plt.xlim(-6, 8)
            #             dx = xmax-xmin
            #             dy = ymax-ymin
            #             dz = np.max(gt_points_np[2,:]) - np.min(gt_points_np[2,:])#zmax-zmin
            #             r = np.arctan2(evec[1,0], evec[0,0])
            #             print(f'Gt label: {classes[cls]} est_vol: {dx*dz*dy}, dz: {dz}, eval_x/eval_y: {eval[0]/eval[1]}')
            #             xc = 0.5*(rectangleCoordinates[0,2] + rectangleCoordinates[0,0])
            #             yc = 0.5*(rectangleCoordinates[1,2] + rectangleCoordinates[1,0])
            #             zc = 0.5*(np.max(gt_points_np[2,:]) + np.min(gt_points_np[2,:]))
            #             approx_boxes[i,:3] = np.array([xc, yc, zc]) #means #center
            #             approx_boxes[i,3] = dx #len
            #             approx_boxes[i,4] = dy #width
            #             approx_boxes[i,5] = dz #height
            #             approx_boxes[i,6] = r #yaw
            #             pointwise_gt_cls[point_indices==i] = cls
                

            #             print('\n')
            #             print('i: ', i)
            #             print('center: ', means)
            #             print('yaw deg: ', np.arctan2(evec[1,0], evec[0,0])*180/np.pi)
            #             print('lens: ', dx, dy, dz)

            #             print('i: ', i)
            #             print('gt_box: ', gt_boxes[i, :])
            #             print('yaw deg: ', gt_boxes[i, -2]*180/np.pi)
            #             print('\n')

            #             plt.scatter(realigned_coords[0, :], realigned_coords[1, :], label='realigned')
            #             #plt.scatter(gt_points_np[0, :], gt_points_np[1, :], label='gt points')

            #             # four sides of the rectangle
            #             plt.plot(rectangleCoordinates[0, 0:2], rectangleCoordinates[1, 0:2], color='g') # | (up)
            #             plt.plot(rectangleCoordinates[0, 1:3], rectangleCoordinates[1, 1:3], color='g') # -->
            #             plt.plot(rectangleCoordinates[0, 2:], rectangleCoordinates[1, 2:], color='g')    # | (down)
            #             plt.plot([rectangleCoordinates[0, 3], rectangleCoordinates[0, 0]], [rectangleCoordinates[1, 3], rectangleCoordinates[1, 0]], color='g')    # <--
            #             # plot the eigen vactors scaled by their eigen values
            #             plt.plot([xc, xc + eval[0] * evec[0, 0]],  [yc, yc + eval[0] * evec[1, 0]], label="e.vec1", color='r')
            #             plt.plot([xc, xc + eval[1] * evec[0, 1]],  [yc, yc + eval[1] * evec[1, 1]], label="e.vec2", color='g')
            #             plt.xlabel('x')
            #             plt.ylabel('y')
            #             # min/max bbox
            #             draw2DRectangle(np.min(gt_points_np[0,:]), np.min(gt_points_np[1,:]), np.max(gt_points_np[0,:]), np.max(gt_points_np[1,:]))
            #             title = "Est Yaw: {:.2f}, Est Yaw2: {:.2f}, Gt Yaw: {:.2f}".format(np.arctan2(evec[1,0], evec[0,0])*180/np.pi, 
            #                                                                                np.arctan(evec[1,0]/ evec[0,0])*180/np.pi, 
            #                                                                                gt_boxes[i, -2]*180/np.pi)
            #             plt.title(title)
            #             plt.show()


                # V.draw_scenes(batch_dict['points'][:pc1_len,1:], gt_boxes=gt_boxes, 
                #               ref_boxes=approx_boxes, ref_labels=None, ref_scores=None, 
                #               color_feature=None, draw_origin=True, 
                #               point_features=None)

                # #Compute tsne
                # point_features=batch_dict['point_features'][:pc1_len, :]
                # if isinstance(point_features, torch.Tensor):
                #     point_features = point_features.cpu().numpy()
                # tsne = TSNE(n_components=2).fit_transform(point_features)
                # pickle.dump(tsne, open("tsne.pkl", "wb"))

                # #load tsne
                # tsne = pickle.load(open("tsne.pkl", "rb"))
                # visualize_tsne(pointwise_gt_cls, tsne)
                # V.draw_scenes(batch_dict['points'][:pc1_len,1:], gt_boxes=batch_dict['gt_boxes'][0], 
                #               ref_boxes=None, ref_labels=None, ref_scores=None, 
                #               color_feature=None, draw_origin=True, 
                #               point_features=tsne.flatten())


        if self.training:
            loss, tb_dict, disp_dict = self.get_training_loss()

            ret_dict = {
                'loss': loss,
                'batch_dict': batch_dict
            }
            return ret_dict, tb_dict, disp_dict
        else:
            pred_dicts, recall_dicts = self.post_processing(batch_dict) #gt_boxes=batch_dict['gt_boxes'][0]
            # V.draw_scenes(batch_dict['points'][:pc1_len,1:], gt_boxes=approx_boxes, 
            #                   ref_boxes=pred_dicts[0]['pred_boxes'], ref_labels=pred_dicts[0]['pred_labels'], ref_scores=pred_dicts[0]['pred_scores'], 
            #                   color_feature=None, draw_origin=True, 
            #                   point_features=None)
            return pred_dicts, recall_dicts

    def get_training_loss(self):
        disp_dict = {}
        loss = 0
        tb_dict = None
        if self.point_head is not None:
            loss_point, tb_dict = self.point_head.get_loss()
            loss += loss_point
        if self.roi_head is not None:
            loss_rcnn, tb_dict = self.roi_head.get_loss(tb_dict)
            loss += loss_rcnn
        return loss, tb_dict, disp_dict
