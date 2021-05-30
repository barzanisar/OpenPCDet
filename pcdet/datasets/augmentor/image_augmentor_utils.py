import copy
import numpy as np

# flip image + 3d gt boxes + 2d bounding boxes + depth map
# def random_flip_horizontal(image, depth_map, gt_boxes, gt_boxes2d, calib):
#     """
#     Performs random horizontal flip augmentation
#     Args:
#         image: (H_image, W_image, 3), Image
#         depth_map: (H_depth, W_depth), Depth map
#         gt_boxes: (N, 7), 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
#         gt_boxes2d: (N, 4), 2D box labels in image coordinates [x1, y1, x2, y2]
#         calib: calibration.Calibration, Calibration object
#     Returns:
#         aug_image: (H_image, W_image, 3), Augmented image
#         aug_depth_map: (H_depth, W_depth), Augmented depth map
#         aug_gt_boxes: (N, 7), Augmented 3D box labels in LiDAR coordinates [x, y, z, w, l, h, ry]
#         aug_gt_boxes2d: (N, 4), Augmented 2D box labels in image coordinates [x1, y1, x2, y2]
#
#     """
#     # Randomly augment with 50% chance
#     enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])
#
#     if enable:
#         # Flip images
#         aug_image = np.fliplr(image)
#         aug_depth_map = np.fliplr(depth_map)
#
#         # Flip 3D gt_boxes by flipping the centroids in image space
#         aug_gt_boxes = copy.copy(gt_boxes)
#         locations = aug_gt_boxes[:, :3]
#         img_pts, img_depth = calib.lidar_to_img(locations)
#         W = image.shape[1]
#         img_pts[:, 0] = W - img_pts[:, 0]
#         pts_rect = calib.img_to_rect(u=img_pts[:, 0], v=img_pts[:, 1], depth_rect=img_depth)
#         pts_lidar = calib.rect_to_lidar(pts_rect)
#         aug_gt_boxes[:, :3] = pts_lidar
#         aug_gt_boxes[:, 6] = -1 * aug_gt_boxes[:, 6]
#
#          # Flip 2D GT boxes
#         aug_gt_boxes2d = copy.copy(gt_boxes2d)
#         aug_gt_boxes2d[:, [2, 0]] = W - aug_gt_boxes2d[:, [0, 2]]
#     else:
#         aug_image = image
#         aug_depth_map = depth_map
#         aug_gt_boxes = gt_boxes
#         aug_gt_boxes2d = gt_boxes2d
#
#     return aug_image, aug_depth_map, aug_gt_boxes, aug_gt_boxes2d


# apply fliping only on image and 2d bounding box
def random_flip_horizontal(image, gt_boxes2d):
    """
    Performs random horizontal flip augmentation
    Args:
        image: (H_image, W_image, 3), Image
        gt_boxes2d: (N, 4), 2D box labels in image coordinates [x1, y1, x2, y2]
    Returns:
        aug_image: (H_image, W_image, 3), Augmented image
        aug_gt_boxes2d: (N, 4), Augmented 2D box labels in image coordinates [x1, y1, x2, y2]

    """
    # Randomly augment with 50% chance
    enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

    if enable:
        # Flip images
        aug_image = np.fliplr(image)

        # Flip 2D GT boxes
        W = image.shape[1]
        aug_gt_boxes2d = copy.copy(gt_boxes2d)
        aug_gt_boxes2d[:, [2, 0]] = W - aug_gt_boxes2d[:, [0, 2]]
    else:
        aug_image = image
        aug_gt_boxes2d = gt_boxes2d

    if enable:
        enable = 1
    else:
        enable = 0
    return aug_image, aug_gt_boxes2d, enable
