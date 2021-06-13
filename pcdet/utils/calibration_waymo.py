import numpy as np


class CameraParameters(object):
    def __init__(self, cam_params: {}):
        # Camera intrinsics
        # 1d Array of [f_u, f_v, c_u, c_v, k {1, 2}, p{1, 2}, k{3}].
        self.fu = cam_params['intrinsics'][0]
        self.fv = cam_params['intrinsics'][1]
        self.cu = cam_params['intrinsics'][2]
        self.cv = cam_params['intrinsics'][3]
        self.K = np.array([[self.fu, 0,       self.cu],
                           [0,       self.fv, self.cv],
                           [0,       0,       1],
                           ])
        # Camera distortion params
        self.k1 = cam_params['intrinsics'][4]
        self.k2 = cam_params['intrinsics'][5]
        self.p1 = cam_params['intrinsics'][6]
        self.p2 = cam_params['intrinsics'][7]
        self.k3 = cam_params['intrinsics'][8]
        # Camera extrinsics
        self.C2V = cam_params['extrinsics'] # 4 * 4


class Calibration(CameraParameters):
    # Note: for WAYMO, saved pts_lidar are in vehicle frame not lidar frame
    # Note: for WAYMO, intrinsics do not include baseline as camera frame is at center of lens
    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3) # waymo lidar pts are in vehicle frame
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        v2c = np.linalg.inv(self.C2V)
        pts_rect = pts_lidar_hom @ v2c.T
        return pts_rect[:, 0:3]

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        # project camera points to normalized image plane
        u_d = - pts_rect[:, 1] / pts_rect[:, 0]
        v_d = - pts_rect[:, 2] / pts_rect[:, 0]
        pts_norm_plane = np.concatenate((u_d.reshape((-1, 1)),
                                           v_d.reshape((-1, 1)),
                                           np.ones(u_d.shape[0]).reshape((-1, 1))), axis=1)

        # project normalized points on camera pixel coordinate frame
        pts_img = pts_norm_plane @ self.K.T
        pts_rect_depth = np.sqrt(pts_rect[:, 0]**2 + pts_rect[:, 1]**2 + pts_rect[:, 2]**2)
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu
        y = ((v - self.cv) * depth_rect) / self.fv
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect

    def corners3d_to_img_boxes(self, corners3d):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        # 3*3 to 3*4 to be similar to kitti (waymo does not need baseline shift)
        K = np.zeros((3, 4), dtype=np.float32)
        K[0:3, 0:3] = self.K
        img_pts = np.matmul(corners3d_hom, K.T)  # (N, 8, 3)

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner