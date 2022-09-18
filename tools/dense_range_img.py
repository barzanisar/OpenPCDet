import numpy as np
import tensorflow as tf
import cv2
from visual_utils import open3d_vis_utils as V
from sklearn.linear_model import RANSACRegressor
import matplotlib.pyplot as plt

#https://www.wevolver.com/specs/hdl-64e.lidar.sensor
beam_inclination_max = np.radians(2.0)
beam_inclination_min = np.radians(-24.9)


#https://www.manualslib.com/download/1988532/Velodyne-Hdl-64e-S3.html
horizontal_res = 3e-3 #np.radians(0.1728) #3e-3 in snow sim 
HEIGHT = 64 #64  channels
WIDTH = np.ceil(np.radians(360)/horizontal_res).astype(int)

# 5x5 diamond kernel
DIAMOND_KERNEL_5 = np.array(
    [
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [1, 1, 1, 1, 1],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
    ], dtype=np.uint8)

lidar_path = '/home/barza/OpenPCDet/data/dense/lidar_hdl64_strongest/2018-02-04_12-13-33_00100.bin'

points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)

def filter_below_groundplane(pointcloud, tolerance=1):
    valid_loc = (pointcloud[:, 2] < -1.4) & \
                (pointcloud[:, 2] > -1.86) & \
                (pointcloud[:, 0] > 0) & \
                (pointcloud[:, 0] < 40) & \
                (pointcloud[:, 1] > -15) & \
                (pointcloud[:, 1] < 15)
    pc_rect = pointcloud[valid_loc]
    print(pc_rect.shape)
    if pc_rect.shape[0] <= pc_rect.shape[1]:
        w = [0, 0, 1]
        h = -1.55
    else:
        reg = RANSACRegressor().fit(pc_rect[:, [0, 1]], pc_rect[:, 2])
        w = np.zeros(3)
        w[0] = reg.estimator_.coef_[0]
        w[1] = reg.estimator_.coef_[1]
        w[2] = 1.0
        h = reg.estimator_.intercept_
        w = w / np.linalg.norm(w)

        print(reg.estimator_.coef_)
        print(reg.get_params())
        print(w, h)
    height_over_ground = np.matmul(pointcloud[:, :3], np.asarray(w))
    height_over_ground = height_over_ground.reshape((len(height_over_ground), 1))
    above_ground = np.matmul(pointcloud[:, :3], np.asarray(w)) - h > -tolerance
    print(above_ground.shape)
    return np.hstack((pointcloud[above_ground, :], height_over_ground[above_ground]))

def _combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.

    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.

    Args:
        tensor: A tensor of any type.

    Returns:
        A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def scatter_nd_with_pool(index,
                         value,
                         shape,
                         pool_method=tf.math.unsorted_segment_max):
    """Similar as tf.scatter_nd but allows custom pool method.

    tf.scatter_nd accumulates (sums) values if there are duplicate indices.

    Args:
        index: [N, 2] tensor. Inner dims are coordinates along height (row) and then
        width (col).
        value: [N, ...] tensor. Values to be scattered.
        shape: (height,width) list that specifies the shape of the output tensor.
        pool_method: pool method when there are multiple points scattered to one
        location.

    Returns:
        image: tensor of shape with value scattered. Missing pixels are set to 0.
    """
    if len(shape) != 2:
        raise ValueError('shape must be of size 2')
    height = shape[0]
    width = shape[1]
    # idx: [N]
    index_encoded, idx = tf.unique(index[:, 0] * width + index[:, 1])
    value_pooled = pool_method(value, idx, tf.size(input=index_encoded))
    index_unique = tf.stack(
        [index_encoded // width,
        tf.math.mod(index_encoded, width)], axis=-1)
    shape = [height, width]
    value_shape = _combined_static_and_dynamic_shape(value)
    if len(value_shape) > 1:
        shape = shape + value_shape[1:]

    image = tf.scatter_nd(index_unique, value_pooled, shape)
    return image

def compute_inclinations(beam_inclination_min, beam_inclination_max):
    """Computes uniform inclination range based on the given range and height i.e. number of channels.
    """
    diff = beam_inclination_max - beam_inclination_min
    ratios = (0.5 + np.arange(0, HEIGHT)) / HEIGHT #[0.5, ..., 63.5]
    inclination = ratios * diff + beam_inclination_min #[bottom row inclination, ..., top row]
    #reverse
    inclination = inclination[::-1]
    return inclination

def compute_range_image_polar(range_image, inclination):
    """Computes range image polar coordinates.

    Args:
        range_image: [H, W] tensor. Lidar range images.
        inclination: [H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row of the range image.

    Returns:
        range_image_polar: [H, W, 3] polar coordinates.
    """
    # [W].
    ratios = (np.arange(WIDTH, 0, -1) - 0.5) / WIDTH # [0.99, ..., 0.000]
    
    # [W].
    azimuth = (ratios * 2. - 1.) * np.pi # [180 deg in rad, ..., -180]

    # [H, W]
    azimuth_tile = np.tile(azimuth.reshape((1, WIDTH)), (HEIGHT, 1))
    # [H, W]
    inclination_tile = np.tile(inclination.reshape((HEIGHT, 1)), (1, WIDTH))

    #[H, W, 3]
    range_image_polar = np.stack([azimuth_tile, inclination_tile, range_image],
                                 axis=-1)
    return range_image_polar
def compute_range_image_cartesian(range_image_polar):

    """Computes range image cartesian coordinates from polar ones.

    Args:
        range_image_polar: [B, H, W, 3] float tensor. Lidar range image in polar
        coordinate in sensor frame.
        extrinsic: [B, 4, 4] float tensor. Lidar extrinsic.
        pixel_pose: [B, H, W, 4, 4] float tensor. If not None, it sets pose for each
        range image pixel.
        frame_pose: [B, 4, 4] float tensor. This must be set when pixel_pose is set.
        It decides the vehicle frame at which the cartesian points are computed.
        dtype: float type to use internally. This is needed as extrinsic and
        inclination sometimes have higher resolution than range_image.
        scope: the name scope.

    Returns:
        range_image_cartesian: [B, H, W, 3] cartesian coordinates.
    """
    azimuth, inclination, range_image_range = range_image_polar[:,:,0], range_image_polar[:,:,1], range_image_polar[:,:,2]

    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    cos_incl = np.cos(inclination)
    sin_incl = np.sin(inclination)

    # [H, W].
    x = cos_azimuth * cos_incl * range_image_range
    y = sin_azimuth * cos_incl * range_image_range
    z = sin_incl * range_image_range

    # [H, W, 3]
    range_image_points = np.stack([x, y, z], -1)
    
    return range_image_points

def extract_point_cloud_from_range_image(range_image, inclination):
    """Extracts point cloud from range image.

    Args:
        range_image: [H, W] tensor. Lidar range images.
        inclination: [H] tensor. Inclination for each row of the range image.
        0-th entry corresponds to the 0-th row (top row) of the range image.

    Returns:
        range_image_cartesian: [H, W, 3] with {x, y, z} as inner dims in vehicle
        frame.
    """
    range_image_polar = compute_range_image_polar(range_image,  inclination)
    range_image_cartesian = compute_range_image_cartesian(range_image_polar)
    return range_image_cartesian

def build_range_image_from_point_cloud(points, inclination, point_features=None):
    """Build virtual range image from point cloud assuming uniform azimuth.

    Args:
    points: tf tensor with shape [B, N, 3]
    inclination: tf tensor of shape [B, H] that is the inclination angle per
        row. sorted from highest value to lowest.
    point_features: If not None, it is a tf tensor with shape [B, N, 2] that
        represents lidar 'intensity' and 'elongation'.

    Returns:
    range_images : [B, H, W, 3] or [B, H, W] tensor. Range images built from the
        given points. Data type is the same as that of points_vehicle_frame. 0.0
        is populated when a pixel is missing.
    ri_indices: tf int32 tensor [B, N, 2]. It represents the range image index
        for each point.
    ri_ranges: [B, N] tensor. It represents the distance between a point and
        sensor frame origin of each point.
    """

    # [B, N]
    xy_norm = np.linalg.norm(points[:, 0:2], axis=-1)
    # [B, N]
    point_inclination = np.arctan2(points[..., 2], xy_norm)
    # [B, N, H]
    point_inclination_diff = np.abs(inclination.reshape((1,-1)) - point_inclination.reshape((-1,1)))
    # [B, N]
    point_ri_row_indices = np.argmin(point_inclination_diff, axis=-1)

    # [B, N], within [-pi, pi]
    point_azimuth = np.arctan2(points[..., 1], points[..., 0])

    # point_azimuth_gt_pi_mask = point_azimuth > np.pi
    # point_azimuth_lt_minus_pi_mask = point_azimuth < -np.pi
    # point_azimuth = point_azimuth - point_azimuth_gt_pi_mask * 2 * np.pi
    # point_azimuth = point_azimuth + point_azimuth_lt_minus_pi_mask * 2 * np.pi

    # [B, N].
    point_ri_col_indices = WIDTH - 1.0 + 0.5 - (point_azimuth + np.pi) / (2.0 * np.pi) * WIDTH
    point_ri_col_indices = np.round(point_ri_col_indices).astype(int)

    # with tf.control_dependencies([
    #     tf.compat.v1.assert_non_negative(point_ri_col_indices),
    #     tf.compat.v1.assert_less(point_ri_col_indices, tf.cast(width, tf.int32))
    # ]):
    # [B, N, 2]
    ri_indices = np.stack([point_ri_row_indices, point_ri_col_indices], -1)
    # [B, N]
    ri_ranges = np.linalg.norm(points, axis=-1)

    #Convert to tensor
    ri_indices = tf.convert_to_tensor(ri_indices)
    ri_ranges = tf.convert_to_tensor(ri_ranges) 
    
    def build_range_image(args):
        """Builds a range image for each frame.

        Args:
            args: a tuple containing:
            - ri_index: [N, 2] int tensor.
            - ri_value: [N] float tensor.
            - num_point: scalar tensor
            - point_feature: [N, 2] float tensor.

        Returns:
            range_image: [H, W]
        """
        if len(args) == 3:
            ri_index, ri_value, num_point = args
        else:
            ri_index, ri_value, num_point, point_feature = args
            ri_value = tf.concat([ri_value[..., tf.newaxis], point_feature],
                                axis=-1)
            #ri_value = encode_lidar_features(ri_value)

        # pylint: disable=unbalanced-tuple-unpacking
        ri_index = ri_index[0:num_point, :]
        ri_value = ri_value[0:num_point, ...]
        range_image = scatter_nd_with_pool(ri_index, ri_value, [HEIGHT, WIDTH],
                                            tf.math.unsorted_segment_min)
        # if len(args) != 3:
        #     range_image = decode_lidar_features(range_image)
        return range_image

    num_points = ri_ranges.shape[0]
    elems = [ri_indices, ri_ranges, num_points]
    if point_features is not None:
        elems.append(point_features)
    # range_images = tf.map_fn(
    #     fn, elems=elems, dtype=points_vehicle_frame_dtype, back_prop=False)
    
    range_images = build_range_image(elems)
    return range_images.numpy(), ri_indices.numpy(), ri_ranges.numpy()



#compute inclinations
inclinations = compute_inclinations(beam_inclination_min, beam_inclination_max)

#Remove ground plane
pc = filter_below_groundplane(points[:,:3], tolerance=0)

range_image, ri_indices, ri_ranges = build_range_image_from_point_cloud(pc[:,:3], inclinations, point_features=None)
# plt.hist(ri_ranges, bins = 20)
# plt.show()

# plt.hist((range_image[range_image > 0.1]).reshape((-1)), bins = 20)
# plt.show()

# Fill empty pixels with highest range in the kernel
empty_pixels = range_image < 0.1
dilated = cv2.dilate(range_image, DIAMOND_KERNEL_5)
range_image[empty_pixels] = dilated[empty_pixels]


range_image_cartesian = extract_point_cloud_from_range_image(range_image, inclinations)
new_points = range_image_cartesian[empty_pixels].reshape((-1, 3))
final_points = np.vstack((points[:,:3], new_points))





# range_image_color = cv2.applyColorMap(np.uint8(range_image / np.amax(range_image) * 255),
#                     cv2.COLORMAP_JET)
# cv2.namedWindow('range_img', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('range_img', range_image_color)
# cv2.waitKey()

V.draw_scenes(points[:,:3])

V.draw_scenes(final_points[:,:3])
b=1
