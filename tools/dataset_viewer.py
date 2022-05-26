import argparse
import glob
import pickle
from pathlib import Path


import open3d
from visual_utils import open3d_vis_utils as V

import numpy as np

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.utils import common_utils
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2

def convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0):
    """
    Modified from the codes of Waymo Open Dataset.
    Convert range images to point cloud.
    Args:
        frame: open dataset frame
        range_images: A dict of {laser_name, [range_image_first_return, range_image_second_return]}.
        camera_projections: A dict of {laser_name,
            [camera_projection_from_first_return, camera_projection_from_second_return]}.
        range_image_top_pose: range image pixel pose for top lidar.
        ri_index: 0 for the first return, 1 for the second return.

    Returns:
        points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        cp_points: {[N, 6]} list of camera projections of length 5 (number of lidars).
    """
    calibrations = sorted(frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []
    points_NLZ = []
    points_intensity = []
    points_elongation = []

    frame_pose = tf.convert_to_tensor(np.reshape(np.array(frame.pose.transform), [4, 4]))
    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(range_image_top_pose.data), range_image_top_pose.shape.dims
    )
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[..., 0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation) # for vehicle pose wrt global frame when this pixel's 3D point was recorded
    for c in calibrations:
        range_image = range_images[c.name][ri_index] #range, intensity, elongation,  is in any no label zone.
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test #uniform inclination
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4]) #Lidar frame to vehicle frame.

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor # vehicle pose for each 3D point recorded by Top lidar
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0) # vehicle pose
        range_image_mask = range_image_tensor[..., 0] > 0
        range_image_NLZ = range_image_tensor[..., 3]
        range_image_intensity = range_image_tensor[..., 1]
        range_image_elongation = range_image_tensor[..., 2]
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.where(range_image_mask))
        points_NLZ_tensor = tf.gather_nd(range_image_NLZ, tf.compat.v1.where(range_image_mask))
        points_intensity_tensor = tf.gather_nd(range_image_intensity, tf.compat.v1.where(range_image_mask))
        points_elongation_tensor = tf.gather_nd(range_image_elongation, tf.compat.v1.where(range_image_mask))
        cp = camera_projections[c.name][0]
        cp_tensor = tf.reshape(tf.convert_to_tensor(cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor, tf.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())
        points_NLZ.append(points_NLZ_tensor.numpy())
        points_intensity.append(points_intensity_tensor.numpy())
        points_elongation.append(points_elongation_tensor.numpy())

    return points, cp_points, points_NLZ, points_intensity, points_elongation

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
    """Show a camera image and the given camera labels."""

    ax = plt.subplot(*layout)

    # # Draw the camera labels.
    # for camera_label in camera_labels:
    #     # Ignore camera labels that do not correspond to this camera.
    #     if camera_label.name != camera_image.name:
    #         continue
    #
    #     # Iterate over the individual labels.
    #     for label in camera_label.labels:
    #         # Draw the object bounding box.
    #         ax.add_patch(patches.Rectangle(
    #             xy=(label.box.center_x - 0.5 * label.box.length,
    #                 label.box.center_y - 0.5 * label.box.width),
    #             width=label.box.length,
    #             height=label.box.width,
    #             linewidth=1,
    #             edgecolor='red',
    #             facecolor='none'))

    # Show the camera image.
    plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
    plt.title(dataset_pb2.CameraName.Name.Name(camera_image.name))
    plt.grid(False)
    plt.axis('off')

class WaymoDataset():
    def __init__(self, seq_file=None, simLidar=False):
        self.root_path = Path('/home/barza/OpenPCDet/data/waymo')
        self.seq_dir = self.root_path / 'domain_adaptation_0000'# #TODO: 'domain_adaptation_0000'#
        self.noisy_lidar_seq_dir = self.root_path / 'da_processed_data' #'noisy_pocessed_data'
        if seq_file is not None:
            self.seq_file_list = [seq_file]
        else:
            self.get_seq_file_list()
        self.gen_next = self.get_next()
        self.simLidar = simLidar

    def get_seq_file_list(self):
        self.seq_file_list = glob.glob(str(self.seq_dir / '*.tfrecord'))
        self.seq_file_list.sort()

    def get_next(self):
       for seq_path in self.seq_file_list:
           seq_name = seq_path.split('/')[-1].split('.')[0]
           if self.simLidar:
               noisy_seq_path = self.noisy_lidar_seq_dir / seq_name
               if not noisy_seq_path.exists():
                   print('skipping: ', noisy_seq_path)
                   continue

           sequence_dataset = tf.data.TFRecordDataset(str(seq_path), compression_type='')

           for cnt, data in enumerate(sequence_dataset):
               if cnt > 0:
                   break
               print(f'Seq: {seq_path}, frame: {cnt}')
               frame = dataset_pb2.Frame()
               frame.ParseFromString(bytearray(data.numpy()))
               stats =frame.context.stats
               b=1
               
               if self.simLidar:
                    lidar_path = self.noisy_lidar_seq_dir / seq_name / (str(cnt).zfill(4) + '.npy')
                    points_intensity_elong = np.load(lidar_path)
               else:
                   (range_images, camera_projections, range_image_top_pose) = frame_utils.parse_range_image_and_camera_projection(frame)
                   points, _, _, points_intensity, points_elongation = convert_range_image_to_point_cloud(frame, range_images, camera_projections, range_image_top_pose, ri_index=0)
                   #points_intensity_elong = np.concatenate((points[0], points_intensity[0].reshape(-1,1), points_elongation[0].reshape(-1,1)), axis =1)

                   points_all = np.concatenate(points, axis=0)
                   points_intensity = np.concatenate(points_intensity, axis=0).reshape(-1, 1)
                   points_elongation = np.concatenate(points_elongation, axis=0).reshape(-1, 1)

                   points_intensity_elong = np.concatenate([
                       points_all, np.tanh(points_intensity), points_elongation
                   ], axis=-1).astype(np.float32)

               print(frame.context.stats.weather)
               data_dict = {
                   'points': points_intensity_elong,
                   'frame_id': cnt,
                   'frame': frame
               }

               yield data_dict

class CadcDataset():
    def __init__(self, split=None, index=None):

        """

        :param root_path: path to dataset
        :param split: name of .txt file in ImageSets e.g. 'train'
        :param index: id for one sample e.g. '2018_03_06,0002,0000000043'
        """
        self.root_path = Path('/home/barza/OpenPCDet/data/cadc')
        self.lidar_ext = '.bin'
        self.img_ext = '.png'

        self.get_sample_file_list(split, index)

    def get_sample_file_list(self, split=None, index=None):

        if split is not None:
            split_path = self.root_path / 'ImageSets' / (split + '.txt')
            sample_list = [x.strip().split(' ') for x in open(split_path).readlines()]
            
            lidar_file_list = [str(self.root_path / x[0] / x[1] / 'labeled' / 'lidar_points' / 'data' / (x[2]  + self.lidar_ext)) for x in sample_list]
            image_file_list = [str(self.root_path / x[0] / x[1] / 'labeled' / 'image_00' / 'data' / (x[2]  + self.img_ext)) for x in sample_list]

        elif index is not None:
            date, seq, id = index.split(',')
            lidar_file_list = [str(self.root_path / date / seq / 'labeled' / 'lidar_points' / 'data' / (id + self.lidar_ext))]
            image_file_list = [str(self.root_path / date / seq / 'labeled' / 'image_00' / 'data' / (id + self.img_ext))]
        else:
            raise NotImplementedError

        lidar_file_list.sort()
        image_file_list.sort()
        assert len(lidar_file_list) == len(image_file_list)
        self.sample_file_list = {'lidar_file_list': lidar_file_list,
                                 'image_file_list': image_file_list}

    def __len__(self):
        return len(self.sample_file_list['lidar_file_list'])

    def __getitem__(self, i):

        if self.lidar_ext == '.bin':
            points = np.fromfile(self.sample_file_list['lidar_file_list'][i], dtype=np.float32)
            try:
                points = points.reshape((-1, 5))
            except Exception:
                points = points.reshape((-1, 4))
        elif self.lidar_ext == '.npy':
            points = np.load(self.sample_file_list['lidar_file_list'][i])
        else:
            raise NotImplementedError

        image = cv2.imread(self.sample_file_list['image_file_list'][i], -1)

        data_dict = {
            'points': points,
            'frame_id': i,
            'imgs': [image]
        }

        return data_dict


class DenseDataset():
    def __init__(self, split=None, index=None):

        """

        :param root_path: path to dataset
        :param split: name of .txt file in ImageSets e.g. 'all', 'dense_fog_day'
        :param index: id for one sample e.g. '2018-02-03_21-51-05_00200'
        """
        self.root_path = Path('/home/barza/OpenPCDet/data/dense') 
        self.lidar_ext = '.bin'
        self.img_ext = '.png'
        self.lidar_dir = self.root_path / 'lidar_hdl64_strongest'
        self.image_dir = self.root_path / 'cam_stereo_left_lut'
        self.get_sample_file_list(split, index)

    def get_sample_file_list(self, split=None, index=None):

        if split is not None:
            split_path = self.root_path / 'ImageSets' / (split + '.txt')
            sample_list = [x.strip().replace(',','_') for x in open(split_path).readlines()]

            lidar_file_list = [str(self.lidar_dir / (x + self.lidar_ext)) for x in sample_list]
            image_file_list = [str(self.image_dir / (x + self.img_ext)) for x in sample_list]
            
        elif index is not None:
            lidar_file_list = [str(self.lidar_dir / (index + self.lidar_ext))]
            image_file_list = [str(self.image_dir / (index+ self.img_ext))]
            
        else:
            lidar_file_list = glob.glob(str(self.lidar_dir / f'*{self.lidar_ext}'))
            image_file_list = glob.glob(str(self.image_dir / f'*{self.img_ext}'))


        lidar_file_list.sort()
        image_file_list.sort()
        assert len(lidar_file_list) == len(image_file_list)
        self.sample_file_list = {'lidar_file_list': lidar_file_list,
                                 'image_file_list': image_file_list}

    def __len__(self):
        return len(self.sample_file_list['lidar_file_list'])

    def __getitem__(self, i):

        if self.lidar_ext == '.bin':
            points = np.fromfile(self.sample_file_list['lidar_file_list'][i], dtype=np.float32)
            try:
                points = points.reshape((-1, 5))
            except Exception:
                points = points.reshape((-1, 4))
        elif self.lidar_ext == '.npy':
            points = np.load(self.sample_file_list['lidar_file_list'][i])
        else:
            raise NotImplementedError

        image = cv2.imread(self.sample_file_list['image_file_list'][i], -1)

        print('Reading: ', self.sample_file_list['lidar_file_list'][i])
        data_dict = {
            'points': points,
            'frame_id': i,
            'imgs': [image]
        }

        return data_dict

class KittiDataset():
    def __init__(self,  index=None):

        self.root_path = Path('/home/barza/OpenPCDet/data/kitti')
        self.lidar_ext = '.bin'
        self.img_ext = '.png'
        self.lidar_dir = root_path / 'training' / 'velodyne'
        self.image_dir = root_path / 'training' / 'image_2'
        self.get_sample_file_list(index)
        
    
    def get_sample_file_list(self, index=None):

        if index is None:
            lidar_file_list = glob.glob(str(self.lidar_dir / f'*{self.lidar_ext}')) 
            image_file_list = glob.glob(str(self.image_dir / f'*{self.img_ext}'))
        else:
            lidar_file_list = [str(self.lidar_dir / (index.zfill(6) + self.lidar_ext))]
            image_file_list = [str(self.image_dir / (index.zfill(6) + self.img_ext))]

        lidar_file_list.sort()
        image_file_list.sort()
        assert len(lidar_file_list) == len(image_file_list)
        self.sample_file_list = {'lidar_file_list': lidar_file_list,
                                 'image_file_list': image_file_list}
        
    def __len__(self):
        return len(self.sample_file_list['lidar_file_list'])

    def __getitem__(self, index):
        
        if self.lidar_ext == '.bin':
            points = np.fromfile(self.sample_file_list['lidar_file_list'][index], dtype=np.float32).reshape(-1, 4)
        elif self.lidar_ext == '.npy':
            points = np.load(self.sample_file_list['lidar_file_list'][index])
        else:
            raise NotImplementedError
        
        image = cv2.imread(self.sample_file_list['image_file_list'][index], -1)
        
        data_dict = {
            'points': points,
            'frame_id': index,
            'imgs': [image]
        }

        return data_dict


def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--dataset', type=str, default='kitti',
                        help='specify dataset e.g. kitti, waymo, dense')
    parser.add_argument('--split', type=str, default=None, help='specify the split for Dense, Cadc')
    parser.add_argument('--index', type=str, default=None, help='specify the index/sample id of point cloud data file for Kitti, Dense, Cadc')
    parser.add_argument('--seq_path', type=str, default=None, help='specify seq tfrecord path')
    parser.add_argument('--tfrecord', action='store_true', help= 'specify if tfrecord')
    parser.add_argument('--simLidar', action='store_true', help='specify if .npy lidar')

    args = parser.parse_args()

    return args

def range_wise_intensity(pc):
    fig = plt.figure()
    ranges = np.linalg.norm(pc[:, 0:3], axis=1)
    intensity = pc[:, 3]

    range_bin_step = 5
    range_bins = np.arange(start=0, stop=80 + range_bin_step +1, step=range_bin_step) #np.linspace(start=0, stop=min(ranges.max()+bin_step, 100), num= 20+1, endpoint=True)
    bin_indices = np.digitize(ranges, range_bins)
    num_bins = range_bins.shape[0]-1
    means = np.zeros(num_bins)
    std = np.zeros(num_bins)
    for bin_i in range(1, num_bins+1):
        idx_for_ranges_in_bin_i = np.argwhere(bin_indices == bin_i)
        means[bin_i-1] = intensity[idx_for_ranges_in_bin_i].mean()
        std[bin_i-1] = intensity[idx_for_ranges_in_bin_i].std()


    plt.plot(range_bins[:-1], means, label='real rain', color= 'r')
    plt.plot(range_bins[:-1], means - std, '--', color='r')
    plt.plot(range_bins[:-1], means + std, '--', color='r')
    plt.scatter(range_bins[:-1], means, marker='x')
    plt.xticks(range_bins)
    plt.grid()
    plt.ylabel('mean intensity')
    plt.xlabel('range bins')
    plt.show()
    plt.pause(0.001)

    return range_bins, means, std

def genHistogram(pc):
    fig = plt.figure('hist', figsize=(15,5))
    range = np.linalg.norm(pc[:, 0:3], axis=1)
    intensity = pc[:,3]

    dict_data = {'range': range, 'intensity': intensity}

    with open('hist_range_intensity_clear.pkl', 'wb') as f:
        pickle.dump(dict_data, f)

    with open('hist_range_intensity.pkl', 'rb') as f:
        dict_data = pickle.load(f)


    fig.add_subplot(1,2,1)
    n, bins, patches = plt.hist(x=range,
                                bins=np.linspace(start=0.1, stop=min(range.max(), 100), num=20 + 1, endpoint=True),
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)
    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Range')
    plt.ylabel('Number of Points')
    plt.title(f'Visibility of Point Cloud')
    maxfreq = n.max()
    # Set a clean upper y-axis limit.
    plt.ylim(ymax=np.ceil(maxfreq / 10) * 10 if maxfreq % 10 else maxfreq + 10)


    fig.add_subplot(1, 2, 2)
    n, bins, patches = plt.hist(x=intensity,
                                bins=np.linspace(start=0, stop=256, num=20 + 1, endpoint=True),
                                color='#0504aa',
                                alpha=0.7, rwidth=0.85)


    plt.grid(axis='y', alpha=0.75)
    plt.xlabel('Intensity')
    plt.ylabel('Number of Points')
    plt.title(f'Intensities of Point Cloud')
    plt.savefig('hist_Rr.png')
    plt.show()
    plt.pause(0.001)

def get_colors(pc, color_feature=None):
    # create colormap
    if color_feature == 0:
        feature = pc[:, 0]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 1:

        feature = pc[:, 1]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 2:
        feature = pc[:, 2]
        min_value = np.min(feature)
        max_value = np.max(feature)

    elif color_feature == 3:
        feature = pc[:, 3]
        min_value = np.min(feature)
        max_value = 1

    elif color_feature == 4:
        feature = pc[:, 4]
        min_value = np.min(feature)
        max_value = np.max(feature)

    else:
        feature = np.linalg.norm(pc[:, 0:3], axis=1)
        min_value = np.min(feature)
        max_value = np.max(feature)


    norm = mpl.colors.Normalize(vmin=min_value, vmax=max_value)


    cmap = cm.jet  # sequential

    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    colors = m.to_rgba(feature)
    colors[:, [2, 1, 0, 3]] = colors[:, [0, 1, 2, 3]]
    colors[:, 3] = 0.5

    return colors[:, :3]

def main():
    args= parse_config()
    logger = common_utils.create_logger()
    logger.info('-----------------Quick Demo of OpenPCDet-------------------------')
    
    if 'kitti' in args.dataset:
        dataset = KittiDataset(index = args.index)
    elif 'dense' in args.dataset:
        dataset = DenseDataset(split=args.split, index= args.index)
    elif 'cadc' in args.dataset:
        dataset = CadcDataset(split=args.split, index= args.index)
    elif 'waymo' in args.dataset:
        dataset = WaymoDataset(seq_file=args.seq_path, simLidar=args.simLidar)
    #logger.info(f'Total number of samples: \t{len(dataset)}')
    
    # To hold the plot but continue execution of code
    plt.ion()
    plt.show()

    # Draw first image
    idx = 0
    data_dict = next(dataset.gen_next) if args.tfrecord else dataset[idx]
    plt.figure('image')
    if not args.tfrecord:
        image = cv2.cvtColor(data_dict['imgs'][0], cv2.COLOR_BGR2RGB)
        plt.imshow(image)
    else:
        #plt.figure(figsize=(25, 20))
        show_camera_image(data_dict['frame'].images[0], data_dict['frame'].projected_lidar_labels, [1, 1, 1])
        # for index, image in enumerate(data_dict['frame'].images):
        #     show_camera_image(image, data_dict['frame'].projected_lidar_labels, [3, 3, index + 1])
    plt.pause(0.001)
    vis = V.init_vis()
    #genHistogram(data_dict['points'][:, :4])
    range_wise_intensity(data_dict['points'])
    V.draw_scene(vis, points=data_dict['points'][:, :3], point_colors = get_colors(data_dict['points'], color_feature=3))


    def right_click_id(vis):
        plt.close('all')
        nonlocal idx
        print('right_click')
        idx = min(idx + 1, len(dataset)-1)
        data_dict = dataset[idx]
        image = cv2.cvtColor(data_dict['imgs'][0], cv2.COLOR_BGR2RGB)
        plt.figure('image')
        plt.imshow(image)
        plt.pause(0.001)
        vis.clear_geometries()
        genHistogram(data_dict['points'][:, :4])
        V.draw_scene(vis, points=data_dict['points'][:, :3], point_colors=get_colors(data_dict['points'], color_feature=3))

    def right_click_next(vis):
        plt.close('all')
        nonlocal idx
        print('right_click_next')
        data_dict = next(dataset.gen_next)
        show_camera_image(data_dict['frame'].images[0], data_dict['frame'].projected_lidar_labels, [1, 1, 1])
        # for index, image in enumerate(data_dict['frame'].images):
        #     show_camera_image(image, data_dict['frame'].projected_lidar_labels, [3, 3, index + 1])
        plt.pause(0.001)
        vis.clear_geometries()
        #genHistogram(data_dict['points'][:, :4])
        range_wise_intensity(data_dict['points'])
        V.draw_scene(vis, points=data_dict['points'][:, :3], point_colors=get_colors(data_dict['points'], color_feature=3))

    def exit_key(vis):
        vis.destroy_window()
    
    
    if args.tfrecord:
        vis.register_key_callback(262, right_click_next)  # right_arrow_key for next frame
    else:
        vis.register_key_callback(262, right_click_id)  # right_arrow_key for next frame
    vis.register_key_callback(32, exit_key)  # space_bar to exit
    vis.poll_events()
    vis.run()


    logger.info('Demo done.')


if __name__ == '__main__':
    main()
