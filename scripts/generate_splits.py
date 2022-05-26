import numpy as np
from pathlib import Path
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# """
# Shortens train.txt
# """
# root_path = Path('/home/barza/OpenPCDet/data/kitti')
# split = 'train'
# new_split_name = 'new_train'
# shorten_by = 0.3
#
#
# src_txt = root_path / 'ImageSets' / (split + '.txt')
# dst_txt = root_path / 'ImageSets' / ( new_split_name + '.txt')
# sample_id_list = [x.strip() for x in open(src_txt).readlines()] #if split_dir.exists() else None
#
# short_sample_id_list = np.random.choice(sample_id_list, int(shorten_by*len(sample_id_list)))
# short_sample_id_list = sorted(list(short_sample_id_list))
#
# with open(dst_txt, 'w') as f:
#     for i, sample_idx in enumerate(short_sample_id_list):
#         f.write(sample_idx)
#         if i != len(short_sample_id_list)-1:
#             f.write('\n')



"""
Shortens train.txt
"""
root_path = Path('/home/barza/OpenPCDet/data/waymo')
splits = ['domain_adaptation_training_training_0000', 'domain_adaptation_validation_validation_0000'] #['train', 'val']
new_splits_name = ['train_rainy', 'val_rainy'] #['train_0000', 'val_0000']

def show_camera_image(camera_image, camera_labels, layout, cmap=None):
  """Show a camera image and the given camera labels."""

  ax = plt.subplot(*layout)

  # Draw the camera labels.
  for camera_label in camera_labels:
    # Ignore camera labels that do not correspond to this camera.
    if camera_label.name != camera_image.name:
      continue

    # Iterate over the individual labels.
    for label in camera_label.labels:
      # Draw the object bounding box.
      ax.add_patch(patches.Rectangle(
        xy=(label.box.center_x - 0.5 * label.box.length,
            label.box.center_y - 0.5 * label.box.width),
        width=label.box.length,
        height=label.box.width,
        linewidth=1,
        edgecolor='red',
        facecolor='none'))

  # Show the camera image.
  plt.imshow(tf.image.decode_jpeg(camera_image.image), cmap=cmap)
  plt.title(dataset_pb2.CameraName.Name.Name(camera_image.name))
  plt.grid(False)
  plt.axis('off')


def get_samples_that_exist(sample_id_list):
    sample_id_list_exists = []
    for i, sample_idx in enumerate(sample_id_list):
        sample_id_path = root_path / 'raw_data' / sample_idx
        if sample_id_path.exists():
            sample_id_list_exists.append(sample_idx)

    return sample_id_list_exists

def get_rainy_samples(sample_id_list, dir_dataset='raw_data', show_image = False):
    sample_id_list_rainy = []
    for i, sample_idx in enumerate(sample_id_list):
        sequence_path = root_path / dir_dataset / sample_idx

        if sequence_path.exists():
            sequence_dataset = tf.data.TFRecordDataset(str(sequence_path), compression_type='')

            for cnt, data in enumerate(sequence_dataset):
                print(f'Seq: {i}, frame: {cnt}')
                frame = dataset_pb2.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                weather = frame.context.stats.weather
                location = frame.context.stats.location
                tofd = frame.context.stats.time_of_day
                if weather != 'sunny':
                    if show_image:
                        plt.figure(figsize=(25, 20))
                        for index, image in enumerate(frame.images):
                            show_camera_image(image, frame.projected_lidar_labels, [3, 3, index + 1])
                        plt.show()
                    sample_id_list_rainy.append(sample_idx)
                    print(f'seq_idx: {sample_idx}, frame: {cnt}, weather: {weather}, location: {location}, tofd: {tofd} \n')
                    break
        print(f'Processed sequence: {i}')
    return sample_id_list_rainy



func = get_rainy_samples # get_samples_that_exist

for split, new_split in zip(splits, new_splits_name):
    # src_txt = root_path / 'ImageSets' / (split + '.txt')
    # sample_id_list = [x.strip() for x in open(src_txt).readlines()]
    sample_id_path_list = glob.glob(f'/home/barza/OpenPCDet/data/waymo/{split}/*.tfrecord')
    sample_id_list = [Path(full_path).name for full_path in sample_id_path_list]
    new_sample_id_list = func(sample_id_list, split)

    dst_txt = root_path / 'ImageSets' / (new_split + '.txt')
    with open(dst_txt, 'w') as f:
        for i, sample_idx in enumerate(new_sample_id_list):
            f.write(sample_idx)
            if i != len(new_sample_id_list) - 1:
                f.write('\n')