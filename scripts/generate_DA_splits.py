import numpy as np
from pathlib import Path
import tensorflow as tf
from waymo_open_dataset.utils import frame_utils, transform_utils, range_image_utils
from waymo_open_dataset import dataset_pb2
import glob
import matplotlib.pyplot as plt
import matplotlib.patches as patches


"""
Shortens train.txt
"""
root_path = Path('/media/barza/WD_BLACK/datasets/waymo')
dataset_dir = root_path / 'domain_adaptation'
splits_dir = [dataset_dir / 'training', dataset_dir / 'validation']
splits_txt = ['train_all.txt', 'val_all.txt'] #['train_da', 'val_da']

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

def get_rainy_samples(seq_path_list, show_image = True):
    seq_list_rainy = []
    for i, sequence_path in enumerate(seq_path_list):
        sequence_dataset = tf.data.TFRecordDataset(str(sequence_path), compression_type='')

        for cnt, data in enumerate(sequence_dataset):
            print(f'Seq: {i}, frame: {cnt}')
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            weather = frame.context.stats.weather
            location = frame.context.stats.location
            tofd = frame.context.stats.time_of_day
            print(weather)
            if weather != 'sunny':
                if show_image:
                    plt.figure(figsize=(25, 20))
                    for index, image in enumerate(frame.images):
                        show_camera_image(image, frame.projected_lidar_labels, [3, 3, index + 1])
                    plt.show()
                seq_list_rainy.append(Path(sequence_path).name)
                print(f'seq_idx: {Path(sequence_path).name}, frame: {cnt}, weather: {weather}, location: {location}, tofd: {tofd} \n')
                break
        print(f'Processed sequence: {i}')
    return seq_list_rainy



for split_dir, split_txt in zip(splits_dir, splits_txt):
    glob_str = str(split_dir) + '/*/*.tfrecord'
    seq_path_list = glob.glob(glob_str)
    seq_name_list = [Path(seq_path).name for seq_path in seq_path_list]
    rainy_seq_name_list = get_rainy_samples(seq_path_list)

    # dst_txt = root_path / 'ImageSets' / (split_txt + '.txt')
    # with open(dst_txt, 'w') as f:
    #     for i, sample_idx in enumerate(sample_id_list):
    #         f.write(sample_idx)
    #         if i != len(sample_id_list) - 1:
    #             f.write('\n')



train_txt_1 = root_path / 'ImageSets' / ('train.txt')
train_txt_2 = root_path / 'ImageSets' / ('train_da.txt')
sample_id_list_train = [x.strip() for x in open(train_txt_1).readlines()] + [x.strip() for x in open(train_txt_2).readlines()]

val_txt_1 = root_path / 'ImageSets' / ('val.txt')
val_txt_2 = root_path / 'ImageSets' / ('val_da.txt')
sample_id_list_val = [x.strip() for x in open(val_txt_1).readlines()] + [x.strip() for x in open(val_txt_2).readlines()]

seq_lists = [sample_id_list_train, sample_id_list_val]
for split, seq_list in zip(splits_txt, seq_lists):
    dst_txt = root_path / 'ImageSets' / split
    with open(dst_txt, 'w') as f:
        for i, sample_idx in enumerate(seq_list):
            f.write(sample_idx)
            if i != len(seq_list) - 1:
                f.write('\n')
                
                
