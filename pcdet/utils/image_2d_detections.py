import numpy as np

def get_detections_from_label(label_file, image_shape, min_det_threshold=0.0):
    with open(label_file, 'r') as f:
        lines = f.readlines()
    objects_2d = [ImageDetections2d(line) for line in lines]
    detection_heat_map = np.zeros(image_shape, dtype=np.float32)
    for detection in objects_2d:
        detection_bb = detection.box2d.astype(np.int32)
        detection_confidence = detection.confidence
        detection_heat_map[detection_bb[1]:detection_bb[3],
                            detection_bb[0]:detection_bb[2]] = np.maximum(detection_confidence, detection_heat_map[detection_bb[1]:detection_bb[3],
                                                                                                detection_bb[0]:detection_bb[2]])
    return detection_heat_map


class ImageDetections2d(object):
    def __init__(self, line):
        label = line.strip().split(' ')
        self.box2d = np.array((float(label[4]), 
                                float(label[5]), 
                                float(label[6]), 
                                float(label[7])), dtype=np.float32)
        self.confidence = float(label[-1])