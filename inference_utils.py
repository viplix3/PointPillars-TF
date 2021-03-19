import numpy as np
import cv2 as cv
from typing import List
from config import Parameters
from readers import DataReader
from processors import DataProcessor


class BBox(tuple):
    """ bounding box tuple that can easily be accessed while being compatible to cv2 rotational rects """

    def __new__(cls, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        bbx_tuple = ((float(bb_x), float(bb_y)), (float(bb_length), float(bb_width)), float(np.rad2deg(bb_yaw)))
        return super(BBox, cls).__new__(cls, tuple(bbx_tuple))

    def __init__(self, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        self.x = bb_x
        self.y = bb_y
        self.z = bb_z
        self.length = bb_length
        self.width = bb_width
        self.height = bb_height
        self.yaw = bb_yaw
        self.heading = bb_heading
        self.cls = bb_cls
        self.conf = bb_conf
        self.class_dict = { 0: "Car",
                            1: "Pedestrian",
                            2: "Cyclist",
                            3: "Misc."}

    def __str__(self):
        return "BB | Cls: %s, x: %f, y: %f, z: %f, l: %f, w: %f, h: %f, yaw: %f, conf: %f" % (
            self.cls, self.x, self.y, self.z, self.length, self.width, self.height, self.yaw, self.conf)

    def to_kitti_format(self, P2: np.ndarray, R0: np.ndarray, V2C: np.ndarray):
        self.x, self.y, self.z = BBox.lidar_to_camera(self.x, self.y, self.z, P2, R0, V2C) # velodyne to camera coordinate projection

        # TODO: Get 2D BB
        # model predicts angles w.r.t. z-axis in LiDAR coordinate frame
        # changing it to camera coordinate, where the angle is w.r.t y-axis
        # z-axis in LiDAR coordinate frame == -(y-axis) of camera coordinate frame
        angle_y = - self.yaw - (2 * np.pi)
        angle_y = angle_y - (np.pi / 2)
        self.yaw = angle_y
        bbox_2d = self.get_2D_bb(P2)

        # TODO: Check alpha calculation


    def get_2D_bb(self, P: np.ndarray):
        """ Projects the 3D box onto the image plane and provides 2D BB 
            1. Get 3D bounding box vertices
            2. Rotate 3D bounding box with yaw angle
            3. Multiply with LiDAR to camera projection matrix
            4. Multiply with camera to image projection matrix
        """
        Ry = get_y_axis_rotation_matrix(self.yaw) # rotation matrix around y-axis
        l = self.length
        w = self.width
        h = self.height

        # 3d bb corner coordinates in camera coordinate frame, coordinate system is at the center of box
        # x-axis -> right (width), y-axis -> bottom (height), z-axis -> forward (length)
        #     1 -------- 0
        #    /|         /|
        #   2 -------- 3 .
        #   | |        | |
        #   . 5 -------- 4
        #   |/         |/
        #   6 -------- 7
        bb_3d_x_corner_coordinates = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        bb_3d_y_corner_coordinates = [-h/2, -h/2, -h/2, -h/2, h/2, h/2, h/2, h/2]
        bb_3d_z_corner_coordinates = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]

        # box rotation by yaw angle
        bb_3d_corners = R @ np.vstack([bb_3d_x_corner_coordinates, bb_3d_y_corner_coordinates, bb_3d_z_corner_coordinates])
        # box translation by centroid coordinates
        bb_3d_corners = bb_3d_corners[0, :] + self.x
        bb_3d_corners = bb_3d_corners[1, :] + self.y
        bb_3d_corners = bb_3d_corners[2, :] + self.z

    @staticmethod
    def get_y_axis_alinged_rotation_matrix(rotation_angle):
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = [[cos_theta,  0,   sin_theta],
                           [0,          1,     0      ],
                           [-sin_theta  0,   cos_theta]]
        return np.array(rotation_matrix)

    @staticmethod
    def lidar_to_camera(x: float, y: float, z: float, P2: np.ndarray, R: np.ndarray, V2C: np.ndarray):
        """ Projects the box centroid from LiDAR coordinate system to camera coordinate system using calibration matrices """
        box_centroid = [x, y, z, 1]
        box_centroid = V2C @ box_centroid
        box_centroid = P2 @ box_centroid
        return box_centroid[:3]

def gather_boxes_in_kitti_format(boxes: List[BBox], indices: List, P2: np.ndarray, R0: np.ndarray, Tr_velo_to_cam: np.ndarray):
    """ gathers boxes left after nms and converts them to kitti evaluation toolkit expected format """
    if len(indices) == 0:
        return
    nms_boxes = [boxes[idx].to_kitti_format(P2, R0, Tr_velo_to_cam) for idx in indices]
    print(nms_boxes)
    return nms_boxes

def rotational_nms(set_boxes, confidences, occ_threshold=0.7, nms_iou_thr=0.5):
    """ rotational NMS
    set_boxes = size NSeqs list of size NDet lists of tuples. each tuple has the form ((pos, pos), (size, size), angle)
    confidences = size NSeqs list of lists containing NDet floats, i.e. one per detection
    """
    assert len(set_boxes) == len(confidences) and 0 < occ_threshold < 1 and 0 < nms_iou_thr < 1
    if not len(set_boxes):
        return []
    assert (isinstance(set_boxes[0][0][0], float) or isinstance(set_boxes[0][0][0], int)) and \
           (isinstance(confidences[0], float) or isinstance(confidences[0], int))
    nms_boxes = []

    ## If batch_size > 1
    # for boxes, confs in zip(set_boxes, confidences):
    #     assert len(boxes) == len(confs)
    #     indices = cv.dnn.NMSBoxesRotated(boxes, confs, occ_threshold, nms_iou_thr)
    #     indices = indices.reshape(len(indices)).tolist()
    #     nms_boxes.append([boxes[i] for i in indices])

    ## IF batch_size == 1
    indices = cv.dnn.NMSBoxesRotated(set_boxes, confidences, occ_threshold, nms_iou_thr)
    indices = indices.reshape(len(indices)).tolist()
    # nms_boxes.append([set_boxes[i] for i in indices])
    return indices


def generate_bboxes_from_pred(occ, pos, siz, ang, hdg, clf, anchor_dims, occ_threshold=0.5):
    """ Generating the bounding boxes based on the regression targets """

    # Get only the boxes where occupancy is greater or equal threshold.
    real_boxes = np.where(occ >= occ_threshold)
    # Get the indices of the occupancy array
    coordinates = list(zip(real_boxes[0], real_boxes[1], real_boxes[2]))
    # Assign anchor dimensions as original bounding box coordinates which will eventually be changed
    # according to the predicted regression targets
    anchor_dims = anchor_dims
    real_anchors = np.random.rand(len(coordinates), len(anchor_dims[0]))

    for i, value in enumerate(real_boxes[2]):
        real_anchors[i, ...] = anchor_dims[value]

    # Change the anchor boxes based on regression targets, this is the inverse of the operations given in
    # createPillarTargets function (src/PointPillars.cpp)
    predicted_boxes = []
    for i, value in enumerate(coordinates):
        real_diag = np.sqrt(np.square(real_anchors[i][0]) + np.square(real_anchors[i][1]))
        real_x = value[0] * Parameters.x_step * Parameters.downscaling_factor + Parameters.x_min
        real_y = value[1] * Parameters.y_step * Parameters.downscaling_factor + Parameters.y_min
        bb_x = pos[value][0] * real_diag + real_x
        bb_y = pos[value][1] * real_diag + real_y
        bb_z = pos[value][2] * real_anchors[i][2] + real_anchors[i][3]
        # print(position[value], real_x, real_y, real_diag)
        bb_length = np.exp(siz[value][0]) * real_anchors[i][0]
        bb_width = np.exp(siz[value][1]) * real_anchors[i][1]
        bb_height = np.exp(siz[value][2]) * real_anchors[i][2]
        bb_yaw = -np.arcsin(np.clip(ang[value], -1, 1)) + real_anchors[i][4]
        bb_heading = np.round(hdg[value])
        bb_cls = np.argmax(clf[value])
        bb_conf = occ[value]
        predicted_boxes.append(BBox(bb_x, bb_y, bb_z, bb_length, bb_width, bb_height,
                                    bb_yaw, bb_heading, bb_cls, bb_conf))

    return predicted_boxes


class GroundTruthGenerator(DataProcessor):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, label_files: List[str], calibration_files: List[str] = None,
                 network_format: bool = False):
        super(GroundTruthGenerator, self).__init__()
        self.data_reader = data_reader
        self.label_files = label_files
        self.calibration_files = calibration_files
        self.network_format = network_format

    def __len__(self):
        return len(self.label_files)

    def __getitem__(self, file_id: int):
        label = self.data_reader.read_label(self.label_files[file_id])
        R, t = self.data_reader.read_calibration(self.calibration_files[file_id])
        label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)
        if self.network_format:
            occupancy, position, size, angle, heading, classification = self.make_ground_truth(label_transformed)
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [occupancy, position, size, angle, heading, classification]
        return label_transformed


def focal_loss_checker(y_true, y_pred, n_occs=-1):
    y_true = np.stack(np.where(y_true == 1))
    if n_occs == -1:
        n_occs = y_true.shape[1]
    occ_thr = np.sort(y_pred.flatten())[-n_occs]
    y_pred = np.stack(np.where(y_pred >= occ_thr))
    p = 0
    for gt in range(y_true.shape[1]):
        for pr in range(y_pred.shape[1]):
            if np.all(y_true[:, gt] == y_pred[:, pr]):
                p += 1
                break
    print("#matched gt: ", p, " #unmatched gt: ", y_true.shape[1] - p, " #unmatched pred: ", y_pred.shape[1] - p,
          " occupancy threshold: ", occ_thr)
