import numpy as np
import cv2 as cv
from typing import List
from config import Parameters
from readers import DataReader
from processors import DataProcessor
import tensorflow as tf
from scipy.special import softmax

class BBox(Parameters, tuple):
    """ bounding box tuple that can easily be accessed while being compatible to cv2 rotational rects """

    def __new__(cls, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        bbx_tuple = ((float(bb_x), float(bb_y)), (float(bb_length), float(bb_width)), float(np.rad2deg(bb_yaw)))
        return super(BBox, cls).__new__(cls, tuple(bbx_tuple))

    def __init__(self, bb_x, bb_y, bb_z, bb_length, bb_width, bb_height, bb_yaw, bb_heading, bb_cls, bb_conf):
        super(BBox, self).__init__()

        self.x = bb_x
        self.y = bb_y
        self.z = bb_z
        
        self.length = bb_length
        self.width = bb_width
        self.height = bb_height

        self.z -= self.height/2.
        
        # self.length -= 0.3
        # self.width -= 0.3
        # self.length = self.length * (self.x_max - self.x_min)
        # self.width = self.width * (self.y_max - self.y_min)
        # self.height = self.height * (self.z_max - self.z_min)

        self.yaw = bb_yaw
        self.heading = bb_heading
        self.cls = bb_cls
        self.conf = bb_conf
        self.class_dict = { 0: "Car",
                            1: "Pedestrian",
                            2: "Cyclist",
                            3: "Misc."}

    def __str__(self):
        return "BB | Cls: %s, x: %f, y: %f, z: %f, l: %f, w: %f, h: %f, yaw: %f, heading: %f, conf: %f" % (
            self.cls, self.x, self.y, self.z, self.length, self.width, self.height, self.yaw, self.heading, self.conf)
    
    @staticmethod
    def bb_class_colour_map(class_name: str):
        if class_name == "Car":
            return (0, 255, 255)
        elif class_name == "Pedestrian":
            return (0, 0, 255)
        elif class_name == "Cyclist":
            return (255, 0, 0)
        else:
            return (102, 0, 102)

    def to_kitti_format(self, P2: np.ndarray, R0: np.ndarray, V2C: np.ndarray):
        self.x, self.y, self.z = BBox.lidar_to_camera(self.x, self.y, self.z, R0, V2C) # velodyne to camera coordinate projection

        # model predicts angles w.r.t. z-axis in LiDAR coordinate frame
        # changing it to camera coordinate, where the angle is w.r.t y-axis
        # z-axis in LiDAR coordinate frame == -(y-axis) of camera coordinate frame
        self.yaw = self.yaw + np.pi/2

        # if(int(self.heading) == 0):
        #   self.yaw = 2*np.pi + self.yaw

        bbox_2d_image_coordinate, bbox_3d_image_coordinate = self.get_2D_BBox(P2) # [num_boxes, box_attributes]

        # TODO: Check alpha calculation
        # alpha = -1.0
        beta = np.arctan2(self.z, self.x)
        alpha = -np.sign(beta) * np.pi/2 + beta + self.yaw

        return [self.class_dict[self.cls], -1.0, -1, alpha, bbox_2d_image_coordinate[0][0], bbox_2d_image_coordinate[0][1],
                                            bbox_2d_image_coordinate[0][2], bbox_2d_image_coordinate[0][3],
                                            self.height, self.width, self.length, self.x, self.y, self.z, self.yaw, self.conf], bbox_3d_image_coordinate 


    def get_2D_BBox(self, P: np.ndarray):
        """ Projects the 3D box onto the image plane and provides 2D BB 
            1. Get 3D bounding box vertices
            2. Rotate 3D bounding box with yaw angle
            3. Multiply with LiDAR to camera projection matrix
            4. Multiply with camera to image projection matrix
        """
        yaw_rotation_matrix = BBox.get_y_axis_rotation_matrix(self.yaw) # rotation matrix around y-axis
        l = self.length
        w = self.width
        h = self.height

        # 3d bb corner coordinates in camera coordinate frame, coordinate system is at the bottom center of box
        # x-axis -> right (length), y-axis -> bottom (height), z-axis -> forward (width)
        #     7 -------- 4
        #    /|         /|
        #   6 -------- 5 .
        #   | |        | |
        #   . 3 -------- 0
        #   |/         |/
        #   2 -------- 1
        bb_3d_x_corner_coordinates = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        bb_3d_y_corner_coordinates = [0, 0, 0, 0, -h, -h, -h, -h]
        bb_3d_z_corner_coordinates = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
        bb_3d_corners = np.vstack([ bb_3d_x_corner_coordinates, bb_3d_y_corner_coordinates,
                                        bb_3d_z_corner_coordinates])

        # box rotation by yaw angle
        bb_3d_corners = yaw_rotation_matrix @ bb_3d_corners
        # box translation by centroid coordinates
        bb_3d_corners[0, :] = bb_3d_corners[0, :] + self.x
        bb_3d_corners[1, :] = bb_3d_corners[1, :] + self.y
        bb_3d_corners[2, :] = bb_3d_corners[2, :] + self.z

        # camera coordinate frame to image coordinate frame box projection
        img_width = (self.x_max - self.x_min) / self.x_step
        img_height = (self.y_max - self.y_min) / self.y_step
        bbox_2d_image, bbox_corners_image = BBox.camera_to_image(bb_3d_corners, P, img_width, img_height)
        return bbox_2d_image, bbox_corners_image

    @staticmethod
    def get_x_axis_rotation_matrix(rotation_angle):
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = [[1,      0.,             0.],
                           [0.,   cos_theta,    -sin_theta],
                           [0.,     sin_theta,   cos_theta]]
        return np.array(rotation_matrix)

    @staticmethod
    def get_y_axis_rotation_matrix(rotation_angle):
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = [[cos_theta,  0.,   sin_theta],
                           [0.,          1.,     0.      ],
                           [-sin_theta,  0.,  cos_theta]]
        return np.array(rotation_matrix)

    @staticmethod
    def get_z_axis_rotation_matrix(rotation_angle):
        cos_theta = np.cos(rotation_angle)
        sin_theta = np.sin(rotation_angle)
        rotation_matrix = [[cos_theta,  -sin_theta, 0.],
                           [sin_theta,   cos_theta, 0.],
                           [0.,             0.,     1.]]
        return np.array(rotation_matrix)

    @staticmethod
    def lidar_to_camera(x: float, y: float, z: float, R: np.ndarray, V2C: np.ndarray):
        """ Projects the box centroid from LiDAR coordinate system to camera coordinate system using calibration matrices """
        box_centroid = [x, y, z, 1]
        box_centroid = V2C @ box_centroid
        box_centroid = R @ box_centroid
        return box_centroid[:3]

    @staticmethod
    def camera_to_image(bbox_3d_corners: np.ndarray, P: np.ndarray, img_width: float, img_height: float):
        """ box is in camera coordinate frame and reference roatation has already been done to the box 
            1. Convert the BB coordinated into an homogenous matrix
            2. Project the matrix from camera coodinate frame to image coordinate frame using projection matrix P
            3. Normalize by the z-coordinate values
        """
        box_3d_coords_homogenous = np.concatenate((bbox_3d_corners, np.ones((1, 8))), axis=0) # concat([3, 8], [1, 8]) -> [4, 8]
        box_coords_image = P @ box_3d_coords_homogenous # matmul ([3, 4], [4, 8]) -> [3, 8]
        box_x_coords = box_coords_image[0, :] / box_coords_image[2, :] # Normalizing all the x-coords with z-coords
        box_y_coords = box_coords_image[1, :] / box_coords_image[2, :] # Normalizing all the y-coords with z-coords
        xmin, xmax = np.min(box_x_coords), np.max(box_x_coords)
        ymin, ymax = np.min(box_y_coords), np.max(box_y_coords)

        xmin = np.clip(xmin, 0, img_width)
        xmax = np.clip(xmax, 0, img_width)
        ymin = np.clip(ymin, 0, img_height)
        ymax = np.clip(ymax, 0, img_height)

        bbox_2d_image = np.concatenate((xmin.reshape(-1, 1), ymin.reshape(-1, 1), xmax.reshape(-1, 1), ymax.reshape(-1, 1)), axis=1)
        bbox_3d_image = np.concatenate((box_x_coords.reshape(-1, 8, 1), box_y_coords.reshape(-1, 8, 1)), axis=2)
        return bbox_2d_image, bbox_3d_image


def draw_projected_box3d(image, qs, color=(255, 0, 255), thickness=2):
    ''' Draw 3d bounding box in image
        qs: (8,3) array of vertices for the 3d box in following order:
            7 -------- 4
           /|         /|
          6 -------- 5 .
          | |        | |
          . 3 -------- 0
          |/         |/
          2 -------- 1
    '''
    qs = np.squeeze(qs)
    qs = qs.astype(np.int32)
    for k in range(0, 4):
        # Ref: http://docs.enthought.com/mayavi/mayavi/auto/mlab_helper_functions.html
        i, j = k, (k + 1) % 4
        # use LINE_AA for opencv3
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k + 4, (k + 1) % 4 + 4
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)

        i, j = k, k + 4
        cv.line(image, (qs[i, 0], qs[i, 1]), (qs[j, 0], qs[j, 1]), color, thickness)
    return image    


def gather_boxes_in_kitti_format(boxes: List[BBox], indices: List, P2: np.ndarray, R0: np.ndarray, Tr_velo_to_cam: np.ndarray):
    """ gathers boxes left after nms and converts them to kitti evaluation toolkit expected format """
    if len(indices) == 0:
        return [], []
    bb_3d_corners, kitti_format_bb = [], []

    for idx in indices:
        bb_kitti, bb_3d = boxes[idx].to_kitti_format(P2, R0, Tr_velo_to_cam)
        bb_3d_corners.append(bb_3d)
        kitti_format_bb.append(bb_kitti)

    return kitti_format_bb, bb_3d_corners


def dump_predictions(predictions: List, file_path: str):
    """ Dumps the model predictions in txt files so that it can be used by KITTI evaluation toolkit """
    with open(file_path, 'w') as out_txt_file:
        if len(predictions):
            for bbox in predictions:
                for bbox_attribute in bbox:
                    out_txt_file.write("{} ".format(bbox_attribute))
                out_txt_file.write("\n")


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
        bb_cls = np.argmax(softmax(clf[value]))
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

@tf.function
def pillar_net_predict_server(inputs, model):
    """ tf.function wrapper for faster inference """
    return model(inputs, training=False)