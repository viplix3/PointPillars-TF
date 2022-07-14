from typing import List
import numpy as np
import tensorflow as tf

from tensorflow.keras.utils import Sequence

from config import Parameters
from point_pillars import createPillars, createPillarsTarget
from readers import DataReader, Label3D
from sklearn.utils import shuffle
import sys


def select_best_anchors(arr):
    dims = np.indices(arr.shape[1:]) # arr -> [num_GT_BB, grid_x, grid_y, num_anchors, 10]
    # 10 -> [occupancy, x_encoded, y_encoded, z_encoded, l_w.r.t._anchor, w_w.r.t_anchor, h_w.r.t._anchor, sin(GT_yaw - anchor_yaw), yaw_class, box_class_id]
    # arr[..., 0:1] gets the occupancy value from occ in {-1, 0, 1}, i.e. {bad match, neg box, pos box}
    ind = (np.argmax(arr[..., 0:1], axis=0),) + tuple(dims) # mask on the basis of occupancy == 1 -> [occupancy, grid_x, grid_y, num_anchors, 10]
    return arr[ind]


class DataProcessor(Parameters):

    def __init__(self):
        super(DataProcessor, self).__init__()
        anchor_dims = np.array(self.anchor_dims, dtype=np.float32)
        self.anchor_dims = anchor_dims[:, 0:3] # length, width, height
        self.anchor_z = anchor_dims[:, 3] # z-center
        self.anchor_yaw = anchor_dims[:, 4] # yaw-angle
        # Counts may be used to make statistic about how well the anchor boxes fit the objects
        self.pos_cnt, self.neg_cnt = 0, 0

    @staticmethod
    def transform_labels_into_lidar_coordinates(labels: List[Label3D], R: np.ndarray, t: np.ndarray):
        transformed = [] # Creating transformes list but returning labels
        for label in labels:
            # @ -> Matrix multiplication
            label.centroid = label.centroid @ np.linalg.inv(R).T - t # Why transposing the inversed orthogonal matrix? Where is Camera rotation handling?
            label.dimension = label.dimension[[2, 1, 0]] # h, w, l -> w, l, h
            label.yaw -= np.pi / 2 
            while label.yaw < -np.pi:
                label.yaw += (np.pi * 2)
            while label.yaw > np.pi:
                label.yaw -= (np.pi * 2)
            transformed.append(label)
        return labels

    def camera_to_lidar(self, labels: List[Label3D], P: np.ndarray, R: np.ndarray, V2C: np.ndarray):
        """ Transforms all the box parameters from camera coordinate frame to LiDAR coordinate frame """
        for label in labels:
            # In camera coordinate frame: length (or depth) will be along z-axis, width will be along x-axis, height will be along y-axis
            # In lidar coordinate frame: length (or depth) will be along x-axis, width will be along y-axis, height will be along z-axis
            label.dimension = label.dimension[[2, 1, 0]] # h, w, l -> l, w, h, now they are alinged to the LiDAR coordinate frame

            label_centroid = label.centroid # (x, y, z) of BB in camera coordinates (in meters)
            label_centroid_rectified = np.array([label_centroid[0], label_centroid[1], label_centroid[2], 1])

            # Transforming 3x3 reference camera rotation matrix into 4x4 rotation matrix
            R_rectification = np.zeros((4, 4))
            R_rectification[:3, :3] = R
            R_rectification[3, 3] = 1
            # inversing the rotation, so now, rotation w.r.t. camera is gone
            label_centroid_rectified = np.matmul(np.linalg.inv(R_rectification), label_centroid_rectified)
            # Projecting from camera to LiDAR
            label_centroid_rectified = np.matmul(DataProcessor.inverse_rigid_trans(V2C), label_centroid_rectified)
            label_centroid_rectified = label_centroid_rectified[:3]
            label.centroid = label_centroid_rectified
            label.centroid[2] += label.dimension[2]/2. # Getting z-center of the 3D BB

            # # Adding an offset to the length and width values
            # label.dimension[0] = label.dimension[0] + 0.3 # Adding some constant factor as GT values are very small
            # label.dimension[1] = label.dimension[1] + 0.3 # Adding some constant factor as GT values are very small

            # # Normalizing box dimensions
            # label.dimension[0] = label.dimension[0] / (self.x_max - self.x_min)
            # label.dimension[1] = label.dimension[1] / (self.y_max - self.y_min)
            # label.dimension[2] = label.dimension[2] / (self.z_max - self.z_min)


            # yaw angle has been provided w.r.t y-axis in the camera coordinate
            # label.yaw = -(label.yaw + np.pi/2) # Rotation w.r.t z-axis of LiDAR coordinate frame
            label.yaw = - label.yaw - np.pi/2
            # while label.yaw < -np.pi:
            #     label.yaw += (np.pi * 2)
            # while label.yaw > np.pi:
            #     label.yaw -= (np.pi * 2)
        return labels

    @staticmethod
    def inverse_rigid_trans(Tr):
        ''' Inverse a rigid body transform matrix (3x4 as [R|t])
            [R'|-R't; 0|1]
        '''
        inv_Tr = np.zeros_like(Tr)  # 3x4
        inv_Tr[0:3, 0:3] = np.transpose(Tr[0:3, 0:3]) # Inverse rotation matrix
        inv_Tr[0:3, 3] = np.dot(-np.transpose(Tr[0:3, 0:3]), Tr[0:3, 3]) # Inverse translation, takes into account the rotation
        return inv_Tr

    def make_point_pillars(self, points: np.ndarray, print_flag: bool = False):

        assert points.ndim == 2
        assert points.shape[1] == 4
        assert points.dtype == np.float32

        pillars, indices = createPillars(points,
                                         self.max_points_per_pillar,
                                         self.max_pillars,
                                         self.x_step,
                                         self.y_step,
                                         self.x_min,
                                         self.x_max,
                                         self.y_min,
                                         self.y_max,
                                         self.z_min,
                                         self.z_max,
                                         print_flag)

        return pillars, indices

    def make_ground_truth(self, labels: List[Label3D]):

        # filter labels by classes (cars, pedestrians and Trams)
        # Label has 4 properties (Classification (0th index of labels file),
        # centroid coordinates, dimensions, yaw)
        labels = list(filter(lambda x: x.classification in self.classes, labels))

        if len(labels) == 0:
            pX, pY = int(self.Xn / self.downscaling_factor), int(self.Yn / self.downscaling_factor)
            a = int(self.anchor_dims.shape[0])
            return np.zeros((pX, pY, a), dtype='float32'), np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_dims), dtype='float32'), np.zeros((pX, pY, a), dtype='float32'), \
                   np.zeros((pX, pY, a, self.nb_classes), dtype='float64')

        # For each label file, generate these properties except for the Don't care class
        target_positions = np.array([label.centroid for label in labels], dtype=np.float32)
        target_dimension = np.array([label.dimension for label in labels], dtype=np.float32)
        target_yaw = np.array([label.yaw for label in labels], dtype=np.float32)
        target_class = np.array([self.classes[label.classification] for label in labels], dtype=np.int32)

        # assert np.all(target_yaw >= -np.pi) & np.all(target_yaw <= np.pi)
        assert len(target_positions) == len(target_dimension) == len(target_yaw) == len(target_class)

        target, pos, neg = createPillarsTarget(target_positions,
                                               target_dimension,
                                               target_yaw,
                                               target_class,
                                               self.anchor_dims,
                                               self.anchor_z,
                                               self.anchor_yaw,
                                               self.positive_iou_threshold,
                                               self.negative_iou_threshold,
                                               self.nb_classes,
                                               self.downscaling_factor,
                                               self.x_step,
                                               self.y_step,
                                               self.x_min,
                                               self.x_max,
                                               self.y_min,
                                               self.y_max,
                                               self.z_min,
                                               self.z_max,
                                               False)
        self.pos_cnt += pos
        self.neg_cnt += neg

        # return a merged target view for all objects in the ground truth and get categorical labels
        sel = select_best_anchors(target)
        ohe = tf.keras.utils.to_categorical(sel[..., 9], num_classes=self.nb_classes, dtype='float64')

        return sel[..., 0], sel[..., 1:4], sel[..., 4:7], sel[..., 7], sel[..., 8], ohe


class SimpleDataGenerator(DataProcessor, Sequence):
    """ Multiprocessing-safe data generator for training, validation or testing, without fancy augmentation """

    def __init__(self, data_reader: DataReader, batch_size: int, lidar_files: List[str], label_files: List[str] = None,
                 calibration_files: List[str] = None):
        super(SimpleDataGenerator, self).__init__()
        self.data_reader = data_reader
        self.batch_size = batch_size
        self.lidar_files = lidar_files
        self.label_files = label_files
        self.calibration_files = calibration_files

        assert (len(self.calibration_files) == len(self.lidar_files))

        if self.label_files is not None:
            assert len(self.calibration_files) == len(self.lidar_files)
            assert len(self.label_files) == len(self.lidar_files)

    def __len__(self):
        return len(self.lidar_files) // self.batch_size

    def __getitem__(self, batch_id: int):
        file_ids = np.arange(batch_id * self.batch_size, self.batch_size * (batch_id + 1))
        #         print("inside getitem")
        pillars = []
        voxels = []
        occupancy = []
        position = []
        size = []
        angle = []
        heading = []
        classification = []

        for i in file_ids:
            lidar = self.data_reader.read_lidar(self.lidar_files[i])
            # For each file, dividing the space into a x-y grid to create pillars
            # Voxels are the pillar ids
            pillars_, voxels_ = self.make_point_pillars(lidar)

            pillars.append(pillars_)
            voxels.append(voxels_)

            if self.label_files is not None:
                label = self.data_reader.read_label(self.label_files[i])
                # R, t = self.data_reader.read_calibration(self.calibration_files[i])
                # label_transformed = self.transform_labels_into_lidar_coordinates(label, R, t)

                P2, R0, Tr_velo_to_cam = self.data_reader.read_calibration(self.calibration_files[i]) # Correct calibration reading
                # Labels are transformed into the lidar coordinate bounding boxes
                # Label has 7 values, centroid, dimensions and yaw value.
                label_transformed = self.camera_to_lidar(label, P2, R0, Tr_velo_to_cam) # Correct transformation

                # These definitions can be found in point_pillars.cpp file
                # We are splitting a 10 dim vector that contains this information.
                occupancy_, position_, size_, angle_, heading_, classification_ = self.make_ground_truth(
                    label_transformed)

                occupancy.append(occupancy_)
                position.append(position_)
                size.append(size_)
                angle.append(angle_)
                heading.append(heading_)
                classification.append(classification_)

        pillars = np.concatenate(pillars, axis=0)
        voxels = np.concatenate(voxels, axis=0)

        if self.label_files is not None:
            occupancy = np.array(occupancy)
            position = np.array(position)
            size = np.array(size)
            angle = np.array(angle)
            heading = np.array(heading)
            classification = np.array(classification)
            return [pillars, voxels], [occupancy, position, size, angle, heading, classification]
        else:
            return [pillars, voxels]

    def on_epoch_end(self):
        #         print("inside epoch")
        if self.label_files is not None:
            self.lidar_files, self.label_files, self.calibration_files = \
                shuffle(self.lidar_files, self.label_files, self.calibration_files)
