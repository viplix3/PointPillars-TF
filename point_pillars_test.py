import unittest
import os
import cv2
import logging
import numpy as np
import tensorflow as tf
from easydict import EasyDict as edict

from config import Parameters
from processors import DataProcessor
from readers import KittiDataReader
from inference_utils import BBox, draw_projected_box3d

class PointPillarsTest(unittest.TestCase):

    def setUp(self):
        self.data_processor = DataProcessor()
        self.data_reader = KittiDataReader()
        # self.file_name = "000000" # single ped test image
        self.file_name = "000020" # single car test image
        # self.file_name = "000067" # multi-object file

        root_path = "/media/ADAS1/ADAS_LiDAR/KITTI/Extracted_Data/testing/data/object"
        # root_path = r"E:\LiDAR_OD\KITTI\data\test_data\data\object"
        image_file = os.path.join(root_path, "image_2", "{}.png".format(self.file_name))
        lidar_data_file = os.path.join(root_path, "velodyne", "{}.bin".format(self.file_name))
        lidar_label_file = os.path.join(root_path, "label_2", "{}.txt".format(self.file_name))
        calib_file = os.path.join(root_path, "calib", "{}.txt".format(self.file_name))

        # LiDAR data checks
        self.lidar_data = self.data_reader.read_lidar(lidar_data_file)
        if self.file_name == "000000":
            assert self.lidar_data.shape == (115384, 4)
        elif self.file_name == "000020":
            assert self.lidar_data.shape == (126456, 4)
        else:
            assert self.lidar_data.shape == (114305, 4)

        # GT labels check
        self.lidar_label = self.data_reader.read_label(lidar_label_file)
        if self.file_name == "000000" or self.file_name == "000020":
            assert len(self.lidar_label) == 1
        else:
            assert len(self.lidar_label) == 7

        # Calibration files check
        calib_data = self.data_reader.read_calibration(calib_file)
        self.calib = edict()
        self.calib["P2"] = calib_data[0]
        self.calib["R0"] = calib_data[1]
        self.calib["Tr_velo_to_cam"] = calib_data[2]
        assert np.shape(self.calib["P2"]) == (3, 4)
        assert np.shape(self.calib["R0"]) == (3, 3)
        assert np.shape(self.calib["Tr_velo_to_cam"]) == (3, 4)

        # Image file reading
        self.image_data = cv2.imread(image_file)

    def test_pillar_creation(self):
        pillars, indices = self.data_processor.make_point_pillars(points=self.lidar_data)

        assert pillars.shape == (1, Parameters.max_pillars, Parameters.max_points_per_pillar,
                                    Parameters.nb_features)
        assert pillars.dtype == np.float32
        assert indices.shape == (1, Parameters.max_pillars, 3)
        assert indices.dtype == np.int32

        pillars = tf.constant(pillars, dtype=tf.float32)
        indices = tf.constant(indices, dtype=tf.int32)
        feature_map = tf.scatter_nd(indices, tf.reduce_mean(pillars, axis=2), (1, 504, 504, 7))[0]
        arr = feature_map
        assert (arr.shape == (504, 504, 7))
    

    def test_gt_in_camera_coordinate_frame(self):
        # 3d bb corner coordinates in camera coordinate frame, coordinate system is at the bottom center of box
        # x-axis -> right (length), y-axis -> bottom (height), z-axis -> forward (width)
        #     7 -------- 4
        #    /|         /|
        #   6 -------- 5 .
        #   | |        | |
        #   . 3 -------- 0
        #   |/         |/
        #   2 -------- 1
        for bbox_idx in range(len(self.lidar_label)):
            x = self.lidar_label[bbox_idx].centroid[0]
            y = self.lidar_label[bbox_idx].centroid[1]
            z = self.lidar_label[bbox_idx].centroid[2]
            l = self.lidar_label[bbox_idx].dimension[2]
            w = self.lidar_label[bbox_idx].dimension[1]
            h = self.lidar_label[bbox_idx].dimension[0]

            bb_3d_x_corner_coordinates = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            bb_3d_y_corner_coordinates = [0, 0, 0, 0, -h, -h, -h, -h]
            bb_3d_z_corner_coordinates = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]
            
            # box rotation by yaw angle
            yaw_rotation_matrix = BBox.get_y_axis_rotation_matrix(self.lidar_label[bbox_idx].yaw)
            bb_3d_corners = np.vstack([ bb_3d_x_corner_coordinates, bb_3d_y_corner_coordinates,
                                        bb_3d_z_corner_coordinates])

            bb_3d_corners = yaw_rotation_matrix @ bb_3d_corners

            # box translation by centroid coordinates
            bb_3d_corners[0, :] = bb_3d_corners[0, :] + x
            bb_3d_corners[1, :] = bb_3d_corners[1, :] + y
            bb_3d_corners[2, :] = bb_3d_corners[2, :] + z

            # projection box to LiDAR space using Tr_velo_to_cam
            self.camera_to_lidar_box_projection(bb_3d_corners)

            # camera coordinate frame to image coordinate frame box projection
            bbox_2d_image, bbox_corners_image = BBox.camera_to_image(bb_3d_corners, self.calib["P2"])

            # Plotting the 3D BB
            self.image_data = draw_projected_box3d(self.image_data, bbox_corners_image)
        cv2.imwrite("{}-cam_to_image_projection.png".format(self.file_name), self.image_data)
        logging.info("Image {}.png writted at path {} with 3D-BB projected on image plane.".format(
            self.file_name, os.getcwd()))
        

    def test_gt_in_lidar_coordinate_frame(self):
        # 3d bb corner coordinates in LiDAR coordinate frame, coordinate system is at the bottom center of box
        # x-axis -> forward (width), y-axis -> left (length), z-axis -> top (height)
        #     7 -------- 4
        #    /|         /|
        #   6 -------- 5 .
        #   | |        | |
        #   . 3 -------- 0
        #   |/         |/
        #   2 -------- 1

        # convert GT from camera to LiDAR coordinate space
        bb_lidar_coordiante_frame = self.lidar_label.copy()
        bb_lidar_coordiante_frame = self.data_processor.camera_to_lidar(bb_lidar_coordiante_frame, 
                    P=self.calib["P2"], R=self.calib["R0"], V2C=self.calib["Tr_velo_to_cam"])
        assert len(bb_lidar_coordiante_frame) == len(self.lidar_label)

        for bbox_idx in range(len(self.lidar_label)):
            x = bb_lidar_coordiante_frame[bbox_idx].centroid[0]
            y = bb_lidar_coordiante_frame[bbox_idx].centroid[1]
            z = bb_lidar_coordiante_frame[bbox_idx].centroid[2]
            l = bb_lidar_coordiante_frame[bbox_idx].dimension[0]
            w = bb_lidar_coordiante_frame[bbox_idx].dimension[1]
            h = bb_lidar_coordiante_frame[bbox_idx].dimension[2]

            # bb_3d_x_corner_coordinates = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
            # bb_3d_y_corner_coordinates = [0, 0, 0, 0, -h, -h, -h, -h]
            # bb_3d_z_corner_coordinates = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

            bb_3d_x_corner_coordinates = [w/2., -w/2., -w/2., w/2., w/2., -w/2., -w/2., w/2.]
            bb_3d_y_corner_coordinates = [-l/2., -l/2., l/2., l/2., -l/2., -l/2., l/2., l/2.]
            bb_3d_z_corner_coordinates = [0, 0, 0, 0, h, h, h, h]
            
            # box rotation by yaw angle
            yaw_rotation_matrix = BBox.get_z_axis_rotation_matrix(bb_lidar_coordiante_frame[bbox_idx].yaw)
            bb_3d_corners = np.vstack([ bb_3d_x_corner_coordinates, bb_3d_y_corner_coordinates,
                                        bb_3d_z_corner_coordinates])

            bb_3d_corners = yaw_rotation_matrix @ bb_3d_corners

            # box translation by centroid coordinates
            bb_3d_corners[0, :] = bb_3d_corners[0, :] + x
            bb_3d_corners[1, :] = bb_3d_corners[1, :] + y
            bb_3d_corners[2, :] = bb_3d_corners[2, :] + z

            print("LiDAR projected box using yaw angle")
            print(bb_3d_corners, end="\n\n")


    def camera_to_lidar_box_projection(self, bb_3d_corners_camera_coordiante_frame):
        print("Box in camera coordinate frame")
        print(bb_3d_corners_camera_coordiante_frame, end="\n\n")

        bb_3d_x_coord, bb_3d_y_coord, bb_3d_z_coord = np.split(bb_3d_corners_camera_coordiante_frame, 3, axis=0)
        bb_3d_corners_lidar_coordiante_frame = np.concatenate((bb_3d_z_coord, -bb_3d_x_coord, -bb_3d_y_coord), axis=0)

        bb_3d_corners_lidar_coordiante_frame = bb_3d_corners_camera_coordiante_frame

        bbox_3d_homogenous = np.concatenate((bb_3d_corners_lidar_coordiante_frame, 
                np.ones((1, 8))), axis=0)
        R_ref_rect = np.zeros((4, 4))
        R_ref_rect[:3, :3] = self.calib["R0"]
        R_ref_rect[3, 3] = 1
        bbox_3d_homogenous = np.linalg.inv(R_ref_rect) @ bbox_3d_homogenous # [4, 4] @ [4, 8] -> [4, 8]
        bbox_3d_homogenous = np.matmul(DataProcessor.inverse_rigid_trans(self.calib["Tr_velo_to_cam"]), 
                                bbox_3d_homogenous) # [3, 4] @ [4, 8] -> [3, 8]
        print("LiDAR projected box using projection matrices")
        print(bbox_3d_homogenous, end="\n\n\n")


    # @staticmethod
    # def test_pillar_target_creation():

    #     dims = np.array([[3.7, 1.6, 1.4], [3.7, 1.6, 1.4], [0.8, 0.6, 1.7]], dtype=np.float32)
    #     posn = np.array([[50, 10, 0], [20, 0, 0], [30, 5, 0]], dtype=np.float32)
    #     yaws = np.array([0, 0, 90], dtype=np.float32)

    #     target = createPillarsTarget(posn,
    #                                  dims,
    #                                  yaws,
    #                                  np.array([1, 1, 2], dtype=np.int32),
    #                                  dims[[0, 2]],
    #                                  np.array([0, 0], dtype=np.float32),
    #                                  np.array([0, 90], dtype=np.float32),
    #                                  0.5,
    #                                  0.4,
    #                                  10,
    #                                  2,
    #                                  0.1,
    #                                  0.1,
    #                                  0,
    #                                  80,
    #                                  -40,
    #                                  40,
    #                                  -3,
    #                                  1,
    #                                  True)

    #     assert target.shape == (3, 400, 400, 2, 10)
    #     assert (target[..., 0] == 1).sum() == 83

    #     selected = target[..., 0:1].argmax(axis=0)
    #     target = select(target, selected)
    #     assert (target.shape == (400, 400, 2, 10))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s]: %(message)s")
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    unittest.main()
