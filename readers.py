import abc
from config import DataParameters
from typing import List

import numpy as np


class Label3D:
    def __init__(self, classification: str, centroid: np.ndarray, dimension: np.ndarray, yaw: float):
        self.classification = classification
        self.centroid = centroid
        self.dimension = dimension
        self.yaw = yaw

    def __str__(self):
        return "GT | Cls: %s, x: %f, y: %f, l: %f, w: %f, yaw: %f" % (
            self.classification, self.centroid[0], self.centroid[1], self.dimension[0], self.dimension[1], self.yaw)


class DataReader:

    @staticmethod
    @abc.abstractmethod
    def read_lidar(file_path: str) -> np.ndarray:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_label(file_path: str) -> List[Label3D]:
        raise NotImplementedError

    @staticmethod
    @abc.abstractmethod
    def read_calibration(file_path: str) -> np.ndarray:
        raise NotImplementedError


class KittiDataReader(DataReader, DataParameters):

    def __init__(self):
        super(KittiDataReader, self).__init__()

    @staticmethod
    def read_lidar(file_path: str):
        return np.fromfile(file_path, dtype=np.float32).reshape((-1, 4))

    @staticmethod
    def read_label(file_path: str):
        with open(file_path, "r") as f:

            lines = f.readlines()

            elements = []
            for line in lines:

                values = line.split()

                element = Label3D(
                    str(values[0]), # Class name
                    np.array(values[11:14], dtype=np.float32), # 3D BB (x, y, z) centroid in camera coordinates (in meters)
                    np.array(values[8:11], dtype=np.float32), # 3D BB height, width, length (in meters)
                    float(values[14]) # BB rotation along y-axis
                )

                if element.classification not in DataParameters.classes.keys():
                    continue
                else:
                    elements.append(element)

        return elements

    @staticmethod
    def read_calibration(file_path: str):
        with open(file_path, "r") as f:
            lines = f.readlines()
            P_2 = np.array(lines[2].split(": ")[1].split(" "), dtype=np.float32).reshape(3, 4) # Projection matrix for left-color camera
            R0 = np.array(lines[4].split(": ")[1].split(" "), dtype=np.float32).reshape(3, 3) # Rectification rotation matrix of left-color camera
            Tr_velo_to_cam = np.array(lines[5].split(": ")[1].split(" "), dtype=np.float32).reshape((3, 4)) # [R | T]
            # R, t = Tr_velo_to_cam[:, :3], Tr_velo_to_cam[:, 3] # First 3x3 is rotation and last vector is translation
            # return R, t
            return P_2, R0, Tr_velo_to_cam # Correct calibration information required for transformation