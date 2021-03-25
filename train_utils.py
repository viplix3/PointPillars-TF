import os
import logging

import tensorflow as tf

def parse_data_ids(imageset_root_path: str):
    """ Parses data ids for training/validation data from the provided path """
    train_imageset_file = os.path.join(imageset_root_path, "train.txt")
    valid_imageset_file = os.path.join(imageset_root_path, "valid.txt")

    train_imageset_list, valid_imageset_list = [], []

    try:
        with open(train_imageset_file, 'r') as train_imageset:
            for image_id in train_imageset.readlines():
                image_id = image_id.strip()
                train_imageset_list.append(image_id)
    except:
        logging.error("Couldn't load training imageset. Please check path: {}".format(
                        train_imageset_file))

    try:
        with open(valid_imageset_file, 'r') as valid_imageset:
            for image_id in valid_imageset.readlines():
                image_id = image_id.strip()
                valid_imageset_list.append(image_id)
    except:
        logging.error("Couldn't load validation imageset. Please check path: {}".format(
                        valid_imageset_file))
    return train_imageset_list, valid_imageset_list

def parse_data_from_ids(data_id_list, data_root_path):
    """ Parses path of velodyne point-cloud, labels and calib files for the provided list of file ids """
    lidar_files, label_files, calib_files= [], [], []

    for file_id in data_id_list:
        lidar_files.append(os.path.join(data_root_path, "velodyne", "{}.bin".format(file_id)))
        label_files.append(os.path.join(data_root_path, "label_2", "{}.txt".format(file_id)))
        calib_files.append(os.path.join(data_root_path, "calib", "{}.txt".format(file_id)))
        
    assert len(lidar_files) == len(label_files)
    assert len(label_files) == len(calib_files)
    return lidar_files, label_files, calib_files