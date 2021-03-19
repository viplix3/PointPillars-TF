import os
from glob import glob
import numpy as np
import tensorflow as tf
from processors import DataProcessor
from inference_utils import generate_bboxes_from_pred, rotational_nms, gather_boxes_in_kitti_format
from readers import KittiDataReader
from config import Parameters
from network import build_point_pillar_graph
import argparse
import logging
from easydict import EasyDict as edict
import time

def generate_config_from_cmd_args():
    parser = argparse.ArgumentParser(description='PointPillars inference on test data.')
    parser.add_argument('--gpu_idx', default=0, type=int, required=False, 
        help='GPU index to use for inference')
    parser.add_argument('--data_root', default=None, required=True, 
        help='Test data root path holding folders velodyne, calib')
    parser.add_argument('--result_dir', default=None, required=True,
        help='Path for dumping result labels in KITTI format')
    parser.add_argument('--model_path', default='./logs/model.h5', type=str, required=False,
        help='Path to the model weights to be used for inference')
    parser.add_argument('--occ_thresh', default=0.7, type=float, required=False,
        help='Occlusion threshold for predicted boxes')
    parser.add_argument('--nms_thresh', default=0.5, type=float, required=False, 
        help='IoU threshold for NMS')
    
    configs = edict(vars(parser.parse_args()))
    return configs

def get_file_names(data_root_path):
    lidar_file_names = sorted(glob(os.path.join(data_root_path, "velodyne", "*.bin")))
    calib_file_names = sorted(glob(os.path.join(data_root_path, "calib", "*.txt")))

    if(len(lidar_file_names) != len(calib_file_names)):
        logging.error("Input dirs require equal number of files")
        exit()

    return lidar_file_names, calib_file_names

def load_model_and_run_inference(configs):
    params = Parameters() # Load all model related parameters
    pillar_net = build_point_pillar_graph(params, batch_size=1)

    logging.info("Loading model from path: {}".format(configs.model_path))
    pillar_net.load_weights(configs.model_path)
    logging.info("Model loaded.")

    lidar_files, calibration_files = get_file_names(configs.data_root)
    logging.debug("1st LiDAR data file: {}".format(lidar_files[0]))
    logging.debug("1st calibration file: {}".format(calibration_files[0]))

    data_reader = KittiDataReader()
    point_cloud_processor = DataProcessor()
    model_exec_time = []

    for idx in range(len(lidar_files)):
        logging.debug("Running for file: {}".format(lidar_files[idx].split('.')[0]))
        lidar_data = data_reader.read_lidar(lidar_files[idx])
        P2, R0, Tr_velo_to_cam = data_reader.read_calibration(calibration_files[idx])

        pillars, voxels = point_cloud_processor.make_point_pillars(points=lidar_data, print_flag=True)
        start = time.time()
        occupancy, position, size, angle, heading, classification = pillar_net.predict([pillars, voxels])
        stop = time.time()
        logging.debug("Single frame PointPillars inference time: {}".format(stop-start))
        model_exec_time.append(stop-start)

        logging.debug("occupancy shape: {}".format(occupancy.shape))
        logging.debug("position shape: {}".format(position.shape))
        logging.debug("size shape: {}".format(size.shape))
        logging.debug("angle shape: {}".format(angle.shape))
        logging.debug("heading shape: {}".format(heading.shape))
        logging.debug("classification shape: {}".format(classification.shape))

        if occupancy.shape[0] == 1:
            logging.debug("Single image inference, reducing batch dim for output tensors")
            occupancy = np.squeeze(occupancy, axis=0)
            position = np.squeeze(position, axis=0)
            size = np.squeeze(size, axis=0)
            angle = np.squeeze(angle, axis=0)
            heading = np.squeeze(heading, axis=0)
            classification = np.squeeze(classification, axis=0)

        start = time.time()
        boxes = generate_bboxes_from_pred(occupancy, position, size, angle, heading, classification, 
            params.anchor_dims, occ_threshold=configs.occ_thresh)
        stop = time.time()
        confidences = [float(box.conf) for box in boxes]
        logging.debug("Single frame box extraction time: {}".format(stop-start))

        logging.debug("Number of predictied boxes pre-nms: {}".format(len(boxes)))
        for idx in range(len(boxes)):
            logging.debug("{:04d}: {}".format(idx, boxes[idx]))

        start = time.time()
        nms_indices = rotational_nms(boxes, confidences, occ_threshold=configs.occ_thresh, nms_iou_thr=configs.nms_thresh)
        stop = time.time()
        logging.debug("Single frame rotational NMS time: {}".format(stop-start))

        logging.debug("Number of boxes post-nms: {}".format(len(nms_indices)))
        for idx in nms_indices:
            logging.debug("{:04d}: {}".format(idx, boxes[idx]))

        prediction_in_kitti_format = gather_boxes_in_kitti_format(boxes, nms_indices, P2, R0, Tr_velo_to_cam)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - [%(levelname)s]: %(message)s")
    pred_config = generate_config_from_cmd_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pred_config.gpu_idx)

    logging.info("Results will be saved at path: {}".format(pred_config.result_dir))

    load_model_and_run_inference(pred_config)