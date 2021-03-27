import os
import time
import logging
import argparse
import numpy as np
import tensorflow as tf
from glob import glob
from easydict import EasyDict as edict

from config import Parameters
from loss import PointPillarNetworkLoss
from network import build_point_pillar_graph
from processors import SimpleDataGenerator
from readers import KittiDataReader

from train_utils import parse_data_ids, parse_data_from_ids


def generate_config_from_cmd_args():
    parser = argparse.ArgumentParser(description='PointPillars training')
    parser.add_argument('--gpu_idx', default=9, type=int, required=False, 
        help='GPU index to use for inference')
    parser.add_argument('--imageset_path', default=None, type=str, required=True,
        help='Path to the root folder containing train_img_ids.txt and val_img_ids.txt')
    parser.add_argument('--data_root', default=None, type=str, required=True, 
        help='Training/Validation data root path holding folders velodyne, calib')
    parser.add_argument('--model_root', default='./logs/', required=True,
        help='Path for dumping training logs and model weights')
    
    configs = edict(vars(parser.parse_args()))
    return configs


def train_PillarNet(configs):

    logging.info("Parsing train/val image ids")
    train_ids, valid_ids = parse_data_ids(configs.imageset_path)
    logging.debug("First 8 train dataset ids: {}".format(train_ids[:9]))
    logging.debug("Frist 8 valid dataset ids: {}".format(valid_ids[:9]))
    logging.info("Training/Validation imageset loaded.")

    params = Parameters()
    logging.info("Building PointPillars model...")
    pillar_net = build_point_pillar_graph(params)
    # inputs: [input_pillars, input_indices]    outputs: [occ, loc, size, angle, heading, clf]

    if os.path.exists(os.path.join(configs.model_root, "model.h5")):
        logging.info("Using pre-trained weights found at path: {}".format(configs.model_root))
        pillar_net.load_weights(os.path.join(configs.model_root, "model.h5"))
    else:
        logging.info("No pre-trained weights found. Initializing weights and training model.")

    loss = PointPillarNetworkLoss(params)
    optimizer = tf.keras.optimizers.Adam(lr=params.learning_rate, decay=params.decay_rate)
    pillar_net.compile(optimizer, loss=loss.losses())
    data_reader = KittiDataReader()

    train_lidar_files, train_label_files, train_calib_files = parse_data_from_ids(train_ids, configs.data_root)
    valid_lidar_files, valid_label_files, valid_calib_files = parse_data_from_ids(valid_ids, configs.data_root)
    
    logging.debug("First 8 train LiDAR pointcloud files: {}".format(train_lidar_files[:9]))
    logging.debug("First 8 training LiDAR label files: {}".format(train_label_files[:9]))
    logging.debug("First 8 training sensor calib files: {}".format(train_label_files[:9]))

    training_gen = SimpleDataGenerator(data_reader, params.batch_size, train_lidar_files, train_label_files, train_calib_files)
    validation_gen = SimpleDataGenerator(data_reader, params.batch_size, valid_lidar_files, valid_label_files, valid_calib_files)

    log_dir = configs.model_root
    epoch_to_decay = int(
        np.round(params.iters_to_decay / params.batch_size * int(np.ceil(float(len(train_label_files)) / params.batch_size))))
    callbacks = [
        tf.keras.callbacks.TensorBoard(log_dir=log_dir, update_freq=10),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(log_dir, "model.h5"),
                                           monitor='val_loss', save_best_only=True),
        tf.keras.callbacks.LearningRateScheduler(
            lambda epoch, lr: lr * 0.8 if ((epoch % epoch_to_decay == 0) and (epoch != 0)) else lr, verbose=True),
        # tf.keras.callbacks.EarlyStopping(patience=20, monitor='val_loss'),
    ]

    try:
        pillar_net.fit(training_gen,
                       validation_data = validation_gen,
                       steps_per_epoch=len(training_gen),
                       callbacks=callbacks,
                       use_multiprocessing=False,
                       epochs=int(params.total_training_epochs),
                       workers=4)
    except KeyboardInterrupt:
        model_str = "interrupted_%s.h5" % time.strftime("%Y%m%d-%H%M%S")
        pillar_net.save(os.path.join(log_dir, model_str))
        logging.warning("Interrupt. Saving output to {}".format(os.path.join(os.getcwd(), log_dir[1:], model_str)))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - [%(levelname)s]: %(message)s")
    train_configs = generate_config_from_cmd_args()
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_configs.gpu_idx)
    # tf.get_logger().setLevel("ERROR")

    logging.info("Model training logs and trained weights will be saved at path: {}".format(train_configs.model_root))
    train_PillarNet(train_configs)