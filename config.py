import numpy as np


class GridParameters:
    x_min = 0.0
    x_max = 48.64
    x_step = 0.16

    y_min = -24.32
    y_max = 24.32
    y_step = 0.16

    z_min = -2.5
    z_max = 0.5

    # derived parameters
    Xn_f = float(x_max - x_min) / x_step
    Yn_f = float(y_max - y_min) / y_step
    Xn = int(Xn_f)
    Yn = int(Yn_f)

    def __init__(self):
        super(GridParameters, self).__init__()


class DataParameters:

    classes = {"Pedestrian":        0,
               "Person_sitting":    0,
               "Cyclist":           1,
               }

    nb_classes = len(np.unique(list(classes.values())))
    assert nb_classes == np.max(np.unique(list(classes.values()))) + 1, 'Starting class indexing at zero.'

    def __init__(self):
        super(DataParameters, self).__init__()


class NetworkParameters:

    max_points_per_pillar = 100
    max_pillars = 12000
    nb_features = 9
    nb_channels = 64
    downscaling_factor = 2

    # length, width, height, z-center, orientation
    anchor_dims = np.array([[0.8, 0.6, 1.73, -0.6, 0], # pedestrian-anchor
                            [0.8, 0.6, 1.73, -0.6, 1.5708], # pedestrian-anchor rotated by 90 degrees
                            [1.76, 0.6, 1.73, -0.6, 0], # cyclist-anchor rotated by 90 degrees
                            [1.76, 0.6, 1.73, -0.6, 1.5708], # cyclist-anchor rotated by 90 degrees
                            ], dtype=np.float32).tolist()
    # TODO: Add cyclist anchor
    nb_dims = 3

    positive_iou_threshold = 0.5
    negative_iou_threshold = 0.35
    batch_size = 4
    total_training_epochs = 160
    iters_to_decay = 66780 # 101040.    # 15 * 4 * ceil(6733. / 4) --> every 15 epochs on 6733 kitti samples, cf. pillar paper
    learning_rate = 2e-4
    decay_rate = 1e-8
    L1 = 0
    L2 = 0
    alpha = 0.25
    gamma = 2.0
                            # original pillars paper values
    focal_weight = 3.0      # 1.0
    loc_weight = 2.0        # 2.0
    size_weight = 2.0       # 2.0
    angle_weight = 1.0      # 2.0
    heading_weight = 0.2    # 0.2
    class_weight = 0.5      # 0.2

    def __init__(self):
        super(NetworkParameters, self).__init__()


class Parameters(GridParameters, DataParameters, NetworkParameters):

    def __init__(self):
        super(Parameters, self).__init__()
