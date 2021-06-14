import os
from yacs.config import CfgNode as CN
from ..utils.log import log_or_print

__C = CN()

base_config = __C

__C.NAMES = CN()
__C.NAMES.BACKBONE = "resnet101"
__C.NAMES.DATASET = "MOT17"
__C.NAMES.CHECKPOINT = "coco_checkpoint.pth"
__C.NAMES.MODEL = "TADAM_MOT17.pth"

__C.PATHS = CN()
__C.PATHS.DATASET_ROOT = "datasets"
__C.PATHS.MODEL_ROOT = "output/models"
__C.PATHS.RESULT_ROOT = "output/results"
__C.PATHS.EVAL_ROOT = "../TrackEval"

__C.TRAINING = CN()
__C.TRAINING = CN()
__C.TRAINING.BATCH_SIZE = 2
__C.TRAINING.EPOCHS = 12
__C.TRAINING.ID_LOSS_RATIO = 0.1
__C.TRAINING.LR = 0.002
__C.TRAINING.LR_GAMMA = 0.5
__C.TRAINING.LR_STEP_SIZE = 3
__C.TRAINING.PRINT_FREQ = 200
__C.TRAINING.SAVE_FREQ = 3
__C.TRAINING.RANDOM_SEED = 123456
__C.TRAINING.MOMENTUM = 0.9
__C.TRAINING.WEIGHT_DECAY = 0.0005
__C.TRAINING.VIS_THRESHOLD = 0.1
__C.TRAINING.WARMUP_LR = 0.02
__C.TRAINING.WARMUP_EPOCHS = 3
__C.TRAINING.WORKERS = 2

__C.TRACKING = CN()
__C.TRACKING.MIN_BOX_SIZE = 3
__C.TRACKING.MIN_SCORE_ACTIVE_TRACKLET = 0.5
__C.TRACKING.MIN_SCORE_DETECTION = 0.05
__C.TRACKING.NMS_ACTIVE_TRACKLET = 0.7
__C.TRACKING.NMS_DETECTION = 0.3
__C.TRACKING.MIN_OVERLAP_AS_DISTRACTOR = 0.2
__C.TRACKING.MIN_RECOVER_GIOU = -0.4
__C.TRACKING.MIN_RECOVER_SCORE = 0.5
__C.TRACKING.MAX_LOST_FRAMES_BEFORE_REMOVE = 100


def load_config(config_file=None):
    """
        Load configurations
        Add or overwrite config from yaml file if specified
    """
    config = base_config
    if config_file is not None:
        config_file_path = os.path.join("lib", "configs", f"{config_file}.yaml")
        if os.path.isfile(config_file_path):
            config.merge_from_file(config_file_path)
            msg = f"Merged config from '{config_file_path}'"
        else:
            print(f"Cannot open the specified yaml config file '{config_file_path}'", level="critical")
            exit(0)
    else:
        msg = f"No yaml config file is specified. Using default config."
    return config, msg
