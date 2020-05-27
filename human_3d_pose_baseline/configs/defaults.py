# references:
# https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/predict_3dpose.py#L27-L60
# https://github.com/weigq/3d_pose_baseline_pytorch/blob/master/opt.py

from yacs.config import CfgNode as CN

_C = CN()

# Input data.
_C.DATA = CN()
_C.DATA.HM36M_DIR = "dataset/h36m"
_C.DATA.POSE_IN_CAMERA_FRAME = True  # Learn 3d poses in camera coordinates.
_C.DATA.ACTIONS = []  # Actions to load. If empty, load all actions.

# Dataloader.
_C.LOADER = CN()
_C.LOADER.TRAIN_BATCHSIZE = 64
_C.LOADER.TRAIN_NUM_WORKERS = 8
_C.LOADER.TEST_BATCHSIZE = 64
_C.LOADER.TEST_NUM_WORKERS = 8

# Model architecture.
_C.MODEL = CN()
_C.MODEL.LINEAR_SIZE = 1024
_C.MODEL.NUM_STAGES = 2
_C.MODEL.DROPOUT_PROB = 0.5
_C.MODEL.PREDICT_14 = False
_C.MODEL.WEIGHT = ""

# Model optimization settings.
_C.SOLVER = CN()
_C.SOLVER.EPOCHS = 200
_C.SOLVER.LR = 1e-3
_C.SOLVER.LR_DECAY_STEP = 100000
_C.SOLVER.LR_DECAY_GAMMA = 0.96

# Model evaluation settings.
_C.EVAL = CN()
_C.EVAL.METRICS_TO_LOG = [
    "MPJPE",
    "MPJPE/Directions",
    "MPJPE/Discussion",
    "MPJPE/Eating",
    "MPJPE/Greeting",
    "MPJPE/Phoning",
    "MPJPE/Photo",
    "MPJPE/Posing",
    "MPJPE/Purchases",
    "MPJPE/Sitting",
    "MPJPE/SittingDown",
    "MPJPE/Smoking",
    "MPJPE/Waiting",
    "MPJPE/WalkDog",
    "MPJPE/Walking",
    "MPJPE/WalkTogether",
]
_C.EVAL.APPLY_PROCRUSTES_ALIGNMENT = False

# Misc.
_C.OUTPUT_DIR = "./results"
_C.USE_CUDA = True


def get_default_config():
    """Get default configutation.

    Returns:
        (yacs.config.CfgNode): Default configuration.
    """
    return _C.clone()
