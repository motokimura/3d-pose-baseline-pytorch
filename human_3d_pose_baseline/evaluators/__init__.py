from .evaluator import Human36M_JointErrorEvaluator


def get_evaluator(config, human36m):
    """[summary]

    Args:
        config ([type]): [description]
        human36m ([type]): [description]

    Returns:
        [type]: [description]
    """
    evaluator = Human36M_JointErrorEvaluator(
        human36m, predict_14=config.MODEL.PREDICT_14
    )
    return evaluator
