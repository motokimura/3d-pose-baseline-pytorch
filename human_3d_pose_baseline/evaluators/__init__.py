from .evaluator import Human36M_JointErrorEvaluator


def get_evaluator(config, human36m):
    """Get Human3.6M joint error evaluator.

    Args:
        config (yacs.config.CfgNode): Configuration.
        human36m (Human36MDatasetHandler): Human3.6M dataset.

    Returns:
        Human36M_JointErrorEvaluator: Human3.6M joint error evaluator.
    """
    evaluator = Human36M_JointErrorEvaluator(
        human36m, predict_14=config.MODEL.PREDICT_14
    )
    return evaluator
