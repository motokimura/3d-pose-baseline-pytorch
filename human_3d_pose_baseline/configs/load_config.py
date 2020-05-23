import argparse

from yacs.config import CfgNode as CN

from .defaults import get_default_config


def load_config():
    """Load configuration.

    Returns:
        (yacs.config.CfgNode): Configuration.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", help="Path to YAML config file", type=str)
    parser.add_argument(
        "opts",
        default=None,
        help="parameter name and value pairs",
        nargs=argparse.REMAINDER,
    )
    args = parser.parse_args()

    config = get_default_config()
    if args.config:
        # Overwrite hyper parameters with the ones given by the YAML file.
        config.merge_from_file(args.config)
    if args.opts:
        # Overwrite hyper parameters with the ones given by command line args.
        config.merge_from_list(args.opts)
    config.freeze()

    return config
