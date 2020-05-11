import logging.config
import os

import yaml


def setup_logging(
    default_path="logging.yaml", default_level=logging.INFO, env_key="LOG_CFG"
):
    """Setup application logging.

    Args:
        default_path (str, optional): logging configuration file. Defaults to
            "logging.yaml".
        default_level (int, optional): logging level. Defaults to logging.INFO.
        env_key (str, optional): environment key for configuration file.
            Defaults to "LOG_CFG".
    """
    path = os.getenv(env_key, default_path)
    if os.path.exists(path):
        with open(path, "rt") as f:
            config = yaml.safe_load(f.read())
        logging.config.dictConfig(config)
    else:
        logging.basicConfig(level=default_level)
