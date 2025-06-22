from typing import Any, Dict

import yaml

CONFIG_PATH = "./config/default.yml"
config = None


def load_config(config_path: str) -> Dict[str, Any]:
    """Load the YAML configuration file as a dictionary for usage.

    Args:
        config_path (str, optional): The path of the config file.

    Returns:
        Dict[str, Any]: The loaded config.
    """
    global config

    if config is None:
        # If no path is sent.
        if config_path is None:
            config_path = CONFIG_PATH

        with open(config_path) as f:
            config = yaml.safe_load(f)

    return config
