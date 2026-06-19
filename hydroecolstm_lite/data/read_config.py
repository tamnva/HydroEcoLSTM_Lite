# -*- coding: utf-8 -*-

"""Configuration reading utilities.

This module provides a small helper to read YAML configuration files and
convert date strings into pandas.Timestamp objects. It performs basic
validation for presence of required keys used by the package.
"""

import yaml
import pandas as pd


def read_config(config_file):

    """Read a YAML configuration file and validate required keys.

    Parameters
    ----------
    config_file : str or Path
        Path to the YAML configuration file.

    Returns
    -------
    dict
        Parsed configuration dictionary with some fields converted to
        appropriate Python types (e.g., datetimes and stringified ids).
    """

    with open(config_file, "r") as file:
        config = yaml.safe_load(file)

    # All required keywords
    keys = [
        "id_train",
        "id_valid",
        "id_test",
        "input_timeseries_features",
        "target_features",
        "train_period",
        "test_period",
        "n_epochs",
        "learning_rate",
        "timeseries_data_file",
    ]

    for key in keys:
        if key not in config.keys():
            raise NameError(f"Error in configuration file, keyword '{key}' is missing")

    if "train_period" in config.keys():
        config["train_period"] = pd.to_datetime(config["train_period"], format="%Y-%m-%d %H:%M")

    if "valid_period" in config.keys():
        config["valid_period"] = pd.to_datetime(config["valid_period"], format="%Y-%m-%d %H:%M")
    if "test_period" in config.keys():
        config["test_period"] = pd.to_datetime(config["test_period"], format="%Y-%m-%d %H:%M")

    if "inference_period" in config.keys():
        config["inference_period"] = pd.to_datetime(config["inference_period"], format="%Y-%m-%d %H:%M")
        config["id_inference"] = [str(ids) for ids in config["id_inference"]]

    # To string
    config["id_train"] = [str(ids) for ids in config["id_train"]]
    config["id_test"] = [str(ids) for ids in config["id_test"]]
    config["id_valid"] = [str(ids) for ids in config["id_valid"]]

    return config
    return config