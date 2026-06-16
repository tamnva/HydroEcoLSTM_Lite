#!/usr/bin/env python
"""Run and orchestrate training for HydroEcoLSTM-Lite models.

This module provides a convenience function `run_config` which:
- reads and scales the dataset according to a configuration
- creates the model from configuration
- optionally loads an initial state dict checkpoint
- trains the model via the `Trainer` class

The `config` argument is expected to be a dict-like object with
keys used throughout the package (see `hydroecolstm_lite.data.read_config`).
Typical keys include data paths, model hyperparameters and an optional
`init_model_state_dict` entry pointing to a checkpoint file.
"""

import torch
from pathlib import Path
from collections import OrderedDict
from hydroecolstm_lite.data.read_data import read_train_valid_test_data
from hydroecolstm_lite.data.read_data import get_scaler_name
from hydroecolstm_lite.data.scaler import Scaler
from hydroecolstm_lite.model.create_model import create_model
from hydroecolstm_lite.train.trainer import Trainer


def run_config(config):
    """Run a full train/validation workflow from a configuration.

    Parameters
    ----------
    config : dict
        Configuration dictionary containing dataset paths, model options and
        training parameters. If `init_model_state_dict` is present it should
        be a list where the first element is a path to a PyTorch checkpoint.

    Returns
    -------
    tuple
        A 4-tuple `(data_scaled, scaler, model, trainer)` where:
        - `data_scaled` is a dict with scaled train/validation/test arrays
        - `scaler` contains fitted scaler objects for timeseries and static data
        - `model` is the trained PyTorch model instance
        - `trainer` is the `Trainer` instance used for training
    """

    data = read_train_valid_test_data(config)

    # Transform timeseries and static attributes
    col_scaler_timeseries = get_scaler_name(config, True)
    col_scaler_static = get_scaler_name(config, False)

    scaler = {}

    scaler["timeseries_data"] = Scaler()
    scaler["timeseries_data"].fit(data["timeseries_data_train"], col_scaler_timeseries)

    scaler["static_data"] = Scaler()
    scaler["static_data"].fit(data["static_data"], col_scaler_static)

    data_scaled = {}

    for key in data.keys():
        if "timeseries_data" in key:
            data_scaled[key] = scaler["timeseries_data"].transform(data[key])
        else:
            data_scaled[key] = scaler["static_data"].transform(data[key])

    # create model from config
    model = create_model(config)

    # optionally initialise model weights from a checkpoint
    if "init_model_state_dict" in config.keys():
        state_dict_file = Path(config["init_model_state_dict"][0])

        if state_dict_file.exists():

            # Load state dict to CPU first for compatibility
            state = torch.load(state_dict_file, map_location="cpu")

            # If checkpoint is a wrapper dict, extract the actual state_dict
            if isinstance(state, dict) and ("state_dict" in state or "model_state_dict" in state):
                state = state.get("state_dict", state.get("model_state_dict"))

            # Strip 'module.' prefix from keys saved from DataParallel wrappers
            new_state = OrderedDict()

            for k, v in state.items():
                name = k[7:] if k.startswith("module.") else k
                new_state[name] = v

            model.load_state_dict(new_state)

    trainer = Trainer(config, model)

    model = trainer.train(
        data_scaled["timeseries_data_train"],
        data_scaled["timeseries_data_valid"],
        data_scaled["static_data"],
    )

    return data_scaled, scaler, model, trainer

