
"""Model construction helpers.

This module exposes a small factory function `create_model` that instantiates
the LSTM+FC architecture used in the project and optionally loads a provided
state dictionary.
"""

from hydroecolstm_lite.model.lstm import lstm
from hydroecolstm_lite.utility.load_state_dict import load_state_dict


def create_model(config, state_dict_file=None):
    """Create and optionally initialize a model from configuration.

    Parameters
    ----------
    config : dict
        Model configuration passed to `lstm`.
    state_dict : dict, optional
        Optional PyTorch state dict to load into the model.

    Returns
    -------
    nn.Module
        Instantiated model ready for training or inference.
    """

    # Create the model
    model = lstm(config)

    # Assign state dict
    if state_dict_file is not None:
        
        model = load_state_dict(model, state_dict_file)

    return model