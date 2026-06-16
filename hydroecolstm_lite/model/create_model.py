
"""Model construction helpers.

This module exposes a small factory function `create_model` that instantiates
the LSTM+FC architecture used in the project and optionally loads a provided
state dictionary.
"""

from hydroecolstm_lite.model.lstm_linears import Lstm_Linears


def create_model(config, state_dict=None):
    """Create and optionally initialize a model from configuration.

    Parameters
    ----------
    config : dict
        Model configuration passed to `Lstm_Linears`.
    state_dict : dict, optional
        Optional PyTorch state dict to load into the model.

    Returns
    -------
    nn.Module
        Instantiated model ready for training or inference.
    """

    # Create the model
    model = Lstm_Linears(config)

    # Assign state dict
    if state_dict is not None:
        model.load_state_dict(state_dict)

    return model