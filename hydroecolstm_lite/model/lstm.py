
import torch
from torch import nn

"""LSTM + fully-connected head model used for sequence regression.

This module implements `lstm`, a simple wrapper combining a
torch.nn.LSTM module followed by a fully-connected `Linears` head. The
architecture and dimensions are driven by the provided `config` dictionary.
"""

# lstm model
class lstm(nn.Module):
    """LSTM model with a configurable fully-connected output head.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys like `hidden_size`, `num_layers`,
        `dropout`, `Regression` block and feature lists.
    """

    def __init__(self, config, **kwargs):
        
        super(lstm, self).__init__()

        self.output_size = len(config["target_features"])
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]*min(1.0, self.num_layers - 1.0)
        self.input_timeseries_features = config["input_timeseries_features"]
        
        if "input_static_features" in config.keys(): 
            self.input_static_features = config["input_static_features"]
            self.input_features = (self.input_timeseries_features + 
                                   self.input_static_features)
        else:
            self.input_static_features = None
            self.input_features = self.input_timeseries_features
    
        
        # Columns of output tensor will be this order
        self.target_features = config["target_features"]
        
        # Standard LSTM from torch input = [batch, sequence, features]
        self.lstm = nn.LSTM(input_size=len(self.input_features), 
                            hidden_size=self.hidden_size, 
                            num_layers=self.num_layers,
                            dropout=self.dropout,
                            batch_first=True,
                            **kwargs)

        # Fully-connected layer connect hidden and output
        self.fc = nn.Linear(self.hidden_size, len(self.target_features))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape `(batch, seq_len, features)`.

        Returns
        -------
        torch.Tensor
            Predicted sequence with the same time dimension as the input.
        """
        
        y_predict, _ = self.lstm(x)

        return self.fc(y_predict)

