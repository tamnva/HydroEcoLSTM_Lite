
import torch
from torch import nn
from hydroecolstm_lite.model.linears import Linears

"""LSTM + fully-connected head model used for sequence regression.

This module implements `Lstm_Linears`, a simple wrapper combining a
torch.nn.LSTM module followed by a fully-connected `Linears` head. The
architecture and dimensions are driven by the provided `config` dictionary.
"""

# LSTM + Linears
class Lstm_Linears(nn.Module):
    """LSTM model with a configurable fully-connected output head.

    Parameters
    ----------
    config : dict
        Configuration dictionary with keys like `hidden_size`, `num_layers`,
        `dropout`, `Regression` block and feature lists.
    """

    def __init__(self, config, **kwargs):
        
        super(Lstm_Linears, self).__init__()

        self.output_size = len(config["target_features"])
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]*min(1.0, self.num_layers - 1.0)
        self.linears_num_layers = config["Regression"]["num_layers"]
        self.linears_activation_function = config["Regression"]["activation_function"]
        self.linears_num_neurons = self.find_num_neurons(config=config)
        self.input_timeseries_features = config["input_timeseries_features"] 
        self.input_static_features = config["input_static_features"]
        
        # Columns of input tensor should follow this order
        self.input_features = (self.input_timeseries_features + 
                               self.input_static_features)
        
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
        self.linear = Linears(num_layers=self.linears_num_layers, 
                              activation_function=self.linears_activation_function,
                              num_neurons=self.linears_num_neurons)
     
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

        return self.linear(y_predict)
    
    
    # Find number of neuron in each linear layers, including the input layer
    def find_num_neurons(self, config) -> int:
        """Determine neuron counts for the fully-connected head.

        Returns a list of neuron counts used to construct the `Linears`
        module (including input and output sizes).
        """

        # First number of neurons from the input layers
        num_neurons = [self.hidden_size]

        if "Regression" in config:
            if len(config["Regression"]["num_neurons"]) > 1:
                for i in range(len(config["Regression"]["num_neurons"]) - 1):
                    num_neurons.append(config["Regression"]["num_neurons"][i])

        num_neurons.append(self.output_size)

        return num_neurons