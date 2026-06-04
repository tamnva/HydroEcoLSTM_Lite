
import torch
from torch import nn
import pandas as pd
from hydroecolstm_lite.model.linears import Linears

# LSTM + Linears
class Lstm_Linears(nn.Module):
    def __init__(self, config, **kwargs):
        
        super(Lstm_Linears, self).__init__()

        self.output_size = len(config["target_features"])
        self.hidden_size = config["hidden_size"]
        self.num_layers = config["num_layers"]
        self.dropout = config["dropout"]*min(1.0, self.num_layers - 1.0)
        self.linears_num_layers = config["Regression"]["num_layers"]
        self.linears_activation_function = config["Regression"]["activation_function"]
        self.linears_num_neurons = self.find_num_neurons(config=config)
        self.input_features = (config["input_dynamic_features"] + 
                               config["input_static_features"])
        
        # Standard LSTM from torch
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
     
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        
        if isinstance(x, pd.DataFrame):
            ids = x['id'].unique()
            
            for id in ids:
                y_predict_id, _ = self.lstm(torch.tensor(
                    x[x['id'] == id][self.input_features].values,
                    dtype=torch.float32))
                
                if id == ids[0]:
                    y_predict = y_predict_id
                else:
                    y_predict = torch.cat([y_predict, y_predict_id], dim=0)
                
        else:
            y_predict, _ = self.lstm(x)
        
        return self.linear(y_predict) 
    
    
    # Find number of neuron in each linear layers, including the input layer
    def find_num_neurons(self, config) -> int:
        # First number of neurons from the input layers ()
        num_neurons = [self.hidden_size]

        if "Regression" in config:
            if len(config["Regression"]["num_neurons"]) > 1:
                for i in range(len(config["Regression"]["num_neurons"])-1):
                    num_neurons.append(config["Regression"]["num_neurons"][i])

        num_neurons.append(self.output_size)

        return num_neurons