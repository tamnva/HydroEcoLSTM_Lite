
from torch import nn
from typing import List

class Linears(nn.Module):
        
    def __init__(self, num_layers: int, activation_function: List[str], 
                 num_neurons: List[int]):
        super(Linears, self).__init__()
        
        # Activation functions
        activation_functions = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid(), 
                                "tanh": nn.Tanh(), "softtplus": nn.Softplus(), 
                                "identity": nn.Identity()}
        
        # Create list to store different linear layers
        layers = []
        
        # Create layers of user-defined network
        for i in range(num_layers): 
            layers.append(nn.Linear(num_neurons[i], num_neurons[i+1])) 
            
            if i < num_layers - 1: 
                layers.append(activation_functions[activation_function[i]])
        
        # Combined all layers together using sequential
        self.model = nn.Sequential(*layers)
        
    # Forward run
    def forward(self, x):
        # model output
        output = self.model(x)
        
        # Return output
        return output
    
    