
from hydroecolstm_lite.model.lstm_linears import Lstm_Linears

def create_model(config, state_dict=None):
    
    # Create the model
    model = Lstm_Linears(config)
     
    # Assign state dict
    if state_dict is not None: 
        model.load_state_dict(state_dict)
    
    return model