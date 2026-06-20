

import torch
from collections import OrderedDict

def load_state_dict(model, state_dict_file=None):
    
    if state_dict_file.exists():

        # Load state dict to CPU first for compatibility
        state = torch.load(state_dict_file, map_location="cpu")

        # If checkpoint is a wrapper dict, extract the actual state_dict
        if isinstance(state, dict) and (
                "state_dict" in state or "model_state_dict" in state):
            
            state = state.get("state_dict", state.get("model_state_dict"))

        # Strip 'module.' prefix from keys saved from DataParallel wrappers
        new_state = OrderedDict()

        for k, v in state.items():
            name = k[7:] if k.startswith("module.") else k
            new_state[name] = v

        model.load_state_dict(new_state)
        
    return model
    
        