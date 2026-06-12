import torch

def get_device(config):
    
    device_cfg = config.get('device', 'cpu')
    
    if isinstance(device_cfg, str) and device_cfg.lower() in ('auto', 'default'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    elif isinstance(device_cfg, str) and device_cfg.lower() in ('cuda', 'gpu'):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    else:
        # fallback to cpu for any other value
        device = torch.device('cpu')
        
    return device


    