import torch


def get_device(config):
    """Return a torch.device according to `config['device']`.

    Accepted values in the config are:
    - 'auto' or 'default': select CUDA if available, otherwise CPU
    - 'cuda' or 'gpu': attempt to use CUDA (fall back to CPU if unavailable)
    Any other value will result in CPU being returned.
    """

    device_cfg = config.get("device", "cpu")

    if isinstance(device_cfg, str) and device_cfg.lower() in ("auto", "default"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    elif isinstance(device_cfg, str) and device_cfg.lower() in ("cuda", "gpu"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    else:
        # fallback to cpu for any other value
        device = torch.device("cpu")

    return device


