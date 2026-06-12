import logging
from pathlib import Path


def get_logger(config=None, name="hydroecolstm_lite"):
    """Return a configured logger for the package.

    If `config` contains `output_directory`, a file handler is added
    writing to `hydroecolstm_lite.log` in that directory. Subsequent
    calls return the same logger instance.
    """
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)

    fmt = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    # Add file handler when output directory is provided in config
    try:
        out_dir = None
        if config and isinstance(config, dict) and "output_directory" in config:
            out_dir = Path(config["output_directory"][0])
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)
            fh = logging.FileHandler(out_dir / "hydroecolstm_lite.log")
            fh.setLevel(logging.INFO)
            fh.setFormatter(fmt)
            logger.addHandler(fh)
    except Exception:
        # never fail logger creation because of file issues
        pass

    return logger
