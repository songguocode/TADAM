import os
import logging
from datetime import date, datetime

level_dict = {"DEBUG": 10, "INFO": 20, "WARNING": 30, "ERROR": 40, "CRITICAL": 50}


def log_or_print(logger, msg, level="info"):
    level = level.upper()
    assert level in level_dict.keys()
    if logger is not None:
        logger.log(level_dict[level], msg)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"{timestamp} [{level}]:\t{msg}")


def get_logger(
    name="global",
    save_file=False,
    overwrite_file=False,
    log_dir=None,
    log_name=None,
    console_verbose=False,
):
    """
        Setup and return a logger
    """
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s]:\t%(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    handlers = []
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG if console_verbose else logging.INFO)
    handlers.append(console_handler)
    # File handler
    if save_file:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        file_handler = logging.FileHandler(os.path.join(log_dir, f"{log_name}_log.txt"),
            mode="w" if overwrite_file else "a")
        file_handler.setFormatter(formatter)
        file_handler.setLevel(logging.DEBUG)
        handlers.append(file_handler)
    # Removes any handler in RootLogger, due to a defect in torch DDP
    # See https://discuss.pytorch.org/t/distributed-1-8-0-logging-twice-in-a-single-process-same-code-works-properly-in-1-7-0/114103/6
    logging.getLogger().handlers = []
    # Setup logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    for h in handlers:
        logger.addHandler(h)

    return logger
