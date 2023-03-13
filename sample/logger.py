"""setups logger"""
import logging

logging.root.setLevel(logging.NOTSET)


def init_logger(name, log_level):
    logger = logging.getLogger(name)
    log_format = "%(levelname)s - %(lineno)d - %(name)s - %(asctime)s - %(message)s"
    formatter = logging.Formatter(log_format)

    if not logger.handlers:
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('./logs/sampling.log')
        c_handler.setLevel(log_level)
        f_handler.setLevel(log_level)

        c_handler.setFormatter(formatter)
        f_handler.setFormatter(formatter)

        logger.addHandler(c_handler)
        logger.addHandler(f_handler)

    return logger


def get_global_logger(name, log_level=logging.INFO):
    return init_logger(name, log_level)
