import logging
from pathlib import Path


def setup_logger(name, log_file=None, log_files=None, level=logging.INFO, console=True):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.handlers.clear()
    logger.propagate = False

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    if console:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    all_log_files = []
    if log_file is not None:
        all_log_files.append(log_file)
    if log_files is not None:
        all_log_files.extend(log_files)

    for log_file in all_log_files:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
