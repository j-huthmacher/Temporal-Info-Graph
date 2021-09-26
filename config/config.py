""" This file is used to create a global logger.
"""
from datetime import datetime
import logging
import os

from pathlib import Path

path = "logs/"
Path(path).mkdir(parents=True, exist_ok=True)

if not os.path.exists(f'{path}TIG_{datetime.now().date()}.log'):
    with open(f'{path}TIG_{datetime.now().date()}.log', 'w+'):
        pass

log = logging.getLogger('TIG_Logger')

if not log.handlers:
    log.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(f'{path}TIG_{datetime.now().date()}.log')
    fh.setLevel(logging.INFO)

    # create formatter
    formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                  datefmt='%d.%m.%Y %H:%M:%S')

    # add formatter to ch
    ch.setFormatter(formatter)
    fh.setFormatter(formatter)

    # add ch to logger
    log.addHandler(ch)
    log.addHandler(fh)

def create_logger(fpath: str, name: str = 'TIG_Logger', suffix: str = ""):
    """ Function to create a logger and set unified configurations.

        Args:
            fpath: str
                File path where the logs are stored.
            name: str
                Name of the logger.
            suffix: str
                Suffix that is used for the log file.
        Return:
            logger object
    """
    Path(fpath).mkdir(parents=True, exist_ok=True)

    if not os.path.exists(f'{fpath}{suffix}TIG_{datetime.now().date()}.log'):
        with open(f'{fpath}{suffix}TIG_{datetime.now().date()}.log', 'w+'):
            pass

    log_ = logging.getLogger(name)

    if not log_.handlers:
        log_.setLevel(logging.INFO)

        # create console handler and set level to debug
        ch_ = logging.StreamHandler()
        ch_.setLevel(logging.INFO)

        fh_ = logging.FileHandler(f'{fpath}{suffix}TIG_{datetime.now().date()}.log')
        ch_.setLevel(logging.INFO)

        # create formatter
        formatter_ = logging.Formatter('%(asctime)s %(levelname)s: %(message)s',
                                       datefmt='%d.%m.%Y %H:%M:%S')

        # add formatter to ch
        ch_.setFormatter(formatter_)
        fh_.setFormatter(formatter_)

        # add ch to logger
        log_.addHandler(ch_)
        log_.addHandler(fh_)

    return log_
