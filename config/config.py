""" This file contains gobal configurations.

    @author: jhuthmacher
"""

from datetime import datetime
import logging
import os

from pathlib import Path


##################
# Shared Configs #
##################
local_log = True

##################

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

def create_logger(fpath, name = 'TIG_Logger', suffix = ""):
    """
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

        # log.propagate = False

    return log_
    # Custom logger configuration.
    # log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
    #                 datefmt='%d.%m.%Y %H:%M:%S',
    #                 level=log.INFO,
    #                 handlers=[
    #                     log.FileHandler(f'logs/DH_{datetime.now().date()}.log'),
    #                     log.StreamHandler()
    #                 ])
