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

Path("logs/").mkdir(parents=True, exist_ok=True)

if not os.path.exists(f'logs/TIG_{datetime.now().date()}.log'):
    with open(f'logs/TIG_{datetime.now().date()}.log', 'w+'):
        pass

log = logging.getLogger('TIG_Logger')

if not log.handlers:
    log.setLevel(logging.INFO)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fh = logging.FileHandler(f'logs/TIG_{datetime.now().date()}.log')
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

    log.propagate = False

# Custom logger configuration.
# log.basicConfig(format='%(asctime)s %(levelname)s: %(message)s',
#                 datefmt='%d.%m.%Y %H:%M:%S',
#                 level=log.INFO,
#                 handlers=[
#                     log.FileHandler(f'logs/DH_{datetime.now().date()}.log'),
#                     log.StreamHandler()
#                 ])
