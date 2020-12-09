"""
    @author: jhuthmacher
"""
import numpy as np

from config.config import log

def print_size(tensor, byte_per_element):
    """
    """
    log.info("Size Unit", (np.prod(tensor.shape) * byte_per_element) // 10**6, "MB")
