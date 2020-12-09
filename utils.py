"""
    @author: jhuthmacher
"""
import numpy as np

from config.config import log

def print_size(tensor, byte_per_element, name=""):
    """
    """
    log.info(f"Size Unit ({name}) {(np.prod(tensor.shape) * byte_per_element) // 10**6} MB")
