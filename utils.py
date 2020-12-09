"""
    @author: jhuthmacher
"""

from config.config import log

def print_size(tensor):
    """
    """
    log.info("Size Unit", (np.prod(tensor.shape) * 3) // 10**6, "MB")
