"""
    @author: jhuthmacher
"""
import numpy as np
import torch

from torch import from_numpy
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


class GraphDataset(Dataset):
    """ PyTorch data set to return batches of dynamic graphs.
    """

    def __init__(self, x: np.array, y: np.array):
        """
        """

    def __getitem__(self, index):
        """ Returns an item 
        """
        raise NotImplementedError

    def __len__(self):
        """ Length of the data set.
        """
