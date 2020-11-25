""" Utility functions for the data.

    @author: jhuthmacher
"""
from data.tig_data_set import TIGDataset


def get_max_frames(dataset: TIGDataset):
    """ Calculate the maximum number of frames for a data set.

        Parameter:
            dataset: TIGDataset or torch.Subset
                The data set for wich the maximum number of frames is calculated.
        Return:
            int: Maximum number of frames for the given dataset.
    """
    max_frames = -1

    for data in dataset:
        max_frames = max_frames if max_frames > data.frames else data.frames

    return max_frames