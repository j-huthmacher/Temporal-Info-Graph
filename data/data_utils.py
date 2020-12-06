""" Utility functions for the data.

    @author: jhuthmacher
"""
from typing import Any

import numpy as np

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

def convert_data(data: Any, to="float32"):
    """ Converts the data types in the data to the specified type.

        Important: If you provide a path in "data" the old files will be replaced.
    """

    if isinstance(data, str):
        ####  data is a path ####
        path = data
        data = np.load(data, allow_pickle=True)
        data = data["data"]

        np.savez(path.replace(".npz", ""),
                 x=np.asarray(np.asarray(data)[:, 0].tolist()).astype(to),
                 y=np.asarray(np.asarray(data)[:, 1].tolist()).astype(to))        
    else:
        return [np.asarray(np.asarray(data)[:, 0].tolist()).astype(to),
                np.asarray(np.asarray(data)[:, 1].tolist()).astype(to)]



########################
# Load Results from DB #
########################

def get_loss(db_url: str, experiment: dict):
    """
        Example: experiment  = {"run_id": 85, "name": {"$regex": "MLP\..*\.loss\.epoch\..*"} }
    """
    client = pymongo.MongoClient(db_url)
    collection = client.temporal_info_graph["metrics"]
    cursor = collection.find(experiment)

    trains_loss = []
    val_loss = []
    val_top1 = []
    val_top5 = []

    for doc in cursor:        
        if "loss" in doc["name"] and "train" in doc["name"]:
            trains_loss.append((doc["steps"], doc["values"]))
        elif "loss" in doc["name"] and "val" in doc["name"]:
            val_loss.append((doc["steps"], doc["values"]))
        elif "top1" in doc["name"] and "val" in doc["name"]:
            val_top1.append((doc["steps"], doc["values"]))
        elif "top5" in doc["name"] and "val" in doc["name"]:
            val_top5.append((doc["steps"], doc["values"]))
    
    return trains_loss, val_loss, val_top1, val_top5
