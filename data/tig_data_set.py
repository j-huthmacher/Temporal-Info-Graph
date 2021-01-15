""" Custom data set for the Temporal Info Graph Model 

    @author: jhuthmacher
"""
import requests
from zipfile import ZipFile
from pathlib import Path
import sys
import os
import json
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset, TensorDataset
import networkx as nx 

from config.config import log

##########################
# Custom DataSet Objects #
##########################
class TIGDataset(Dataset):
    """ TIG data set for data that is too large for the memory.
    """
    def __init__(self, name: str, path: str = "./dataset/",
                 s_files: [str] = None, verbose: bool = False):
        """ Initialization of the TIG DataSet

            Paramters:
                name: str
                    Name of the data set.
                    Available: {'kinetic_skeleton', 'kinetic_skeleton_5000'}, where
                    'kinetic_skeleton_5000' is a subset with 5000 samples.
                path: str
                    Path of the output folder, where the data will be located.
        """
        super().__init__()

        self.verbose = verbose
        self.path = f"{path}{name}/"
        self.name = name
        self.file_name = name + ".npz"

        if s_files is not None:
            files = []
            for folder in s_files:
                files.extend([join(folder, f) for f in listdir(folder)])
            self.process(files)
        else:           
            self.x = []
            self.y = []

            # Download and extract data if not exists.
            self.load_data()

    def load_data(self):
        """ Load the data into memory.
            If the data doesn't exist the data is downloaded and extracted.
        """
        if not exists(self.path +  self.file_name):
            if not exists(self.path + self.name + ".rar"):
                #### Download #### 
                url = f'http://85.215.86.232/tig/data/{self.name}.zip'
                r = requests.get(url, allow_redirects=True, stream=True)

                pbar = tqdm(unit="B", total=int(r.headers['Content-Length'])//10**6, desc=f'Download {self.name} ')
                chunkSize = 1024

                Path(self.path).mkdir(parents=True, exist_ok=True)            
                with open(self.path + self.name + ".zip", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunkSize): 
                        if chunk: # filter out keep-alive new chunks
                            pbar.update (len(chunk)//10**6)
                            f.write(chunk)            
                log.info(f"Data set donwloaded! ({self.path + self.name + '.zip'})")
            else:
                log.info(f"Data exist already! ({self.path + self.name + '.zip'})")

            if not exists(self.path + self.name + '.npz'):
                #### Extract ####
                with ZipFile(file=self.path + self.name + '.zip') as zip_file:
                    # Loop over each file
                    for member in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()), desc=f'Extract {self.name} '):
                        zip_file.extract(member, path=self.path)
            else:
                log.info(f"Data exist extracted! ({self.path + self.name + '.npz'})")

        data = np.load(self.path +  self.file_name, allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]

    def split_data(self, x, y, train_ratio=0.8, val_ratio=0.1, mode=2, lim=None):
        """
        """
        train_threshold = int(y.shape[0] * train_ratio)
        val_threshold = int(y.shape[0] * (train_ratio + val_ratio))

        if lim is not None:
            train_threshold = int((lim) * train_ratio)
            val_threshold = int((lim) * (train_ratio + val_ratio))
        else:
            lim = y.shape[0]

        sets = []
        sets.append(list(zip(x[:train_threshold], y[:train_threshold])))

        if mode > 1:
            if mode <= 2:
                sets.append(list(zip(x[train_threshold:lim], y[train_threshold:lim])))
            else:
                sets.append(list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

        if mode > 2:
            sets.append(list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

        return sets[0] if mode <= 1 else sets

    def split(self, train_ratio=0.8, val_ratio=0.1, mode=2, lim=None):
        """ Spit data into train, validation and test set.

            The ratio for the test set is calculated using the train and 
            validation set ratio. In general, the remaining data points are
            used for the test set.
            Example: 80% for training, 10% for validation, then we have 
            90% for train/validation, i.e. the last 10% are used for the 
            test set.

            Paramters:
                train_ratio: float
                    The percentage of the whole data set for the train set
                val_ratio: float
                    The percentage of the whole data set for the validation set
                val_ratio: int
                    Control which sets are returned. Possible options: [1,2,3]
                    1 --> Only train set is returned
                    2 --> Train and validation set is returned
                    3 --> Train, validation and test set is returned
                lim: None or int
                    Limit the used data.
            Return:
                [np.array]: [train_set, val_set, test_set] (depends on the mode)
        """
        train_threshold = int(self.y.shape[0] * train_ratio)
        val_threshold = int(self.y.shape[0] * (train_ratio + val_ratio))

        if lim is not None:
            train_threshold = int((lim) * train_ratio)
            val_threshold = int((lim) * (train_ratio + val_ratio))
        else:
            lim = self.y.shape[0]

        sets = []
        sets.append(list(zip(self.x[:train_threshold], self.y[:train_threshold])))

        if mode > 1:
            if mode <= 2:
                sets.append(list(zip(self.x[train_threshold:lim], self.y[train_threshold:lim])))
            else:
                sets.append(list(zip(self.x[train_threshold:val_threshold], self.y[train_threshold:val_threshold])))
        
        if mode > 2:
            sets.append(list(zip(self.x[val_threshold:lim], self.y[val_threshold:lim])))

        return sets[0] if mode <= 1 else sets

    def stratify(self, num: int, train_ratio=0.8, val_ratio=0.1, mode=2, lim=None):
        """
        """
        unique, counts = np.unique(self.y, return_counts=True)
        merged = np.asarray((unique, counts)).T
        sor = merged[merged[:,1].argsort()][::-1]

        classes = sor[:num, 0] # Top n classes
        class_idx = np.where(np.isin(self.y, classes))

        train_threshold = int(self.y[class_idx].shape[0] * train_ratio)
        val_threshold = int(self.y[class_idx].shape[0] * (train_ratio + val_ratio))

        if lim is not None:
            lim = lim if lim <= self.y[class_idx].shape[0] else self.y[class_idx].shape[0]
            train_threshold = int((lim) * train_ratio)
            val_threshold = int((lim) * (train_ratio + val_ratio))
        else:
            lim = self.y[class_idx].shape[0]

        sets = []
        sets.append(list(zip(self.x[class_idx][:train_threshold], self.y[class_idx][:train_threshold])))

        if mode > 1 :
            if mode <= 2:
                sets.append(list(zip(self.x[class_idx][train_threshold:lim], 
                                    self.y[class_idx][train_threshold:lim])))
            else:
                sets.append(list(zip(self.x[class_idx][train_threshold:val_threshold], 
                                    self.y[class_idx][train_threshold:val_threshold])))

        if mode > 2:
            sets.append(list(zip(self.x[class_idx][val_threshold:lim], self.y[class_idx][val_threshold:lim])))

        return sets[0] if mode <= 1 else sets

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def process(self, file_paths: [str] = None):
        """ Depricated for now!
        """    

        data_list_x = []  # Will store the data for one output file.
        data_list_y = []
        file_num = 0  # Output file index.
        start = 0 # Not used yet

        if file_paths is None:
            file_paths = self.raw_paths

        length = len(file_paths)
        # start = length // 2

        # Iterate over all "raw" files
        for i, json_file in enumerate(tqdm(file_paths, disable=(not self.verbose),
                                           desc="Files processed:")):
                    
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    X = []
                    data_list_y.append(data["label_index"])

                    # y = data["label_index"]

                    # fill data_numpy
                    # Default dimension: (300, 36, 2). 4 because we consider 2 persons.
                    num_frames = 300
                    num_features = 2
                    num_nodes = 36
                    X = np.zeros((num_frames, num_nodes, num_features)).astype(np.float32)

                    for f, frame in enumerate(data['data']):
                        frame["skeleton"].sort(key=lambda x: np.mean(x["score"]), reverse=True)

                        for s, skeleton in enumerate(frame["skeleton"]):
                            if s == 2:
                                break  # Only consider the two skeletons with the highest average confidence
                            # TODO: Append along node axis! (36, 2)
                            idx_min = int(s * num_nodes/2)
                            idx_max = int((s + 1) * num_nodes/2)

                            X[f, idx_min:idx_max, 0] = skeleton['pose'][0::2]
                            X[f, idx_min:idx_max, 1] = skeleton['pose'][1::2]

                            #### Centralization ####
                            X[f, :, :] -= 0.5

                            idx = ((np.asarray(skeleton['score']) - 0.5) == 0)
                            X[f, idx_min:idx_max, 0] = np.where(idx, 0, X[f, idx_min:idx_max, 0])
                            X[f, idx_min:idx_max, 1] = np.where(idx, 0, X[f, idx_min:idx_max, 1])
                            
                            

                # Each data point corresponds to a list of graphs.
                # data_list.append([X, y])
                data_list_x.append(X.astype(np.float32))

                # if i == length // 2:
                #     np.savez(self.processed_dir + "/kinetic_skeleton_1", x=data_list_x, y = data_list_y)
                #     log.inf(f"File stored at {self.processed_dir + '/kinetic_skeleton_1.npz'}")
                #     data_list_y = []
                #     data_list_x = []
                #     break # TODO

            except Exception as e:
                log.error(e, json_file)

        # Output dim (num_samples, 2), per output dim x on pos 0 and y on pos 1
        # X dim: (num_frames, joints/nodes, features), where features is 4
        data_list_y = np.asarray(data_list_y).astype(np.float32)
        np.savez(f"{self.path}{self.name}", x=data_list_x, y = data_list_y)
        log.info(f"File stored at {self.path}{self.name}.npz")


####################
# Helper Functions #
####################

def files_exist(files):
    return len(files) != 0 and all(exists(f) for f in files)

def file_exist(file):
    return exists(file)
