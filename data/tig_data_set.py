""" Custom data set for the Temporal Info Graph Model 

    @author: jhuthmacher
"""
import requests
import patoolib
from pathlib import Path
import sys
import os
import json
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm

import numpy as np
import torch
from torch.utils.data import Dataset
import networkx as nx 

from data import KINECT_ADJACENCY
from config.config import log


##########################
# Custom DataSet Objects #
##########################
class TIGDataset(Dataset):
    """ TIG data set for data that is too large for the memory.
    """
    def __init__(self, name: str, path: str = "./dataset/"):
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

        self.name = name
        self.file_name = name + ".npz"
        self.path = f"{path}{name}/"

        self.small = small

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
                url = f'http://85.215.86.232/tig/data/{self.name}.rar'
                r = requests.get(url, allow_redirects=True, stream=True)

                pbar = tqdm(unit="B", total=int(r.headers['Content-Length']), desc=f'Download {self.name} ')
                chunkSize = 1024

                Path(self.path).mkdir(parents=True, exist_ok=True)            
                with open(self.path + self.name + ".rar", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunkSize): 
                        if chunk: # filter out keep-alive new chunks
                            pbar.update (len(chunk))
                            f.write(chunk)            
                log.info(f"Data set donwloaded! ({self.path + self.name + '.rar'})")
            else:
                log.info(f"Data exist already! ({self.path + self.name + '.rar'})")

            if not exists(self.path + self.name + '.npz'):
                #### Extract ####
                log.info(f"Start extraction! ({self.path + self.name + '.rar'})")
                patoolib.extract_archive(self.path + self.name + ".rar", outdir=self.path, verbosity=-1)
                log.info(f"Data set extracted! ({self.path + self.name + '.npz'})")
            else:
                log.info(f"Data exist extracted! ({self.path + self.name + '.npz'})")

        data = np.load(self.path +  self.file_name, allow_pickle=True)
        self.x = data["x"]
        self.y = data["y"]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])
        
    def process(self):
        """ Depricated for now!
        """    

        data_list_x = []  # Will store the data for one output file.
        data_list_y = []
        file_num = 0  # Output file index.
        start = 0 # Not used yet

        length = len(self.raw_paths)
        # start = length // 2

        # Iterate over all "raw" files
        for i, json_file in enumerate(tqdm(self.raw_paths, disable=(not self.verbose),
                                           desc=f"Files processed:")):
                    
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    X = []
                    data_list_y.append(data["label_index"])      
                    # y = data["label_index"]

                    # fill data_numpy
                    # Default dimension: (300, 18, 4). 4 because we consider 2 persons.
                    X = np.zeros((self.num_frames, self.num_nodes, 4))

                    for f, frame in enumerate(data['data']):
                        frame["skeleton"].sort(key=lambda x: np.mean(x["score"]), reverse=True)

                        for s, skeleton in enumerate(frame["skeleton"]):
                            if s == 2:
                                break  # Only consider the two skeletons with the highest average confidence
                            
                            X[f, :, (s * 2)] = skeleton['pose'][0::2]
                            X[f, :, (s * 2) + 1] = skeleton['pose'][1::2]

                # Each data point corresponds to a list of graphs.
                # data_list.append([X, y])       
                data_list_x.append(X)         

                if i == length // 2:
                    np.savez(self.processed_dir + "/kinetic_skeleton_1", x=data_list_x, y = data_list_y)
                    log.inf(f"File stored at {self.processed_dir + '/kinetic_skeleton_1.npz'}")
                    data_list_y = []
                    data_list_x = []
                    break # TODO

            except Exception as e:
                log.error(e, json_file)

        # Output dim (num_samples, 2), per output dim x on pos 0 and y on pos 1
        # X dim: (num_frames, joints/nodes, features), where features is 4
        np.savez(self.processed_dir + "/kinetic_skeleton_2", x=data_list_x, y = data_list_y)
        log.inf(f"File stored at {self.processed_dir + '/kinetic_skeleton_2.npz'}")


####################
# Helper Functions #
####################

def files_exist(files):
    return len(files) != 0 and all(exists(f) for f in files)

def file_exist(file):
    return exists(file)
