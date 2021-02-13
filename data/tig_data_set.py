""" Custom data set for the Temporal Info Graph Model 

    @author: jhuthmacher
"""
from zipfile import ZipFile
from pathlib import Path
import sys
import os
import json
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm
import pickle
import requests

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import log
from data import KINECT_ADJACENCY, NTU_ADJACENCY

##########################
# Custom DataSet Objects #
##########################


class TIGDataset(Dataset):
    """ TIG data set for data that is too large for the memory.
    """

    def __init__(self, name: str, path: str = "./dataset/",
                 s_files: [str] = None, verbose: bool = False, process_label=False,
                 lim: int = None, split_persons: bool = False):
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
        self.lim = lim
        self.split_persons = split_persons

        self.x = []
        self.y = []

        if name == "stgcn":
            self.load_stgcn_data()
            if split_persons:
                self.A = KINECT_ADJACENCY[:18, :18]
            else:
                self.A = KINECT_ADJACENCY

        elif name == "stgcn_local" or name == "stgcn_2080":
            self.load_stgcn_local()
            if split_persons:
                self.A = KINECT_ADJACENCY[:18, :18]
            else:
                self.A = KINECT_ADJACENCY
        elif name == "ntu_rgb_d_local":
            self.load_ntu_rgb_d_local()
            self.A = NTU_ADJACENCY
        else:
            if s_files is not None:
                files = []
                for folder in s_files:
                    files.extend([join(folder, f) for f in listdir(folder)])
                if process_label:
                    self.process_label(files)
                else:
                    self.process(files)
            else:
                # Download and extract data if not exists.
                self.load_data()
                self.A = KINECT_ADJACENCY

        try:
            path = "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetics-skeleton_labels.npz"
            self.label_info = np.load(path, allow_pickle=True, mmap_mode="r")["labels"]
        except:
            pass

    def load_ntu_rgb_d_local(self):
        """
        """
        path="D:/Temporal Info Graph/data/NTU-RGB-D/xsub/"
        ntu_train_data = np.load(path + 'train_data.npy', mmap_mode="r")
        
        with open(f"{path}train_label.pkl", "rb") as f:
                ntu_train_labels = pickle.load(f)

        if self.lim is not None:
            self.x = ntu_train_data[:self.lim].transpose(0,4,2,3,1).reshape(-1, 300, 25, 3)
            self.y = np.array(ntu_train_labels[1])[:self.lim]
        else:
            self.x = ntu_train_data.transpose(0,4,2,3,1).reshape(-1, 300, 25, 3)
            self.y = np.array(ntu_train_labels[1])

    def load_stgcn_local(self):
        """
        """
        path="D:/Temporal Info Graph/data/Kinetics/kinetics-skeleton/"
        if self.name == "stgcn_2080":
            log.info(f"Load data...")
            data = np.load(path + "stgcn_2080.npz", allow_pickle=True, mmap_mode="r")
            self.x = data["x"]
            self.y = data["y"]        
        else:
            log.info(f"Load data...")
            data = np.load(path + "stgcn.npz", allow_pickle=True, mmap_mode="r")
            self.x = data["x"]
            self.y = data["y"]  
            
            # train_data = np.load(path + 'train_data.npy', mmap_mode="r")[:, :2, :, : , :] # Throw away the confidence score
            # with open(f"{path}train_label.pkl", "rb") as f:
            #     train_labels = np.array(pickle.load(f)[1])

            # val_data = np.load(path + 'val_data.npy', mmap_mode="r")[:, :2, :, : , :] # Throw away the confidence score
            # with open(f"{path}val_label.pkl", "rb") as f:
            #     val_labels = np.array(pickle.load(f)[1])

            # if self.lim is not None:
            #     self.x = np.concatenate([train_data, val_data])[:self.lim]
            #     self.y = np.concatenate([train_labels, val_labels])[:self.lim]
            # else:
            #     self.x = np.concatenate([train_data, val_data])
            #     self.y = np.concatenate([train_labels, val_labels])

            # if self.split_persons:
            #     self.x = self.x.transpose(0, 4, 3, 2, 1).reshape(-1, 300, 18, 2)
            #     self.y = self.y.repeat(2)
            # else:
            #     self.x = self.x.reshape(-1, 2, 300, 36).transpose((0,2,3,1))

        with open(self.path + 'kinetics_class_dict.json', "rb") as f:
            self.classes = json.load(f)

    def load_stgcn_data(self):
        """
        """
        fname = "train_data.npy"
        fname1 = "val_data.npy"

        if not exists(self.path+fname) and not exists(self.path+fname1):
            if not exists(self.path + self.name + ".rar"):
                #### Download ####
                url = f'http://85.215.86.232/tig/data/stgcn_kinetics_skeleton.zip'
                r = requests.get(url, allow_redirects=True, stream=True)

                pbar = tqdm(r.iter_content(chunk_size=chunkSize), unit="MB", total=int(
                            r.headers['Content-Length']) // 10**6, desc=f'Download {self.name} ')
                chunkSize = 1024

                Path(self.path).mkdir(parents=True, exist_ok=True)
                with open(self.path + "stgcn_kinetics_skeleton.zip", 'wb') as f:
                    for chunk in pbar:
                        if chunk:  # filter out keep-alive new chunks
                            pbar.update(len(chunk) // 10**6)
                            f.write(chunk)
                log.info(
                    f"Data set donwloaded! ({self.path + 'stgcn_kinetics_skeleton.zip'})")
            else:
                log.info(
                    f"Data exist already! ({self.path + 'stgcn_kinetics_skeleton.zip'})")

            if not exists(self.path + fname) and not exists(self.path+fname1):
                #### Extract ####
                with ZipFile(file=self.path + 'stgcn_kinetics_skeleton.zip') as zip_file:
                    # Loop over each file
                    for member in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()), desc=f'Extract {self.name} '):
                        zip_file.extract(member, path=self.path)
            else:
                log.info(
                    f"Data exist extracted! ({self.path + fname} / {self.path + fname1})")

        if not exists(self.path + 'kinetics_class_dict.json'):
            #### Download ####
            url = f'http://85.215.86.232/tig/data/kinetics_class_dict.json'
            r = requests.get(url, allow_redirects=True, stream=True)

            pbar = tqdm(r.iter_content(chunk_size=chunkSize), unit="B", total=int(
                        r.headers['Content-Length']) // 10**6, desc=f'Download {self.name} ')
            chunkSize = 1024

            Path(self.path).mkdir(parents=True, exist_ok=True)
            with open(self.path + 'kinetics_class_dict.json', 'wb') as f:
                for chunk in pbar:
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk) // 10**6)
                        f.write(chunk)
            log.info(
                f"Class dictionary donwloaded! ({self.path + 'kinetics_class_dict.json'})")
        else:
            log.info(
                f"Class dictionary already exist! ({self.path + 'kinetics_class_dict.json'})")

        log.info("Load data in memory...")
        train_data = np.load(self.path + 'train_data.npy', mmap_mode="r")[:, :2, :, : , :] # Throw away the confidence score
        with open(f"{self.path}train_label.pkl", "rb") as f:
            train_labels = np.array(pickle.load(f)[1])

        val_data = np.load(self.path + 'val_data.npy', mmap_mode="r")[:, :2, :, : , :] # Throw away the confidence score
        with open(f"{self.path}val_label.pkl", "rb") as f:
            val_labels = np.array(pickle.load(f)[1])

        if self.lim is not None:
            self.x = np.concatenate([train_data, val_data])[:self.lim]
            self.y = np.concatenate([train_labels, val_labels])[:self.lim]
        else:
            self.x = np.concatenate([train_data, val_data])
            self.y = np.concatenate([train_labels, val_labels])

        if self.split_persons:
            self.x = self.x.transpose(0, 4, 3, 2, 1).reshape(-1, 300, 18, 2)
            self.y = self.y.repeat(2)
        else:
            self.x = self.x.reshape(-1, 2, 300, 36).transpose((0,2,3,1))

        with open(self.path + 'kinetics_class_dict.json', "rb") as f:
            self.classes = json.load(f)

        with open(self.path + 'kinetics_class_dict.json', "rb") as f:
            self.classes = json.load(f)

    def load_data(self):
        """ Load the data into memory.
            If the data doesn't exist the data is downloaded and extracted.
        """
        if not exists(self.path + self.file_name):
            if not exists(self.path + self.name + ".rar"):
                #### Download ####
                url = f'http://85.215.86.232/tig/data/{self.name}.zip'
                r = requests.get(url, allow_redirects=True, stream=True)

                pbar = tqdm(unit="B", total=int(
                    r.headers['Content-Length']) // 10**6, desc=f'Download {self.name} ')
                chunkSize = 1024

                Path(self.path).mkdir(parents=True, exist_ok=True)
                with open(self.path + self.name + ".zip", 'wb') as f:
                    for chunk in r.iter_content(chunk_size=chunkSize):
                        if chunk:  # filter out keep-alive new chunks
                            pbar.update(len(chunk) // 10**6)
                            f.write(chunk)
                log.info(
                    f"Data set donwloaded! ({self.path + self.name + '.zip'})")
            else:
                log.info(
                    f"Data exist already! ({self.path + self.name + '.zip'})")

            if not exists(self.path + self.name + '.npz'):
                #### Extract ####
                with ZipFile(file=self.path + self.name + '.zip') as zip_file:
                    # Loop over each file
                    for member in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()), desc=f'Extract {self.name} '):
                        zip_file.extract(member, path=self.path)
            else:
                log.info(
                    f"Data exist extracted! ({self.path + self.name + '.npz'})")

        if not exists(self.path + 'kinetics_class_dict.json'):
            #### Download ####
            url = f'http://85.215.86.232/tig/data/kinetics_class_dict.json'
            r = requests.get(url, allow_redirects=True, stream=True)

            pbar = tqdm(unit="B", total=int(
                r.headers['Content-Length']) // 10**6, desc=f'Download {self.name} ')
            chunkSize = 1024

            Path(self.path).mkdir(parents=True, exist_ok=True)
            with open(self.path + 'kinetics_class_dict.json', 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunkSize):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk) // 10**6)
                        f.write(chunk)
            log.info(
                f"Class dictionary donwloaded! ({self.path + 'kinetics_class_dict.json'})")
        else:
            log.info(
                f"Class dictionary already exist! ({self.path + 'kinetics_class_dict.json'})")

        log.info(f"Load data...")
        data = np.load(self.path + self.file_name, allow_pickle=True, mmap_mode="r")
        self.x = data["x"]
        self.y = data["y"]
        with open(self.path + 'kinetics_class_dict.json', "rb") as f:
            self.classes = json.load(f)

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
                sets.append(
                    list(zip(x[train_threshold:lim], y[train_threshold:lim])))
            else:
                sets.append(
                    list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

        if mode > 2:
            sets.append(
                list(zip(x[train_threshold:val_threshold], y[train_threshold:val_threshold])))

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
        sets.append(
            list(zip(self.x[:train_threshold], self.y[:train_threshold])))

        if mode > 1:
            if mode <= 2:
                sets.append(
                    list(zip(self.x[train_threshold:lim], self.y[train_threshold:lim])))
            else:
                sets.append(list(zip(
                    self.x[train_threshold:val_threshold], self.y[train_threshold:val_threshold])))

        if mode > 2:
            sets.append(
                list(zip(self.x[val_threshold:lim], self.y[val_threshold:lim])))

        return sets[0] if mode <= 1 else sets

    def stratify(self, num: int, train_ratio=0.8, val_ratio=0.1, mode=2, lim=None,
                 ret_idx=False):
        """
        """
        unique, counts = np.unique(self.y, return_counts=True)
        merged = np.asarray((unique, counts)).T
        sor = merged[merged[:, 1].argsort()][::-1]

        classes = sor[:num, 0]  # Top n classes
        class_idx,  = np.where(np.isin(self.y, classes))

        train_threshold = int(self.y[class_idx].shape[0] * train_ratio)
        val_threshold = int(self.y[class_idx].shape[0]
                            * (train_ratio + val_ratio))

        if lim is not None and lim < len(class_idx):
            # lim = lim if lim <= self.y[class_idx].shape[0] else self.y[class_idx].shape[0]
            train_threshold = int((lim) * train_ratio)
            val_threshold = int((lim) * (train_ratio + val_ratio))
        else:
            lim = len(self.y[class_idx])#.shape[0]

        sets = []
        sets.append(list(
            zip(self.x[class_idx][:train_threshold], self.y[class_idx][:train_threshold])))

        if mode > 1:
            if mode <= 2:
                sets.append(list(zip(self.x[class_idx][train_threshold:lim],
                                     self.y[class_idx][train_threshold:lim])))
            else:
                sets.append(list(zip(self.x[class_idx][train_threshold:val_threshold],
                                     self.y[class_idx][train_threshold:val_threshold])))

        if mode > 2:
            sets.append(list(
                zip(self.x[class_idx][val_threshold:lim], self.y[class_idx][val_threshold:lim])))

        if ret_idx:
            return (sets[0] if mode <= 1 else sets), class_idx
        else:
            return sets[0] if mode <= 1 else sets

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

    def process_label(self, file_paths: [str] = None):
        """
        """
        label_list = []

        if file_paths is None:
            file_paths = self.raw_paths

        length = len(file_paths)
        # start = length // 2

        # class

        # Iterate over all "raw" files
        for i, json_file in enumerate(tqdm(file_paths, disable=(not self.verbose),
                                           desc="Files processed:")):

            try:
                with open(json_file) as f:
                    data = json.load(f)

                label_list.append({
                    "id": json_file.split("/")[-1].split(".")[0],
                    "label": data["label"],
                    "label_index": data["label_index"]}
                )
            except:
                pass

        np.savez(f"{self.path}{self.name}_labels", labels=label_list)
        log.info(f"File stored at {self.path}{self.name}_labels.npz")

        # try:
        #     with open(json_file) as f:
        #         data = json.load(f)

        #         label_list.append(json_file)
        #         X = []
        #         data_list_y.append(data["label_index"])
        # else:

    def process(self, file_paths: [str] = None):
        """ Depricated for now!
        """

        data_list_x = []  # Will store the data for one output file.
        data_list_y = []
        file_num = 0  # Output file index.
        start = 0  # Not used yet

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
                    X = np.zeros((num_frames, num_nodes, num_features)).astype(
                        np.float32)

                    for f, frame in enumerate(data['data']):
                        frame["skeleton"].sort(
                            key=lambda x: np.mean(x["score"]), reverse=True)

                        for s, skeleton in enumerate(frame["skeleton"]):
                            if s == 2:
                                break  # Only consider the two skeletons with the highest average confidence
                            # TODO: Append along node axis! (36, 2)
                            idx_min = int(s * num_nodes / 2)
                            idx_max = int((s + 1) * num_nodes / 2)

                            X[f, idx_min:idx_max, 0] = skeleton['pose'][0::2]
                            X[f, idx_min:idx_max, 1] = skeleton['pose'][1::2]

                            #### Centralization ####
                            # TODO: Check if it is applied twice
                            X[f, idx_min:idx_max, :] -= 0.5

                            idx = ((np.asarray(skeleton['score']) - 0.5) == 0)
                            # Reset all coordinates where confidence score is 0
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
        np.savez(f"{self.path}{self.name}", x=data_list_x, y=data_list_y)
        log.info(f"File stored at {self.path}{self.name}.npz")


####################
# Helper Functions #
####################

def files_exist(files):
    return len(files) != 0 and all(exists(f) for f in files)


def file_exist(file):
    return exists(file)
