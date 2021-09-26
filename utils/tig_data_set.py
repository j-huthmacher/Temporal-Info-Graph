""" Custom data set for the Temporal Info Graph Model.
"""
# pylint: disable=not-callable
from zipfile import ZipFile
from pathlib import Path
import json
from os import listdir
from os.path import join, exists

from tqdm import tqdm
import requests
from sklearn import preprocessing

import numpy as np
import torch
from torch.utils.data import Dataset

from config.config import log
from utils import KINECT_ADJACENCY, NTU_ADJACENCY

class TIGDataset(Dataset):
    """ TIG data set for data that is too large for the memory.
    """

    def __init__(self, name: str = None, path: str = "./dataset/", s_files: [str] = None,
                 verbose: bool = False, process_label: bool = False,
                 lim: int = None, split_persons: bool = False):
        """ Initialization of the TIG DataSet

            Args:
                name: str
                    Name of the data set.
                    Options: 
                        * 'kinetic_skeleton': Full kinetics skeleton data set
                        * 'kinetic_skeleton_5000': Subset with (first) 5000 samples
                        * 'stgcn': Full kinetics skeleton data set from STGCN paper
                        * 'stgcn_50_classes': Kinetics skeleton data set from STGCN paper
                                              with 50 classes
                        * 'stgcn_20_classes': Kinetics skeleton data set from STGCN paper
                                              with 20 classes
                        * 'ntu_rgb_d_xsub': NTU-RGB D XSub dataset
                path: str
                    Path of the output folder, where the data will be located.
                s_files: [str]
                    Array paths to single data files that should be usd.
                verbose: bool
                    Determines if the data loading/processing process should be displayed.
                process_label: bool
                    Determines if the labels of the raw data are preprocessed and converted
                    in another format. Just works when s_files is provided.
                lim: int
                    Limit parameter, if only the first n values should be loaded, where n = lim.
                split_persons: bool
                    Determines if multiple persons in a video are splitted in separate data samples.
        """
        super().__init__()

        if name is not None:
            self.verbose = verbose
            self.path = f"{path}{name}/"
            self.name = name
            self.file_name = name + ".npz"
            self.lim = lim
            self.split_persons = split_persons

            self.x = []
            self.y = []

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
            if "stgcn" in name:
                # Use only the half adjacency, since the 2 persons in a sample are splitted.
                self.A = KINECT_ADJACENCY[:18, :18]
            elif "ntu" in name:
                self.A = NTU_ADJACENCY
            else:
                self.A = KINECT_ADJACENCY

    def to_tensor(self):
        """ Convert feature matrix and lables to PyTorch tensors.
        """
        self.x = torch.tensor(self.x)
        self.y = torch.tensor(self.y)

    def load_data(self):
        """ Load the data into memory.
            If the data doesn't exist the data is downloaded and extracted.
        """
        if not exists(self.path + self.file_name):
            if not exists(self.path + self.name + ".zip"):
                #### Download ####
                url = f'http://85.215.86.232/tig/data/{self.name}.zip'
                r = requests.get(url, allow_redirects=True, stream=True)

                chunkSize = 1024
                pbar = tqdm(r.iter_content(chunk_size=chunkSize), unit="B",
                            total=int(r.headers['Content-Length']),
                            desc=f'Download {self.name}',
                            unit_scale=True, unit_divisor=1024)

                Path(self.path).mkdir(parents=True, exist_ok=True)
                with open(self.path + self.name + ".zip", 'wb') as f:
                    for chunk in pbar:
                        if chunk:  # filter out keep-alive new chunks
                            pbar.update(len(chunk))
                            f.write(chunk)
                log.info("Data set donwloaded! (%s%s.zip)", self.path, self.name)
            else:
                log.info("Data exist already! (%s%s.zip)", self.path, self.name)

            if not exists(self.path + self.name + '.npz'):
                #### Extract ####
                with ZipFile(file=self.path + self.name + '.zip') as zip_file:
                    # Loop over each file
                    for member in tqdm(iterable=zip_file.namelist(), total=len(zip_file.namelist()),
                                       desc=f'Extract {self.name} '):
                        zip_file.extract(member, path=self.path)
            else:
                log.info("Data exist extracted! (%s%s.zip)", self.path, self.name)

        if not exists(self.path + 'kinetics_class_dict.json'):
            #### Download ####
            url = 'http://85.215.86.232/tig/data/kinetics_class_dict.json'
            r = requests.get(url, allow_redirects=True, stream=True)

            pbar = tqdm(unit="B", total=int(r.headers['Content-Length']) // 10**6,
                        desc=f'Download {self.name} ')
            chunkSize = 1024

            Path(self.path).mkdir(parents=True, exist_ok=True)
            with open(self.path + 'kinetics_class_dict.json', 'wb') as f:
                for chunk in r.iter_content(chunk_size=chunkSize):
                    if chunk:  # filter out keep-alive new chunks
                        pbar.update(len(chunk) // 10**6)
                        f.write(chunk)
            log.info("Class dictionary donwloaded! (%skinetics_class_dict.json)", self.path)
        else:
            log.info("Class dictionary already exist! (%skinetics_class_dict.json)", self.path)

        log.info("Load data...")
        data = np.load(self.path + self.file_name, allow_pickle=True, mmap_mode="r")
        if self.lim is not None:
            self.x = np.asarray(list(data["x"]))[:self.lim]
            self.y = np.asarray(list(data["y"]))[:self.lim]
        else:
            self.x = np.asarray(list(data["x"]))
            self.y = np.asarray(list(data["y"]))

        le = preprocessing.LabelEncoder()
        self.y = le.fit_transform(self.y)

        with open(self.path + 'kinetics_class_dict.json', "rb") as f:
            self.classes = json.load(f)

    def split_data(self, x: np.ndarray, y: np.ndarray, train_ratio: float = 0.8,
                   val_ratio: float = 0.1, mode: int = 2, lim: int = None):
        """ Split the given data based on the given paramters.
        
            Difference to the split() function in this class is that this function
            split some given data and split() splits the internal data of the instantiated
            object.
            (It is explicitly defined as class function and not as static function)

            Args:
                x: np.ndarray
                    Features.
                y: np.ndarray
                    Labels.
                train_ratio: float
                    Determines the portion of train data.
                val_ratio: float
                    Determines the portion of validation data.
                mode: int
                    Determines the mode, which data sets are returned. E.g. mode = 1 returns
                    only the train data set.
                lim: int
                    Limit paramter if only the first n samples should be returned, where n = lim.
            Return:
                [list]: It returns an array of list with the different data sets,
                i.e. [train_set, val_set].

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

    def split(self, train_ratio: float = 0.8, val_ratio: float = 0.1, mode: int = 2, lim: int = None):
        """ Spit data into train, validation and test set.

            The ratio for the test set is calculated using the train and
            validation set ratio. In general, the remaining data points are
            used for the test set.
            Example: 80% for training, 10% for validation, then we have
            90% for train/validation, i.e. the last 10% are used for the
            test set.

            Args:
                train_ratio: float
                    The percentage of the whole data set for the train set
                val_ratio: float
                    The percentage of the whole data set for the validation set
                mode: int
                    Control which sets are returned. Possible options: [1,2,3]
                    1 --> Only train set is returned
                    2 --> Train and validation set is returned
                    3 --> Train, validation and test set is returned
                lim: None or int
                    Limits the used data.
            Return:
                [np.ndarray]: [train_set, val_set, test_set] (depends on the mode)
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

    def stratify(self, num: int, num_samples: int = None, train_ratio: float = 0.8,
                 val_ratio: float = 0.1, mode: int = 2, lim: int = None, ret_idx: bool = False):
        """ This function makes a stratified split, i.e. it makes sure that the class labels
            are roughly  equally distirbuted in the returned sets.

            The ratio for the test set is calculated using the train and
            validation set ratio. In general, the remaining data points are
            used for the test set.
            Example: 80% for training, 10% for validation, then we have
            90% for train/validation, i.e. the last 10% are used for the
            test set.

            Args:
                num: int
                    Number of how many classes should be considered.
                num_samples: int
                    Number of samples per class. If None the maximum number of possible
                    classes is used.
                train_ratio: float
                    The percentage of the whole data set for the train set
                val_ratio: float
                    The percentage of the whole data set for the validation set
                mode: int
                    Control which sets are returned. Possible options: [1,2,3]
                    1 --> Only train set is returned
                    2 --> Train and validation set is returned
                    3 --> Train, validation and test set is returned
                lim: None or int
                    Limits the used data.
                ret_idx: bool
                    Flag that determines if the indices of the classes that are used
                    are returned as well.
            Return:
                [np.ndarray]: [train_set, val_set, test_set, class_indices]
                              (depends on the mode and ret_idx)
        """
        unique, counts = np.unique(self.y, return_counts=True)
        merged = np.asarray((unique, counts)).T
        sor = merged[merged[:, 1].argsort()][::-1]

        # Top n classes (regarding of how often the class is represented)
        classes = np.unique(sor[:num, 0])
        class_idx = np.array([], dtype=int)
        for cls in classes:
            # Get indices for top n classes
            class_idx = np.append(class_idx, np.where(self.y == cls)[0][:num_samples])

        if mode != 1:
            train_threshold = int(self.y[class_idx].shape[0] * train_ratio)
            val_threshold = int(self.y[class_idx].shape[0] * (train_ratio + val_ratio))
        else:
            train_threshold = None

        if lim is not None and lim < len(class_idx):
            # lim = lim if lim <= self.y[class_idx].shape[0] else self.y[class_idx].shape[0]
            train_threshold = int((lim) * train_ratio)
            val_threshold = int((lim) * (train_ratio + val_ratio))
        else:
            lim = len(self.y[class_idx])

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
            sets.append(list(zip(self.x[class_idx][val_threshold:lim],
                                 self.y[class_idx][val_threshold:lim])))

        if ret_idx:
            return (sets[0] if mode <= 1 else sets), class_idx
        else:
            return sets[0] if mode <= 1 else sets

    def __len__(self):
        """ Length of the data set.
        """
        return len(self.y)

    def __getitem__(self, idx: int):
        """ Returns the sample (x, y) at position idx.
        """
        return (self.x[idx], self.y[idx])

    def process_label(self, file_paths: [str] = None):
        """ This function processes the labels for Kintetics-Skeleton files.

            In other words, this function converts the labes from the Kintetics-Skeleton
            data set in a more suitable format. It stores the processed labels as numpy
            array at location that is set during the initialization of the object.

            Args:
                file_paths: [str]
                    List of file paths, where each corresponds to a sample from the
                    Kinetics-Skeleton data set.
        """
        label_list = []

        # Iterate over all "raw" files
        for json_file in tqdm(file_paths, disable=(not self.verbose), desc="Files processed:"):

            try:
                with open(json_file) as f:
                    data = json.load(f)

                label_list.append({
                    "id": json_file.split("/")[-1].split(".")[0],
                    "label": data["label"],
                    "label_index": data["label_index"]}
                )
            # pylint: disable=broad-except
            except Exception:
                pass

        np.savez(f"{self.path}{self.name}_labels", labels=label_list)
        log.info("File stored at %s%s_labels.npz", self.path, self.name)


    def process(self, file_paths: [str] = None):
        """ DEPRICATED. This function preprocesses the raw Kinetics-Skeleton files.

            This function should ne be used, since it originated from an early stage
            of the project and were not used for the final experiments. It is more or
            less an experimental function that is kept for the case that there is some
            future development in the direction of data preprocessing for this project.

            Args:
                file_paths: [str]
                    List of file paths of the single Kinetics-Skeleton files.
        """

        data_list_x = []  # Will store the data for one output file.
        data_list_y = []

        # Iterate over all "raw" files
        for _, json_file in enumerate(tqdm(file_paths, disable=(not self.verbose),
                                           desc="Files processed:")):
            try:
                with open(json_file) as f:
                    data = json.load(f)
                    X = []
                    data_list_y.append(data["label_index"])

                    # y = data["label_index"]

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
                                # Only consider the two skeletons with highest average confidence
                                break
                            # TODO: Append along node axis! (36, 2)
                            idx_min = int(s * num_nodes / 2)
                            idx_max = int((s + 1) * num_nodes / 2)

                            X[f, idx_min:idx_max, 0] = skeleton['pose'][0::2]
                            X[f, idx_min:idx_max, 1] = skeleton['pose'][1::2]

                            #### Centralization ####
                            X[f, idx_min:idx_max, :] -= 0.5

                            idx = ((np.asarray(skeleton['score']) - 0.5) == 0)
                            # Reset all coordinates where confidence score is 0
                            X[f, idx_min:idx_max, 0] = np.where(idx, 0, X[f, idx_min:idx_max, 0])
                            X[f, idx_min:idx_max, 1] = np.where(idx, 0, X[f, idx_min:idx_max, 1])

                # Each data point corresponds to a list of graphs.
                # data_list.append([X, y])
                data_list_x.append(X.astype(np.float32))

                # if i == length // 2:
                #     np.savez(self.processed_dir + "/kinetic_skeleton_1",
                #              x=data_list_x, y = data_list_y)
                #     log.inf(f"File stored at {self.processed_dir + '/kinetic_skeleton_1.npz'}")
                #     data_list_y = []
                #     data_list_x = []
                #     break

            # pylint: disable=broad-except
            except Exception as e:
                log.error(e, json_file)

        # Output dim (num_samples, 2), per output dim x on pos 0 and y on pos 1
        # X dim: (num_frames, joints/nodes, features), where features is 4
        data_list_y = np.asarray(data_list_y).astype(np.float32)
        np.savez(f"{self.path}{self.name}", x=data_list_x, y=data_list_y)
        log.info("File stored at %s%s.npz", self.path, self.name)


####################
# Helper Functions #
####################
def files_exist(files: [str]):
    """ Determines if all files in an array exist.

        Args:
            files: [str]
                List of file paths.
        Return:
            bool: True if all files exist, False if at least one file does not exist.
    """
    return len(files) != 0 and all(exists(f) for f in files)


def file_exist(file: str):
    """ Determines if a file exists.

        Args:
            file: str
                File path that should bec checked.
        Return:
            bool: True if file exists, False if it does not exist.
    """
    return exists(file)
