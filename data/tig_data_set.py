""" Custom data set for the Temporal Info Graph Model 

    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches

    @author: jhuthmacher
"""
import json
from os import listdir
from os.path import isfile, join, exists
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx

import networkx as nx 

from data import KINECT_ADJACENCY


def files_exist(files):
    return len(files) != 0 and all(exists(f) for f in files)

def file_exist(file):
    return exists(file)

class TIGDataset(InMemoryDataset):
    """ PyTorch data set to return batches of dynamic graphs.
    """

    def __init__(self, path = "./.data/kinetic/", val_dir=None, train_dir=None, test_dir=None, verbose=False, lim=-1):
        """ Initialization of the data set.

            self.processed_paths = [processed_file_dir + processed_file_names]

            Parameters:
                path: str
                    Path the folder with the data.
        """
        self.val_dir = val_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.path = path
        self.lim = lim

        self.verbose = verbose

        super().__init__(path)

        self.data, self.slices = torch.load(self.processed_paths[0])
    
    @property
    def raw_dir(self):
        return join(self.root, '')
    
    @property
    def raw_paths(self):
        r"""The filepaths to find in order to skip the download."""
        # files = to_list(self.raw_file_names)
        return ([join(self.raw_dir, f) for f in self.raw_file_names[0]] + 
            [join(self.raw_dir, f) for f in self.raw_file_names[1]] +
            [join(self.raw_dir, f) for f in self.raw_file_names[2]] +
            [join(self.raw_dir, f) for f in self.raw_file_names[3]])
            

    @property
    def raw_file_names(self):
        """ Property defining the files that will be loaded.
        """        
        val_files = []
        train_files = []
        test_files =[]
        all_files = []

        if self.val_dir is not None:
            path = self.path + self.val_dir
            val_files += [join(self.val_dir, f) for f in listdir(path) if isfile(join(path, f))]
        
        if self.train_dir is not None:
            path = self.path + self.train_dir
            test_files += [join(self.train_dir, f) for f in listdir(path) if isfile(join(path, f))]
        
        if self.test_dir is not None:
            path = self.path + self.test_dir
            test_files += [join(self.test_dir, f) for f in listdir(path) if isfile(join(path, f))]

        if self.val_dir is None and self.train_dir is None and self.test_dir is None:
            all_files += [f for f in listdir(self.path) if isfile(join(self.path, f))]
        
        return val_files, train_files, test_files, all_files

    @property
    def processed_file_names(self):
        """ Name of the output name after the data is processed.
        """
        file_names = []

        if self.val_dir is not None:
            file_names += ['tig_val.pt']
        if self.train_dir is not None:
            file_names += ['tig_train.pt']
        if self.test_dir is not None:
            file_names += ['tig_test.pt']

        if (self.val_dir is None and self.train_dir is None and self.test_dir is None):
            file_names += ['tig_data.pt']
    
        return ['tig_val.pt', 'tig_train.pt', 'tig_test.pt',  'tig_data.pt'] #file_names

    def download(self):
        """ Not used yet! Maybe later direct connection to DB here.
        """
        pass

    def process(self):
        """ Read as well as process local data and creat global data object.

            IMPORTANT: For the first try we only use the first person in a frame!
            Consider multiple persons need a more complex concept.

            IMPORTANT: We stop after 200 frames and ignore the remaining frames, because 
            the number of frames is not unified and it is not possible to create 
            appropriate batches, where the data have different sizes.

            First read the local data into memory. For this we use the attribute self.raw_paths.
            Next, the data has to be brought into the right format. I.e. extract the feature matrices per
            frame.
            Utilizes networkx to create the edge_index from global adjacency.



        """
        
        for l, file_list in enumerate(self.raw_file_names):
            data_list = []

            if file_exist(self.processed_paths[l]):
                continue

            for i, json_file in enumerate(tqdm(file_list, disable=(not self.verbose), desc=f"File list {l}: ")):
                json_file = self.path + json_file
                if self.lim > 0 and self.lim < i:
                    break
                try:
                    with open(json_file) as f:
                        data = json.load(f)
                        X = []
                        edges=[]
                        y = data["label_index"]

                        for k, frame in enumerate(data["data"]):
                            for person in frame["skeleton"]:
                                feature_matrix = []
                                for j in range(0, len(person["pose"])-1, 2):
                                    feature_matrix.append([person["pose"][j], person["pose"][j+1] * -1])

                                break # Only consider the first person for now!

                            X.append(feature_matrix)

                            # List of index tuples, i.e. [from, to]. Convert after with .t().contiguous()!
                            G = nx.Graph(KINECT_ADJACENCY)
                            edge_index = torch.tensor(list(G.edges)).t().contiguous()
                            edges.append(edge_index)

                            if i == 200:
                                break # unify the number of frames

                    # Each data point corresponds to a list of graphs.
                    data = SequenceData(edge_index=edges, x=torch.tensor(X), y=y, frames=len(data["data"]))
                    data_list.append(data)
                except Exception as e:
                    print(e, json_file)

            if len(data_list) > 0:
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[l])


class SequenceData(Data):
    """ Custom data object to handle sequences of graphs.
    """

    def __init__(self, edge_index: list = None, x: list = None, y: int = None, frames: int = None):
        """ Initialization of the sequence data object.

            Parameters:
                edge_index: list
                    List of tensors that represents the edge_index for a specific frame.
                x: list
                    List of feature matrices, where each amtrix corresponds to one specific frame.
                y: int
                    Index of the label of the corresponding action class
                frames: int
                    Number of frames.
        """
        super(SequenceData, self).__init__()
        self.edge_index = edge_index
        self.x = x
        self.y = y
        self.frames = frames
    
    def __inc__(self, key, value):
        """ Mandatory function to properly iterate over the batches.
        """
        if key == 'edge_index':
            # Under the assumption all graphs have the same number of
            # We have to add the frames as well, otherwise we would only jump to
            # the next graph in the sequence.
            return self.x[0].size(0) + self.frames
        else:
            return super(SequenceData, self).__inc__(key, value)