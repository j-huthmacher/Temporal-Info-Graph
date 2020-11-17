""" Custom data set for the Temporal Info Graph Model 

    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches

    @author: jhuthmacher
"""
import json

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.utils import from_networkx

import networkx as nx 

from data import KINECT_ADJACENCY


class TIGDataset(InMemoryDataset):
    """ PyTorch data set to return batches of dynamic graphs.
    """

    def __init__(self, path = "./.data/kinetic/"):
        """ Initialization of the data set.

            self.processed_paths = [processed_file_dir + processed_file_names]

            Parameters:
                path: str
                    Path the folder with the data.
        """
        super(TIGDataset, self).__init__(path)

        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        """ Property defining the files that will be loaded.
        """
        # EXAMPLE FILES.
        return ['__lt03EF4ao.json', '__NrybzYzUg.json', '__PYrzYbzKE.json']

    @property
    def processed_file_names(self):
        """ Name of the output name after the data is processed.
        """
        # EXAMPLE
        return ['data.pt']

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
        data_list = []

        for i, json_file in enumerate(self.raw_paths):
            with open(json_file) as f:
                data = json.load(f)

                X = []
                edges=[]
                y = data["label_index"]

                for i, frame in enumerate(data["data"]):
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

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


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