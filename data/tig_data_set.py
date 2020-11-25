""" Custom data set for the Temporal Info Graph Model 

    https://pytorch-geometric.readthedocs.io/en/latest/notes/introduction.html#mini-batches

    @author: jhuthmacher
"""
import json
from os import listdir, makedirs
from os.path import isfile, join, exists
from tqdm import tqdm

import numpy as np
import torch
from torch_geometric.data import InMemoryDataset, Data, Dataset
from torch_geometric.utils import from_networkx

import networkx as nx 

from data import KINECT_ADJACENCY
from config.config import log


##########################
# Custom DataSet Objects #
##########################

class TIGDataset(Dataset):
    """ TIG data set for data that is too large for the memory.
    """
    def __init__(self, paths: [str], output: str = "./", num_chunks: int = 15,
                 verbose: bool = False):
        """ Initialization of the TIG DataSet

            Important: The number of output files may deviate from the number of chunks.
            The algorithm trys to equally distribute the items to the output files, i.e. 
            it could happen that the number of output files is less than the number of chunks.
            In this case all data was processed and it is stored in the files, i.e. 
            it doesn't mean some data is missing.

            Paramters:
                paths: [str]
                    List of paths to the data directory. Hint: The algorithm reads out each file
                    the directories.
                output: str
                    Path to the output directory where the output files are located.
                num_chunks: int
                    Number of chunks in which the data is splitted. Important: The algorithm
                    distributes the data equally, i.e. we may have less output files than 
                    the number of chunks.
                verbose: bool
                    Determines if the progress is visualized.
        """
        self.num_chunks = num_chunks
        self.paths = paths

        self.verbose = verbose

        self.init_paths()

        super().__init__(output)

    def init_paths(self):
        """ Calculate the path only one time!
        """
        paths = []
        files = []
        for path in self.paths:
            # files += [f for f in listdir(path) if isfile(join(path, f))] # isfile(join(...)) takes much more time
            paths += [join(path, f) for f in listdir(path)]
            files += [f for f in listdir(path)]

        self._raw_file_names = files
        self._raw_paths = paths
    
    def _download(self):
        """ This is the bottle neck!!! Don't check if each file exists
        """
        exist = False
        for f in self.raw_paths:
            exist = exist & isfile(f)

            if exist == False:
                # At least one file is missing
                break

        if exist:
            return
        else:
            # makedirs(self.processed_dir + "/raw")
            # self.download()
            # Has to be adapted for downloading
            pass

    @property
    def raw_paths(self):
        """ The filepaths to find in order to skip the download.
        """
        return self._raw_paths

    @property
    def raw_dir(self):
        """ The directories of the raw data.
        """
        return self.paths
    

    @property
    def raw_file_names(self):
        """ The raw file names.
        """        
        return self._raw_file_names

    @property
    def processed_file_names(self):
        """ Name of the output name after the data is processed.
        """
        # Should be fast enough
        # Makes sure that the data is equally distributed among the output files.
        num_chunks = int(len(self.raw_file_names) // np.ceil(len(self.raw_file_names)/self.num_chunks))
        file_names = [f"kinetic_skeleton_data_{i}.pt" for i in range(num_chunks)]
        return file_names
    
    @property
    def chunk_size(self):
        return np.ceil(len(self.raw_file_names)/self.num_chunks)

    def download(self):
        """ Not used yet! Maybe later direct connection to DB here.
        """
        pass

    def process(self):
        """ Read as well as process local data and creat output data object.

            IMPORTANT: For the first try we only use the first person in a frame!
            Consider multiple persons need a more complex concept.

            IMPORTANT: We stop after n frames and ignore the remaining frames, because 
            the number of frames is not unified and it is not possible to create 
            appropriate batches, where the data have different sizes.

            Utilizes networkx to create the edge_index from global adjacency.
        """    

        data_list = []  # Will store the data for one output file.
        file_num = 0  # Output file index.
        start = 0 # Not used yet

        # if isinstance(file_num, float):
        #     load_chunk = np.ceil(file_num) + 1
        #     data_list = torch.load(self.processed_paths[load_chunk])

        # Iterate over all "raw" files
        for i, json_file in enumerate(tqdm(self.raw_paths, disable=(not self.verbose),
                                           desc=f"Files processed ({file_num} stored): "), start):

            # After a output file reached the limit of data instances in it, write it to the disk and continue with the next file/chunk.
            if i != 0 and i % np.ceil(len(self.raw_file_names)/len(self.processed_paths)) == 0:                
                torch.save(data_list, self.processed_paths[file_num])
                file_num += 1
                data_list = []
                    
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

                        if k == 150:
                            self.dim = k
                            break # unify the number of frames
                            
                # Each data point corresponds to a list of graphs.
                data = SequenceData(edge_index=edges, x=torch.tensor(X), y=y, frames=len(data["data"]))
                data_list.append(data)
            except Exception as e:
                log.eror(e, json_file)

        # Store the last chunk
        torch.save(data_list, self.processed_paths[file_num])

    def len(self):
        return len(self.raw_file_names)

    def get(self, idx):
        chunk = self.chunk_size
        data = torch.load(self.processed_paths[int(np.floor(idx/chunk))])
        return data[int(idx%chunk)]


class TIGInMemoryDataset(InMemoryDataset):
    """ TIG data set for in memory data loading, i.e. for data that fits into the memory.

        Not recommended to use! Use TIGDataset instead.
    """

    def __init__(self, path: str = "./.data/kinetic/", val_dir: str = None, train_dir: str = None, test_dir: str = None,
                 verbose: bool = False, lim: int = -1):
        """ Initialization of the data set.

            self.processed_paths = [processed_file_dir + processed_file_names]

            Hint: If no one of val_dir, train_dir, and test_dir is provided the
            algorithm reads out each file from the folder given by the path.

            Parameters:
                path: str
                    Path the folder containing the data.
                val_dir: str
                    Directory of the validation data.
                train_dir: str
                    Directory of the train data.
                test_dir: str
                    Directory of the test data.
                verbose: bool
                    Determines if the progress is printed.
                lim: int
                    Determines a limit, for the case you only want process
                    some parts of the data, e.g. for testing.
        """
        self.val_dir = val_dir
        self.train_dir = train_dir
        self.test_dir = test_dir
        self.path = path
        self.lim = lim
        self.dim = None

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
        paths = []
        for path in self.raw_file_names:
            paths = [join(self.raw_dir, f) for f in path]

        return path
            

    @property
    def raw_file_names(self):
        """ Property defining the files that will be loaded.
        """        
        val_files = []
        train_files = []
        test_files =[]
        all_files = []

        files = []

        if self.val_dir is not None:
            path = self.path + self.val_dir
            files.append([join(self.val_dir, f) for f in listdir(path) if isfile(join(path, f))])
        
        if self.train_dir is not None:
            path = self.path + self.train_dir
            files.append([join(self.train_dir, f) for f in listdir(path) if isfile(join(path, f))])
        
        if self.test_dir is not None:
            path = self.path + self.test_dir
            files.append([join(self.test_dir, f) for f in listdir(path) if isfile(join(path, f))])

        if self.val_dir is None and self.train_dir is None and self.test_dir is None:
            files.append([f for f in listdir(self.path) if isfile(join(self.path, f))])
        
        return files#val_files, train_files, test_files, all_files

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
    
        return file_names #['tig_val.pt', 'tig_train.pt', 'tig_test.pt',  'tig_data.pt'] #file_names

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

                            if k == 150:
                                self.dim = k
                                break # unify the number of frames
                           

                    # Each data point corresponds to a list of graphs.
                    data = SequenceData(edge_index=edges, x=torch.tensor(X), y=y, frames=len(data["data"]))
                    data_list.append(data)
                except Exception as e:
                    print(e, json_file)

            if len(data_list) > 0:
                data, slices = self.collate(data_list)
                torch.save((data, slices), self.processed_paths[l])


######################
# Custom Data Object #
######################

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

####################
# Helper Functions #
####################

def files_exist(files):
    return len(files) != 0 and all(exists(f) for f in files)

def file_exist(file):
    return exists(file)
