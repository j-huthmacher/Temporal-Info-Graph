""" Main file.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn

from model.temporal_info_graph import TemporalInfoGraph
from model.mlp import MLP
from model.tracker import Tracker
from model.solver import Solver
from experiments import exp_overfit, exp_test,exp_test_trained_enc, exp_tig_overfit, exp_colab, experiment


from data import KINECT_ADJACENCY
from data.tig_data_set import TIGDataset

from config.config import log


##############
# Set up CLI #
##############
parser = argparse.ArgumentParser(prog='tig', description='Temporal Info Graph')

parser.add_argument('--train', dest='train', action='store_true',
                    help='Flag to select trainings mode.')

parser.add_argument('--downstream', dest='downstream', action='store_true',
                    help='Flag to determine if the downstream training should be executed.')

parser.add_argument('--disable_local_store', dest='disable_local_store', action='store_true',
                    help='Flag to determine if the models should be locally stored. Default: Models are stored locally.')

parser.add_argument('--prep_data', dest='prep_data', action="store_true",
                    help='Prepare data.')

# parser.add_argument('--type', dest='type', action="str",
#                     help='Prepare data.')

args = parser.parse_args()


#############
# Execution #
#############
if args.train:    
    db_url = open(".mongoURL", "r").readline()
    torch.cuda.empty_cache()
    name = "TIG_Test_ConfigurationProc"
    config = {
        "data": {
            "name": "kinetic_skeleton_5000",
            "path": "./content/"
        },
        "data_split": {
            "lim": 500
        },
        # "stratify": {
        #     "num": 5
        # },
        "loader": {
            "batch_size": 32
        },
        "emb_tracking": False,
        "encoder": {
            "c_in": 2,
            "c_out": 64,
            "spec_out": 128,
            "out": 2,
            "dim_in": (36, 300),
            "tempKernel": 32
        },
        "encoder_training": {
            "verbose": True,
            "n_epochs": 5
        },
        "classifier": {
            "in_dim": 2,
            "hidden_layers": [128, 512, 1024, 256]
        },
        "classifier_training": {
            "verbose": True,
            "n_epochs": 5
        },
    }

    # Training is executed from here
    tracker = Tracker(name, db_url, interactive=True)
    # tracker.track(exp_overfit)
    tracker.track(experiment, config)


elif args.prep_data:
    # Prepare data 
    data_paths = [
        "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetics_train/",
        "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetics_val/"
    ]

    output = "C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/"

    log.info("Process data")

    _ = TIGDataset(paths = data_paths, output=output, verbose=True)

    log.info("Data set processed")

