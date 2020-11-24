""" Main file.
"""
import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torch_geometric.data import DataLoader

from model.temporal_info_graph import TemporalInfoGraph
from model.mlp import MLP
from model.tracker import Tracker
from model.solver import Solver

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

args = parser.parse_args()


#############
# Execution #
#############
if args.train:    
    #######################
    # Trainings procedure #
    #######################
    db_url = open(".mongoURL", "r").readline()

    num_classes = 399
    classifier = None

    dataset = TIGDataset(path = "notebooks/.data/kinetic/")
    train_loader = DataLoader(dataset, batch_size=2)

    tracker_downstream = None
            
    # Trainings experiment
    def exp_train(_run):
        tracker.run=_run
        tracker.id = _run._id
        tig = TemporalInfoGraph(c_in=2, c_out=6, spec_out=7, out=64, dim_in=(18, 151), tempKernel=32)
        solver = Solver(tig, [train_loader, train_loader])  

        tracker.track_traning(solver.train)({"verbose": True })

        if args.downstream:
            classifier = MLP(64, 399, tracker.solver.model)
            # Downstream experiment
            def exp_downstream(_run):
                tracker_downstream.run = _run
                tracker_downstream.id = _run._id
                solver_downstream = Solver(classifier, [train_loader, train_loader], loss_fn = nn.NLLLoss())

                tracker_downstream.track_traning(solver_downstream.train)({"verbose": True })

                if not args.disable_local_store:
                    Path("./output/").mkdir(parents=True, exist_ok=True)
                    torch.save(tracker_downstream.model, './output/TIG_Downstream.pt')
                    log.info(f"TIG+Downstream model successfully stored ate {'./output/TIG_Downstream.pt'}")

            log.info("Downstream training started")
            tracker_downstream = Tracker("TIG_Downstream_MLP", db_url, interactive=True)
            tracker_downstream.track(exp_downstream)

        if not args.disable_local_store:
            Path("./output/").mkdir(parents=True, exist_ok=True)
            torch.save(tracker.model, './output/TIG.pt')
            log.info(f"TIG model successfully stored ate {'./output/TIG.pt'}")

    log.info("Training started")
    # Training is executed from here
    tracker = Tracker("TIG_Training", db_url, interactive=True)
    tracker.track(exp_train)

