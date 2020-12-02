""" File containing different experiement configurations.
    @author: jhuthmacher
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import numpy as np
from sklearn.model_selection import train_test_split

from model.temporal_info_graph import TemporalInfoGraph
from model.mlp import MLP
from model.tracker import Tracker
from model.solver import Solver

from config.config import log


def exp_test(tracker):
    def run(_run):
        log.info("Start experiment.")
        tracker.run = _run
        tracker.id = _run._id

        data = np.load("C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetic_skeleton_small.npz", allow_pickle=True)
        data = data["data"][:100]

        # 60%, 20%, 20%
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

        train_loader = DataLoader(train, batch_size=1024, collate_fn=coll)
        val_loader = DataLoader(val, batch_size=1024, collate_fn=coll)
        test_loader = DataLoader(test, batch_size=1024, collate_fn=coll)

        tig = TemporalInfoGraph(c_in=4, c_out=6, spec_out=7, out=64, dim_in=(18, 300), tempKernel=32)
        solver = Solver(tig, [train_loader, val_loader])  

        tracker.track_traning(solver.train)({"verbose": True })

        #### DOWNSTREAM ####
        tracker.tag = "MLP"
        # Use the trained model!
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(64, num_classes, tracker.solver.model)

        solver = Solver(classifier, [train_loader, val_loader, test_loader], loss_fn = nn.NLLLoss())

        tracker.track_traning(solver.train)({"verbose": True })
        tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


def exp_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id

        data = np.load("C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/processed/kinetic_skeleton_1.npz", allow_pickle=True)
        data = data["data"][:5000]

        # 60%, 20%, 20%
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

        train_loader = DataLoader(train, batch_size=1024, collate_fn=coll)
        val_loader = DataLoader(val, batch_size=1024, collate_fn=coll)
        test_loader = DataLoader(test, batch_size=1024, collate_fn=coll)

        tig = TemporalInfoGraph(c_in=4, c_out=6, spec_out=7, out=64, dim_in=(18, 300), tempKernel=32)
        tig = tig.cuda()
        solver = Solver(tig, [train_loader, val_loader])  

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 256 })

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(64, num_classes, tracker.solver.model).cuda()

        solver = Solver(classifier, [train_loader, val_loader, test_loader], loss_fn = nn.NLLLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 256 })
        tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


####################
# Helper functions #
####################

def coll(batch):
    x = np.asarray(np.asarray(batch)[:, 0].tolist())
    y = np.asarray(np.asarray(batch)[:, 1].tolist())
    return (x, y)