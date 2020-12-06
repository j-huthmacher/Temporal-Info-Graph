""" File containing different experiement configurations.
    @author: jhuthmacher
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split

from model.temporal_info_graph import TemporalInfoGraph
from model.mlp import MLP
from model.tracker import Tracker
from model.solver import Solver

from config.config import log

def exp_test_trained_enc(tracker):
    def run(_run):
        log.info("Start experiment.")
        
        tracker.run = _run
        tracker.id = _run._id
        # tracker.run.experiment_info['name'] = "MLP_Learning_Overfitting_Behavior"

        data = np.load("C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetic_skeleton_small.npz", allow_pickle=True)
        data = data["data"][:20]

        # 60%, 20%, 20%
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])


        # SGD --> Scattering! https://stats.stackexchange.com/questions/303857/explanation-of-spikes-in-training-loss-vs-iterations-with-adam-optimizer
        train_loader = DataLoader(train, batch_size=len(train), collate_fn=coll)
        val_loader = DataLoader(val, batch_size=len(val), collate_fn=coll)
        # test_loader = DataLoader(test, batch_size=len([1]), collate_fn=coll)


        #### DOWNSTREAM ####
        log.info("Starting downstream training...")
        
        # Use the trained model!
        encoder = torch.load("./output/04122020_1641/TIG_.pt")
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(64, num_classes, [256, 512, 256], encoder).cuda()

        for lr, weight_decay in zip([2e-5], [1e-3]):
            tracker.tag = f"MLP."
            if lr is not None:
                tracker.tag +=f"{'lr_{:.2E}'.format(lr)}."
            if weight_decay is not None:
                tracker.tag +=f"{'wd_{:.2E}'.format(weight_decay)}."

            # optimizer = optim.SGD(classifier.parameters(), lr=lr, momentum=0.9)

            solver = Solver(classifier, [train_loader, val_loader], loss_fn = nn.CrossEntropyLoss())

            train_cfg = {
                "verbose": True,
                "n_epochs": 512,
                "learning_rate": lr,
                "weight_decay": weight_decay
            }

            tracker.track_traning(solver.train)(train_cfg)

        # tracker.log_config(f"{tracker.tag}optimzer", "Trained encoder (500 samples)")

        log.info("Experiment done.")
    return run


def exp_test(tracker):
    def run(_run):
        log.info("Start experiment.")
        tracker.run = _run
        tracker.id = _run._id

        data = np.load("C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetic_skeleton_small.npz", allow_pickle=True)
        data = data["data"][:100]

        # 60%, 20%, 20%
        train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])

        train_loader = DataLoader(train, batch_size=10, collate_fn=coll)
        val_loader = DataLoader(val, batch_size=10, collate_fn=coll)
        test_loader = DataLoader(test, batch_size=10, collate_fn=coll)

        tig = TemporalInfoGraph(c_in=4, c_out=6, spec_out=7, out=64, dim_in=(18, 300), tempKernel=32).cuda()
        solver = Solver(tig, [train_loader, val_loader])  

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 500 })

        #### DOWNSTREAM ####
        log.info("Starting downstream training...")
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(64, num_classes, [128, 512, 256], tracker.solver.model).cuda()

        solver = Solver(classifier, [train_loader, val_loader, test_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 500 })
        # tracker.track_testing(solver.test)({"verbose": True, "n_epochs": 1 })

        log.info("Experiment done.")

    return run


def exp_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id

        data = np.load("C:/Users/email/Documents/Studium/LMU/5_Semester/Masterthesis/Datasets/Kinetics-skeleton/kinetics-skeleton/kinetic_skeleton_small.npz", allow_pickle=True)
        data = data["data"][:500]

        # 60%, 20%, 20%
        train, val, test = np.split(data, [int(.8*len(data)), int(.9*len(data))])

        train_loader = DataLoader(train, batch_size=1, collate_fn=coll)
        # val_loader = DataLoader(val, batch_size=1, collate_fn=coll)
        # test_loader = DataLoader(test, batch_size=1, collate_fn=coll)

        tig = TemporalInfoGraph(c_in=4, c_out=6, spec_out=7, out=256, dim_in=(18, 300), tempKernel=32).cuda()
        solver = Solver(tig, [train_loader, train_loader])  

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = np.max(data[:, 1]) + 1
        classifier = MLP(256, num_classes, [128, 512, 256], tracker.solver.model).cuda()

        solver = Solver(classifier, [train_loader, train_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })
        # tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


####################
# Helper functions #
####################

def coll(batch):
    x = np.asarray(np.asarray(batch)[:, 0].tolist())
    y = np.asarray(np.asarray(batch)[:, 1].tolist())
    return (x, y)