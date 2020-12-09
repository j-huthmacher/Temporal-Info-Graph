""" File containing different experiement configurations.
    @author: jhuthmacher
"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

import numpy as np
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import RandomOverSampler

from model import TemporalInfoGraph, MLP, Solver, Tracker
from data.tig_data_set import TIGDataset


from config.config import log

def exp_colab(tracker):
    def run(_run):
        log.info("Start experiment.")        
        tracker.run = _run
        tracker.id = _run._id

        tracker.save_nth = 25

        #### Data Set Up ####
        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split() # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)
        # test_loader = DataLoader(test, batch_size=len([1]))

        #### Encoder ####
        if tracker.checkpoint is not None:
            log.info("Checkpoint exists")
            solver = Solver((tracker.checkpoint, TemporalInfoGraph), [train_loader, train_loader])
        else:
            tracker.checkpoint_dict["model_params"] = dict(c_in=4, c_out=16, spec_out=16,
                                                           out=16, dim_in=(18, 300), tempKernel=32)

            tig = TemporalInfoGraph(**tracker.checkpoint_dict["model_params"]).cuda()
            tracker.log_config(f"{tracker.tag}num_paramters", str(tig.num_paramters))
            solver = Solver(tig, [train_loader, val_loader])
        
        tracker.track_traning(solver.train)({
            "verbose": True, 
            "n_epochs": 1024
            })
        log.info("Encoder training done!")

        #### Downstream ####
        log.info("Starting downstream training...")
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = 400 #int(np.max(y) + 1)
        classifier = MLP(16, num_classes, [256, 512, 1024], tracker.solver.model).cuda()

        tracker.log_config(f"{tracker.tag}MLP.model", str(classifier))
        tracker.log_config(f"{tracker.tag}MLP.layers", str([64, 256, 512, num_classes]))

        solver = Solver(classifier, [train_loader, val_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({
            "verbose": True,
            "n_epochs": 1024,
            })

        log.info("Experiment done.")
    return run


def exp_test_trained_enc(tracker):
    def run(_run):
        log.info("Start experiment.")
        
        tracker.run = _run
        tracker.id = _run._id
        # tracker.run.experiment_info['name'] = "MLP_Learning_Overfitting_Behavior"

        # data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        # x = data.x[:100]
        # y = data.y[:100]

        # tracker.log_config(f"{tracker.tag}label_distr", str(y))

        # # TODO: Stratified sampling
        # sampler = RandomOverSampler(random_state=0)
        # idx, _ = sampler.fit_sample(np.arange(y.shape[0]).reshape(-1, 1), y)

        # x = x[idx.squeeze()]
        # y = y[idx.squeeze()]

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=100) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)


        # Replace class lables to lower values!
        # keys = np.unique(data[:, 1]) 
        # vals = np.arange(np.unique(data[:, 1]).shape[0])

        # for k,v in zip(keys, vals):
        #     data[data[:, 1] == k, 1] = v


        #### DOWNSTREAM ####
        log.info("Starting downstream training...")

        
        # Use the trained model!
        encoder = torch.load("./output/04122020_1641/TIG_.pt")
        num_classes = int(np.max(y) + 1)
        classifier = MLP(64, num_classes, [256, 512, 1024, 1024, 1024, 1024, 1024, 1024], encoder).cuda()

        tracker.log_config(f"{tracker.tag}MLP.model", str(classifier))
        tracker.log_config(f"{tracker.tag}MLP.layers", str([64, 256, 512, 1024, 1024, 1024, 1024, 1024, 1024, num_classes]))

        solver = Solver(classifier, [train_loader, train_loader], loss_fn = nn.CrossEntropyLoss())

        train_cfg = {
            "verbose": True,
            "n_epochs": 5000,
            "learning_rate": 2,
            "weight_decay": None
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

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=100) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)

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


def exp_tig_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=100) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)

        if tracker.checkpoint is not None:
            log.info("Checkpoint exists")
            solver = Solver((tracker.checkpoint, TemporalInfoGraph), [train_loader, train_loader])
        else:
            tracker.checkpoint_dict["model_params"] = dict(c_in=4, c_out=64, spec_out=128,
                                                           out=256, dim_in=(18, 300), tempKernel=32)
            # tracker.checkpoint["optim_params"] = dict(c_in=4, c_out=64, spec_out=128, out=256, dim_in=(18, 300), tempKernel=32)

            tig = TemporalInfoGraph(**tracker.checkpoint_dict["model_params"]).cuda()
            solver = Solver(tig, [train_loader, train_loader])

        #### Tracking ####
        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })

        log.info("Experiment done.")

    return run


def exp_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id

        data = TIGDataset("kinetic_skeleton_5000", path="/content/")
        train, val = data.split(lim=100) # Default: 80% train, 10% val

        train_loader = DataLoader(train, batch_size=4)
        val_loader = DataLoader(val, batch_size=4)

        tig = TemporalInfoGraph(c_in=4, c_out=64, spec_out=128, out=256, dim_in=(18, 300), tempKernel=32).cuda()
        solver = Solver(tig, [train_loader, train_loader])  

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        # Use the trained model!
        num_classes = int(np.max(y) + 1)
        classifier = MLP(256, num_classes, [128, 512, 1024, 256], tracker.solver.model).cuda()

        solver = Solver(classifier, [train_loader, train_loader], loss_fn = nn.CrossEntropyLoss())

        tracker.track_traning(solver.train)({"verbose": True, "n_epochs": 512 })
        # tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


####################
# Helper functions #
####################

def coll(batch):
    # x = np.asarray(np.asarray(batch)[:, 0].tolist())
    # y = np.asarray(np.asarray(batch)[:, 1].tolist())

    x = np.asarray(batch[0])
    y = np.asarray(batch[1])

    return (x, y)