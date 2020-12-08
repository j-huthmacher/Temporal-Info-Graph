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

def exp_test_trained_enc(tracker):
    def run(_run):
        log.info("Start experiment.")
        
        tracker.run = _run
        tracker.id = _run._id
        # tracker.run.experiment_info['name'] = "MLP_Learning_Overfitting_Behavior"

        data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        x = data.x[:100]
        y = data.y[:100]

        tracker.log_config(f"{tracker.tag}label_distr", str(y))

        # TODO: Stratified sampling
        sampler = RandomOverSampler(random_state=0)
        idx, _ = sampler.fit_sample(np.arange(y.shape[0]).reshape(-1, 1), y)

        x = x[idx.squeeze()]
        y = y[idx.squeeze()]

        tracker.log_config(f"{tracker.tag}label_distr_balanced", str(y))
        tracker.log_config(f"{tracker.tag}raw_train_size", str(y.shape))


        # Replace class lables to lower values!
        # keys = np.unique(data[:, 1]) 
        # vals = np.arange(np.unique(data[:, 1]).shape[0])

        # for k,v in zip(keys, vals):
        #     data[data[:, 1] == k, 1] = v

        # 60%, 20%, 20%
        # train, val, test = np.split(data, [int(.6*len(data)), int(.8*len(data))])


        # SGD --> Scattering! https://stats.stackexchange.com/questions/303857/explanation-of-spikes-in-training-loss-vs-iterations-with-adam-optimizer
        train_loader = DataLoader([x, y], batch_size=y.shape[0], collate_fn=coll)
        # val_loader = DataLoader(val, batch_size=len(val), collate_fn=coll)
        # test_loader = DataLoader(test, batch_size=len([1]), collate_fn=coll)

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

        data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        x = data.x[:100]
        y = data.y[:100]

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


def exp_tig_overfit(tracker):
    def run(_run):
        log.info("Start experiment (Overfitting).")
        tracker.run = _run
        tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        x = data.x[:100]
        y = data.y[:100]

        # 60%, 20%, 20%
        # train, val, test = np.split(data, [int(.8*len(data)), int(.9*len(data))])

        train_loader = DataLoader([x,y], batch_size=y.shape[0], collate_fn=coll)
        # val_loader = DataLoader(val, batch_size=1, collate_fn=coll)
        # test_loader = DataLoader(test, batch_size=1, collate_fn=coll)

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

        data = TIGDataset("kinetic_skeleton_5000", path="./dataset/")
        x = data.x[:100]
        y = data.y[:100]

        # 60%, 20%, 20%
        # train, val, test = np.split(data, [int(.8*len(data)), int(.9*len(data))])

        train_loader = DataLoader([x,y], batch_size=y.shape[0], collate_fn=coll)
        # val_loader = DataLoader(val, batch_size=1, collate_fn=coll)
        # test_loader = DataLoader(test, batch_size=1, collate_fn=coll)

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