""" File containing different experiement configurations.
    @author: jhuthmacher
"""
import io
import json

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from config.config import log
from data import KINECT_ADJACENCY
from data.tig_data_set import TIGDataset
from model import MLP, Solver, TemporalInfoGraph, Tracker

default = {
    "data": {
    },
    "data_split": {
    },
    "stratify": False,
    "loader": {
    },
    "same_loader": False,
    "emb_tracking": False,
    "encoder": {
    },
    "encoder_training": {
    },
    "classifier": False,
    "classifier_training": {
    },
}

#### Experiment Function that contains the whole procedure of an experiment ####
def experiment(tracker: Tracker, config: dict):
    """ Default experiment function.

        Parameters:
            tracker: Tracker
                Tracker object that handles the tracking for the experiment
            config: dict
                Experiment configuration that well be propagated down to the run function.
    """
    config = {**default, **config}

    def run(_run=None):
        """ Function that is executed as an experiment by Sacred.
        """
        log.info("Start experiment.")
        if _run is not None:
            tracker.run = _run
            tracker.id = _run._id
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))

        data = TIGDataset(**config["data"])

        if not config["stratify"] or config["stratify"] == {}:
            # Default: 80% train, 10% val
            train, val = data.split(**config["data_split"])
        else:
            train, val = data.stratify(**config["stratify"])

        if "val_loader" in config["loader"]:
            train_loader = DataLoader(
                train, **config["loader"]["train_loader"])
            val_loader = DataLoader(val, **config["loader"]["val_loader"])
        else:
            # In this case we use the same data loader config for train and validation.
            train_loader = DataLoader(train, **config["loader"])
            val_loader = DataLoader(val, **config["loader"])

        if config["same_loader"]:
            loader = [train_loader, train_loader]
        else:
            loader = [train_loader, val_loader]

        tig = TemporalInfoGraph(**config["encoder"]).cuda()
        solver = Solver(tig, loader)

        #### Tracking ####
        tracker.track_traning(solver.train)(config["encoder_training"])

        if config["emb_tracking"]:
            emb_x = np.array([])
            emb_y = np.array([])
            for batch_x, batch_y in train_loader:
                pred, _ = solver.model(batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1),
                                       torch.tensor(KINECT_ADJACENCY))
                if emb_x.size:
                    emb_x = np.concatenate(
                        [emb_x, pred.detach().cpu().numpy()])
                else:
                    emb_x = pred.detach().cpu().numpy()

                if emb_y.size:
                    emb_y = np.concatenate([emb_y, batch_y.numpy()])
                else:
                    emb_y = batch_y.numpy()

            buffer = io.BytesIO()
            np.savez(buffer, x=emb_x, y=emb_y)
            tracker.add_artifact(buffer.getvalue(), name="embeddings.npz")

            #### Local Embedding Tracking ####
            np.savez(tracker.local_path+"embeddings", x=emb_x, y=emb_y)

        if not "classifier" in config or not isinstance(config["classifier"], dict):
            return

        #### DOWNSTREAM ####
        tracker.tag = "MLP."
        
        #### Set Up Downstream Model ####
        num_classes = (config["classifier"]["num_classes"]
                       if "num_classes" in config["classifier"]
                       else int(np.max(data.y) + 1))
        classifier = MLP(num_class=num_classes, encoder=None,
                         **config["classifier"]).cuda()
        encoder = solver.model

        solver = Solver(classifier, loader, loss_fn=nn.CrossEntropyLoss())
        tracker.track_traning(solver.train)(config["classifier_training"], encoder=encoder)
        # tracker.track_testing(solver.test)({"verbose": True })

        log.info("Experiment done.")

    return run


class Experiment():
    """ Experiment object to easy access output files from an experiment run.
    """
    def __init__(self, path: str):
        """ Initilization.

            Parameters:
                path: str
                    Path pointing to the experiment folder created by an 
                    experiment run.
        """
        self.path = path
        try:
            data = np.load(f"{path}embeddings.npz")
            emb_x = data["x"]
            emb_y = data["y"]

            self.emb_loader = DataLoader(list(zip(emb_x, emb_y)))

            #### Clean Storage ####
            del data
            del emb_x
            del emb_y
        except:
            # Not each experiement has embeddings saved
            pass

        try:
            self.classifier = torch.load(f"{path}TIG_MLP.pt").cuda()
            self.clf_train_loss = np.load(f"{path}TIG_MLP.train_losses.npy")
            self.clf_val_loss = np.load(f"{path}TIG_MLP.val_losses.npy")
            self.clf_train_metrics = np.load(
                f"{path}TIG_MLP.train.metrics.npy")
            self.clf_val_metrics = np.load(f"{path}TIG_MLP.train.metrics.npy")
        except:
            # Not each experiement has  a classifier
            pass

        try:
            self.tig = torch.load(f"{path}TIG_.pt").cuda()
            self.tig_train_loss = np.load(f"{path}TIG_train_losses.npy")
            self.tig_val_loss = np.load(f"{path}TIG_val_losses.npy")
        except:
            pass

        try:
            self.tig_train_metrics = np.load(f"{path}TIG_train.metrics.npy")
            self.tig_val_metrics = np.load(f"{path}TIG_train.metrics.npy")
        except:
            pass

        with open(f"{path}config.json") as file:
            self.config = json.load(file)
            self.config["exp_path"] = self.path

    @property
    def emb(self):
        try:
            return self.emb_loader.dataset
        except:
            return None

    @property
    def emb_x(self):
        return np.array(list(np.array(self.emb)[:, 0]))

    @property
    def emb_y(self):
        return np.array(list(np.array(self.emb)[:, 1]))
