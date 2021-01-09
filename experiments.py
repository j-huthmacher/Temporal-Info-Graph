"""
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
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt
import seaborn as sns

import sacred

# pylint: disable=import-error
from config.config import log
from data import KINECT_ADJACENCY
from data.tig_data_set import TIGDataset
from model import MLP, Solver, TemporalInfoGraph
from tracker import Tracker
from evaluation import svc_classify, mlp_classify, randomforest_classify

#### Default Experiment Configuration ####
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

#### Main Experiment Function ####
def experiment(tracker: Tracker, config: dict):
    """ Default experiment function to run an machine learning experiment.

        Parameters:
            tracker: Tracker
                Tracker object that handles the tracking for the experiment
            config: dict
                Experiment configuration that well be propagated down to the run function.
    """
    #### Merge Custom Config with Default ####
    config = {**default, **config}

    def run(_run: sacred.run.Run = None):
        """ Function that is executed as an experiment by Sacred.

            Paramters:
                _run: sacred.run.Run
                    Sacred run object.
        """
        log.info("Start experiment.")
        if _run is not None:
            tracker.run = _run
            tracker.id = _run._id  #pylint: disable=protected-access
        tracker.log_config(f"{tracker.tag}local_path", str(tracker.local_path))
        

        #### Data Set Up ####
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

        #### TIG Set Up ####
        tig = TemporalInfoGraph(**config["encoder"]).cuda()
        solver = Solver(tig, loader)

        #### Tracking ####
        tracker.track_traning(solver.train)(config["encoder_training"])

        if config["emb_tracking"]:
            # Track the final embeddings after the TIG encoder is trained.
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
            # If the experiment only trains the encoder
            return

        #### MLP Set Up ####
        tracker.tag = "MLP."
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
    """ Experiment class to easy access output files from an experiment run in an
        object oriented matter.
    """
    def __init__(self, path: str):
        """ Initilization.

            Parameters:
                path: str
                    Path pointing to the experiment folder created by an experiment run.
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
        except:  #pylint: disable=bare-except
            # Not each experiement has embeddings saved
            pass

        try:
            self.classifier = torch.load(f"{path}TIG_MLP.pt").cuda()
            self.clf_train_loss = np.load(f"{path}TIG_MLP.train_losses.npy")
            self.clf_val_loss = np.load(f"{path}TIG_MLP.val_losses.npy")
            self.clf_train_metrics = np.load(f"{path}TIG_MLP.train.metrics.npy")
            self.clf_val_metrics = np.load(f"{path}TIG_MLP.train.metrics.npy")
        except:  #pylint: disable=bare-except
            # Not each experiement has  a classifier
            pass

        try:
            self.tig = torch.load(f"{path}TIG_.pt").cuda()
            self.tig_train_loss = np.load(f"{path}TIG_train_losses.npy")
            self.tig_val_loss = np.load(f"{path}TIG_val_losses.npy")
        except:  #pylint: disable=bare-except
            pass

        try:
            self.tig_train_metrics = np.load(f"{path}TIG_train.metrics.npy")
            self.tig_val_metrics = np.load(f"{path}TIG_train.metrics.npy")
        except:  #pylint: disable=bare-except
            pass

        with open(f"{path}config.json") as file:
            self.config = json.load(file)
            self.config["exp_path"] = self.path

    @property
    def emb(self):
        """ Embeddings from the encoder (complete data set).
        """
        try:
            return self.emb_loader.dataset
        except:  #pylint: disable=bare-except
            return None

    @property
    def emb_x(self):
        """ Features of the emebddings.
        """
        return np.array(list(np.array(self.emb)[:, 0])) if self.emb is not None else None

    @property
    def emb_y(self):
        """ Labels of the emebddings.
        """
        return np.array(list(np.array(self.emb)[:, 1])) if self.emb is not None else None

    def evaluate_emb(self, plot: bool = False):
        """ Evaluate the quality of the embeddings by testing different classifier.

            Paramters:
                plot: bool
                    If true the embeddings with their predictions are plotted.
        """
        #### SVM Classifier ####
        pred, acc = svc_classify(self.emb_x, self.emb_y, True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"SVM - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        print("SVM Acc.", acc)

        #### MLP Classifier ####
        pred, acc = mlp_classify(self.emb_x, self.emb_y, True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"MLP (Sklearn) - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        print("MLP Acc.", acc)

        #### Random Forest Classifier ####
        pred, acc = randomforest_classify(self.emb_x, self.emb_y, True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"Random Forest - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        print("Random Forest Acc.", acc)
