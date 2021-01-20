"""
    @author: jhuthmacher
"""
import io
import json
import os
import pprint

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from PIL import Image
import io


from paramiko import SSHClient, SSHConfig, AutoAddPolicy, SFTPClient

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

import sacred

# pylint: disable=import-error
from config.config import log, create_logger
from data import KINECT_ADJACENCY
from data.tig_data_set import TIGDataset
from model import MLP, Solver, TemporalInfoGraph
from tracker import Tracker
from evaluation import svc_classify, mlp_classify, randomforest_classify
from visualization.plots import plot_emb, plot_curve

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
        tig = TemporalInfoGraph(**config["encoder"], A=KINECT_ADJACENCY).cuda()
        solver = Solver(tig, loader)

        #### Tracking ####
        tracker.track_traning(solver.train)(config["encoder_training"])

        if config["emb_tracking"]:
            # Track the final embeddings after the TIG encoder is trained.
            emb_x = np.array([])
            emb_y = np.array([])
            with torch.no_grad():
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
    def __init__(self, path: str, ssh: str = None):
        """ Initilization.

            Parameters:
                path: str
                    Path pointing to the experiment folder created by an experiment run.
        """
        self.img={}
        self.path = path
        self.folder =  os.path.normpath(self.path).split(os.sep)[0]
        self.name = os.path.normpath(self.path).split(os.sep)[-1]
        if ssh is None:
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
            
            #### TIG Model/Losses/Metric ####
            try:
                self.classifier = torch.load(f"{path}TIG_MLP.pt").cuda()
                self.clf_train_loss = np.load(f"{path}TIG_MLP.train_losses.npy")
                self.clf_val_loss = np.load(f"{path}TIG_MLP.val_losses.npy")
                self.clf_train_metrics = np.load(f"{path}TIG_MLP.train.metrics.npy")
                self.clf_val_metrics = np.load(f"{path}TIG_MLP.train.metrics.npy")
            except:  #pylint: disable=bare-except
                # Not each experiement has  a classifier
                pass
            
            #### TIG Model/Losses ####
            try:
                self.tig_train_loss = np.load(f"{path}TIG_train_losses.npy")
                self.tig_val_loss = np.load(f"{path}TIG_val_losses.npy")
                self.tig = torch.load(f"{path}TIG_.pt").cuda()
            except:  #pylint: disable=bare-except
                pass
            

            #### TIG Train Metric ####
            try:
                self.tig_train_metrics = np.load(f"{path}TIG_train.metrics.npy")
                self.tig_val_metrics = np.load(f"{path}TIG_train.metrics.npy")
            except:  #pylint: disable=bare-except
                pass

            with open(f"{path}config.json") as file:
                self.config = json.load(file)
                self.config["exp_path"] = self.path
            
            #### Set Up Logging ####
            self.log = create_logger(fpath = self.config["exp_path"], name="EXP Logger", suffix="EXP_")
        else:
            slurm, jhost = connect_to_slurm()
            sftp = SFTPClient.from_transport(slurm.get_transport())

            try:
                f = sftp.open(f"{path}embeddings.npz")
                f.prefetch()
                f = f.read()
                data = np.load(io.BytesIO(f))
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

            #### TIG Model/Losses/Metric ####
            try:
                f = sftp.open(f"{path}TIG_MLP.pt")
                f.prefetch()
                f = f.read()
                self.classifier = torch.load(io.BytesIO(f)).cuda()
                f = sftp.open(f"{path}TIG_MLP.train_losses.npy")
                f.prefetch()
                f = f.read()
                self.clf_train_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_MLP.val_losses.npy")
                f.prefetch()
                f = f.read()
                self.clf_val_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_MLP.train.metrics.npy")
                f.prefetch()
                f = f.read()
                self.clf_train_metrics = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_MLP.val.metrics.npy")
                f.prefetch()
                f = f.read()
                self.clf_val_metrics = np.load(io.BytesIO(f))
            except:  #pylint: disable=bare-except
                # Not each experiement has  a classifier
                pass
            
            #### TIG Model/Losses ####
            try:
                f = sftp.open(f"{path}TIG_.pt")
                f.prefetch()
                f = f.read()
                self.tig = torch.load(io.BytesIO(f)).cuda()
                f = sftp.open(f"{path}TIG_train_losses.npy")
                f.prefetch()    
                f = f.read()
                self.tig_train_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_val_losses.npy")
                f.prefetch()
                f = f.read()
                self.tig_val_loss = np.load(io.BytesIO(f))
            except:  #pylint: disable=bare-except
                pass
                
            #### TIG Train Metric ####
            try:
                f = sftp.open(f"{path}TIG_train.metrics.npy")
                f.prefetch()
                f = f.read()
                self.tig_train_metrics = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_train.metrics.npy")
                f.prefetch()
                f = f.read()
                self.tig_val_metrics = np.load(io.BytesIO(f))
            except:  #pylint: disable=bare-except
                pass
                
            #### TIG Train Metric ####
            try:
                f = sftp.open(f"{path}TIG.loss.final.png")
                f.prefetch()
                f = f.read()
                self.img["TIG.loss.final.png"] =f

                f = sftp.open(f"{path}MLP.loss.metric.final.png")
                f.prefetch()
                f = f.read()
                self.img["MLP.loss.metric.final.png"] = f

                f = sftp.open(f"{path}MLP.loss.final.png")
                f.prefetch()
                f = f.read()
                self.img["MLP.loss.final.png"] = f
            except:  #pylint: disable=bare-except
                pass
            
            f = sftp.open(f"{path}config.json")
            f.prefetch()
            f = f.read()
            self.config = json.load(io.BytesIO(f))
            self.config["exp_path"] = self.path

            slurm.close()
            jhost.close()
            
    def show_img(self, name):
        """
        """
        image = Image.open(io.BytesIO(self.img[name]))
        return image

    @property
    def tig_input_shape(self, n_batches: int = 2, n_nodes: int = 36, n_times: int = 300):
        # (batch, features, nodes, time)
        return (n_batches,
                self.config["encoder"]["architecture"][0][0],
                n_nodes,
                n_times)

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

    def __repr__(self):
        """
        """

        representation = f"Experiment ({os.path.normpath(self.path).split(os.sep)[-1]})\n"
        representation += f"--- Duration: {self.config['duration'] if 'duration' in self.config else 'Not Finished'}\n"
        representation += f"--- Batchsize: {self.config['loader']['batch_size'] if 'loader' in self.config else 'Config Incomplete'}\n"
        representation += f"--- Batchsize: {self.config['encoder_training']['optimizer']['lr'] if 'encoder_training' in self.config else 'Config Incomplete'}\n"
        representation += f"--- Training set: {self.config['train_length'] if 'train_length' in self.config else 'Config Incomplete'}\n"
        representation += f"--- Validation set: {self.config['val_length'] if 'val_length' in self.config else 'Config Incomplete'}\n"
        representation += f"--- Last JSD Loss (Train): {self.tig_train_loss[-1] if hasattr(self, 'tig_train_loss') else 'Not Finished'}\n"
        representation += f"--- Last JSD Loss (Val): {self.tig_val_loss[-1] if hasattr(self, 'tig_val_loss') else 'Not Finished'}\n"
        representation += f"--- Embedding Shape: {self.emb_x.shape if self.emb_x is not None else 'Not Finished'}\n"

        return representation
    
    def tensorboard(self, replace=False):
        """
        """
        if not os.path.exists(f"./runs/{self.name}") and replace == False:
            writer = SummaryWriter(log_dir=f"./runs/{self.name}")
            writer.add_graph(self.tig.cpu(), torch.rand(self.tig_input_shape).cpu())
            for i, val in enumerate(self.tig_train_loss):
                writer.add_scalar('training loss', val, i)

            for i, val in enumerate(self.tig_val_loss):
                writer.add_scalar('val loss', val, i)
            
            writer.add_text("config", pprint.pformat(self.config, indent=1, width=80).replace("\n", "  \n").replace("     ", "&nbsp;"))

    #### Experiment Visualization ####
    def plot_dash(self):
        """
        """
        fig = plt.figure(figsize=(14, 7), constrained_layout=True)
        gs = fig.add_gridspec(4, 2)

        #### Embeddings ####
        ax = fig.add_subplot(gs[:, 0])
        plot_emb(self.emb_x, self.emb_y, ax = ax, title="Embeddings")

        #### TIG/MLP Loss/Metrics ####
        ax = fig.add_subplot(gs[0, 1])
        self.plot_curve(name="TIG.loss", ax = ax)
        ax = fig.add_subplot(gs[1, 1])
        self.plot_curve(name="TIG.metric", ax = ax)

        if hasattr(self, "clf_train_loss"):
            ax = fig.add_subplot(gs[2, 1])
            self.plot_curve(name="MLP.loss", ax = ax)
            ax = fig.add_subplot(gs[3, 1])
            self.plot_curve(name="MLP.metric", ax = ax)
        else:
            ax = fig.add_subplot(gs[2:, 1])
            i = io.BytesIO(self.img["MLP.loss.metric.final.png"])
            i = mpimg.imread(i, format='PNG')

            ax.imshow(i, interpolation='none', aspect='auto')
            ax.axis('off')

        fig.tight_layout()

        fig.suptitle(f"Experiment: {os.path.normpath(self.path).split(os.sep)[-1]}")

        return fig
    
    def plot_curve(self, name="TIG.loss", ax = None):
        """
        """
        try:
            if name == "TIG.loss":
                model_name = "TIG: "
                title = "Loss (JSD)"
                args = {
                    "data": {
                        "TIG Train Loss (JSD MI)": self.tig_train_loss,
                        "TIG Val Loss (JSD MI)": self.tig_val_loss
                        }
                    }
            elif name == "TIG.metric":
                model_name = "TIG: "
                title = "Metric"
                args = {
                    "data": {
                        "TIG Train Loss (JSD MI)": self.tig_train_metrics,
                        "TIG Val Loss (JSD MI)": self.tig_val_metrics
                        }
                    }
            elif name == "MLP.loss":
                model_name = "MLP: "
                title = "Loss (Cross Entropy)"

                args = {
                    "data": {
                        "MLP Train Loss (JSD MI)": self.clf_train_loss,
                        "MLP Val Loss (JSD MI)": self.clf_val_loss
                        }
                    }
            elif name == "MLP.metric":
                model_name = "MLP: "
                title = "Accuracy (Top-1, Top-5)"

                args = {
                    "data": {
                        "MLP Top-1 Acc.": np.array(self.clf_train_metrics)[:, 0],
                        "MLP Top-5 Acc.": np.array(self.clf_val_metrics)[:, 1],
                        }
                    }
        except:
            if ax is not None:
                ax.axis('off')
            return None

        return plot_curve(**args, title=title, model_name=model_name, ax=ax)

    def plot_emb(self):
        """
        """
        return plot_emb(self.emb_x, self.emb_y)

    ### Experiment Evaluation ####
    def evaluate_emb(self, plot: bool = False):
        """ Evaluate the quality of the embeddings by testing different classifier.

            Paramters:
                plot: bool
                    If true the embeddings with their predictions are plotted.
        """
        if hasattr(self, "clf_val_metrics"):
            self.log.info(f"(Pipeline) MLP Accuracy: {np.max(self.clf_val_metrics)}")
    
        #### SVM Classifier ####
        if hasattr(self, "log"):
            self.log.info("Run SVM...")
        else:
            print("Run SVM...")
        acc = svc_classify(self.emb_x, self.emb_y, search=True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"SVM - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        if hasattr(self, "log"):
            self.log.info(f"SVM Avg. Accuracy (top-1): {acc}")
        else:
            print(f"SVM Avg. Accuracy (top-1): {acc}")
        

        #### MLP Classifier ####
        if hasattr(self, "log"):
            self.log.info("Run MLP...")
        else:
            print("Run MLP...")
        acc = mlp_classify(x=self.emb_x, y=self.emb_y, search=True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"MLP (Sklearn) - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        if hasattr(self, "log"):
            self.log.info(f"MLP Avg. Accuracy (top-1): {acc}")
        else:
            print(f"MLP Avg. Accuracy (top-1): {acc}")

        #### Random Forest Classifier ####
        if hasattr(self, "log"):
            self.log.info("Run Random Forest ...")
        else:
            print("Run Random Forest ...")
        acc = randomforest_classify(x=self.emb_x, y=self.emb_y, search=True)

        if plot:
            pca = PCA(n_components=2, random_state=123)
            x = pca.fit_transform(self.emb_x)

            _, ax = plt.subplots(figsize=(5,5))
            ax.set_title(f"Random Forest - Classifier (Avg. accuracy: {acc})")
            ax.scatter(x[:, 0], x[:, 1], c=pred.astype(int), facecolors='none', s=80,  linewidth=2,
                        cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.scatter(x[:, 0], x[:, 1], c=self.emb_y.astype(int),
                        cmap=sns.color_palette("Spectral", as_cmap=True), edgecolors='w')

        if hasattr(self, "log"):
            self.log.info(f"Random Forest Avg. Accuracy (top-1): {acc}")
        else:
            print(f"Random Forest Avg. Accuracy (top-1): {acc}")


def connect_to_slurm():
    """
    """
    with open(".slurm.json") as file:
        cfg_json = json.load(file)

    jhost = SSHClient()
    jhost.set_missing_host_key_policy(AutoAddPolicy()) 
    config = {
        "username": cfg_json["username"],
        "key_filename":  cfg_json["key_filename"],
    }
    jhost.connect(cfg_json["jhost"], **config)

    jhost_transport = jhost.get_transport()
    local_addr = (cfg_json["jhost"], 22) #edited#
    dest_addr = (cfg_json["slurm_host"], 22) #edited#
    jchannel = jhost_transport.open_channel("direct-tcpip", dest_addr, local_addr)


    slurm = SSHClient()
    slurm.set_missing_host_key_policy(AutoAddPolicy()) 
    slurm.connect(cfg_json["slurm_host"], username=cfg_json["username"],
                  password=cfg_json["password"] , sock=jchannel)
    
    return slurm, jhost