"""
    @author: jhuthmacher
"""
import io
import json
import os
import pprint
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import plotly.express as px

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from PIL import Image
import io


from paramiko import SSHClient, SSHConfig, AutoAddPolicy, SFTPClient

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.gridspec as gridspec
import seaborn as sns

import sacred

# pylint: disable=import-error
from config.config import log, create_logger
from data import KINECT_ADJACENCY
from data.tig_data_set import TIGDataset
from model import MLP, Solver, TemporalInfoGraph
from model.loss import jensen_shannon_mi, bce_loss
from tracker import Tracker
from evaluation import svc_classify, mlp_classify, randomforest_classify
from visualization.plots import plot_emb, plot_curve
from visualization import create_gif

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

        if "print_summary" in config and config["print_summary"]:
            summary(tig, input_size=(2, 36, 300), batch_size=config["loader"]["batch_size"])

        loss_fn = jensen_shannon_mi
        if "loss" in config and config["loss"] == "bce":
            loss_fn = bce_loss


        solver = Solver(tig, loader, loss_fn)

        #### Tracking ####
        tracker.track_traning(solver.train)(config["encoder_training"])

        if "emb_tracking" in config and config["emb_tracking"] != False:
            # Track the final embeddings after the TIG encoder is trained.
            emb_x = np.array([])
            emb_y = np.array([])
            with torch.no_grad():
                for batch_x, batch_y in train_loader:
                    pred, _ = solver.model(batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1))
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

##########################################################
# API to easily access experiment artifacts on the local #
# machine as well as on the slurm cluster (remote)       #
##########################################################
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
        self.ssh = ssh
        if ssh is None:
            try:
                data = np.load(f"{path}embeddings.npz", allow_pickle=True)
                emb_x = data["x"]
                emb_y = data["y"]

                self.emb_loader = DataLoader(list(zip(emb_x, emb_y)))

                #### Clean Storage ####
                del data
                del emb_x
                del emb_y
            except Exception as e:  #pylint: disable=bare-except
                # Not each experiement has embeddings saved
                try:
                    data = np.load(f"{path}embeddings_train.npz", allow_pickle=True)
                    emb_x = data["x"]
                    emb_y = data["y"]

                    self.emb_loader = DataLoader(list(zip(emb_x, emb_y)))

                    #### Clean Storage ####
                    del data
                    del emb_x
                    del emb_y
                except:
                    pass
                pass
            
            #### TIG Model/Losses/Metric ####
            try:
                self.classifier = torch.load(f"{path}TIG_MLP.pt")#.cuda()
                self.clf_train_loss = np.load(f"{path}TIG_MLP.train_losses.npy")
                self.clf_val_loss = np.load(f"{path}TIG_MLP.val_losses.npy")
                try:
                    self.clf_train_metrics = np.load(f"{path}TIG_MLP.train.metrics.npz")
                    self.clf_val_metrics = np.load(f"{path}TIG_MLP.val.metrics.npz")
                    if "top-k" in self.clf_val_metrics:
                        self.clf_val_metrics = {}
                        self.clf_val_metrics["val. top-1"] = np.load(f"{path}TIG_MLP.val.metrics.npz")["top-k"][:, 0]
                        self.clf_val_metrics["val. top-5"] = np.load(f"{path}TIG_MLP.val.metrics.npz")["top-k"][:, 1]
                        self.clf_train_metrics = {}
                        self.clf_train_metrics["top-1"] = np.load(f"{path}TIG_MLP.train.metrics.npz")["top-k"][:, 0]
                        self.clf_train_metrics["top-5"] = np.load(f"{path}TIG_MLP.train.metrics.npz")["top-k"][:, 1]
                except:
                    # Legacy support
                    self.clf_train_metrics = {}
                    self.clf_val_metrics = {}
                    self.clf_train_metrics["top-1"] = np.load(f"{path}TIG_MLP.train.metrics.npy")[:, 0]
                    self.clf_val_metrics["val. top-1"]  = np.load(f"{path}TIG_MLP.val.metrics.npy")[:, 0]
                    self.clf_train_metrics["top-5"] = np.load(f"{path}TIG_MLP.train.metrics.npy")[:, 1]
                    self.clf_val_metrics["val. top-5"]  = np.load(f"{path}TIG_MLP.val.metrics.npy")[:, 1]
            except:  #pylint: disable=bare-except
                # Not each experiement has  a classifier
                pass
            
            #### TIG Model/Losses ####
            try:
                self.tig_train_loss = np.load(f"{path}TIG_train_losses.npy")
                self.tig_val_loss = np.load(f"{path}TIG_val_losses.npy")
                self.tig = torch.load(f"{path}TIG_.pt")#.cuda()
            except:  #pylint: disable=bare-except
                pass
            

            #### TIG Train Metric ####
            try:
                self.tig_train_metrics = np.load(f"{path}TIG_train.metrics.npz")
                self.tig_val_metrics = np.load(f"{path}TIG_val.metrics.npz")
            except:  #pylint: disable=bare-except
                self.tig_train_metrics = {}
                self.tig_val_metrics = {}
                # self.tig_train_metrics["top-k"] = np.load(f"{path}TIG_train.metrics.npy")
                # self.tig_val_metrics["val. top-k"]  = np.load(f"{path}TIG_val.metrics.npy")
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
                try:
                    f = sftp.open(f"{path}embeddings_train.npz")
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
                except:
                    pass
                pass

            #### TIG Model/Losses/Metric ####
            try:                
                f = sftp.open(f"{path}TIG_MLP.train_losses.npy")
                f.prefetch()
                f = f.read()
                self.clf_train_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_MLP.val_losses.npy")
                f.prefetch()
                f = f.read()
                self.clf_val_loss = np.load(io.BytesIO(f))
                try:
                    f = sftp.open(f"{path}TIG_MLP.train.metrics.npz")
                    f.prefetch()
                    f = f.read()
                    self.clf_train_metrics = np.load(io.BytesIO(f))
                    f = sftp.open(f"{path}TIG_MLP.val.metrics.npz")
                    f.prefetch()
                    f = f.read()
                except:
                    self.clf_train_metrics = {}
                    self.clf_val_metrics = {}
                    f = sftp.open(f"{path}TIG_MLP.train.metrics.npy")
                    f.prefetch()
                    f = f.read()
                    self.clf_train_metrics["top-1"] = np.load(io.BytesIO(f))[:,0]
                    self.clf_train_metrics["top-5"] = np.load(io.BytesIO(f))[:,1]
                    f = sftp.open(f"{path}TIG_MLP.val.metrics.npy")
                    f.prefetch()
                    f = f.read()
                    self.clf_val_metrics["val. top-1"] = np.load(io.BytesIO(f))[:,0]
                    self.clf_val_metrics["val. top-5"] = np.load(io.BytesIO(f))[:,1]
                f = sftp.open(f"{path}TIG_MLP.pt")
                f.prefetch()
                f = f.read()
                self.classifier = torch.load(io.BytesIO(f))#.cuda()
            except:  #pylint: disable=bare-except
                # Not each experiement has  a classifier                
                pass
            
            #### TIG Model/Losses ####
            try:                
                f = sftp.open(f"{path}TIG_train_losses.npy")
                f.prefetch()    
                f = f.read()
                self.tig_train_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_val_losses.npy")
                f.prefetch()
                f = f.read()
                self.tig_val_loss = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_.pt")
                f.prefetch()
                f = f.read()
                self.tig = torch.load(io.BytesIO(f))#.cuda()
            except Exception as e:  #pylint: disable=bare-except
                print("Exception (TIG)", e)
                pass
                
            #### TIG Train Metric ####
            try:
                f = sftp.open(f"{path}TIG_train.metrics.npz")
                f.prefetch()
                f = f.read()
                self.tig_train_metrics = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_train.metrics.npz")
                f.prefetch()
                f = f.read()
                self.tig_val_metrics = np.load(io.BytesIO(f))
            except:  #pylint: disable=bare-except
                self.tig_train_metrics = {}
                self.tig_val_metrics = {}
                f = sftp.open(f"{path}TIG_train.metrics.npy")
                f.prefetch()
                f = f.read()
                self.tig_train_metrics["top-k"] = np.load(io.BytesIO(f))
                f = sftp.open(f"{path}TIG_train.metrics.npy")
                f.prefetch()
                f = f.read()
                self.tig_val_metrics["val. top-k"] = np.load(io.BytesIO(f))
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

        if "start_time" in self.config:
            runtime = str(datetime.now() -
                          datetime.strptime(self.config["start_time"],"%d.%m.%Y %H:%M:%S"))
        else:
            runtime = "Not available!"

        representation = f"Experiment ({os.path.normpath(self.path).split(os.sep)[-1]})\n"
        representation += f"--- Epoch: {self.tig_train_loss.shape[0] if hasattr(self, 'tig_train_loss') else 'Loss not available.'}\n"
        representation += f"--- Duration: {self.config['duration'] if 'duration' in self.config else 'Not Finished'}\n"
        representation += f"--- Runtime: {runtime}\n"
        representation += f"--- Batchsize: {self.config['loader']['batch_size'] if 'loader' in self.config else 'Config Incomplete'}\n"
        representation += f"--- Learning Rate: {self.config['encoder_training']['optimizer']['lr'] if 'encoder_training' in self.config else 'Config Incomplete'}\n"
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
    def plot_dash(self, figsize=(14, 7), mode="PCA"):
        """
        """
        plt.rcParams['figure.constrained_layout.use'] = True
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        # gs = gridspec.GridSpec(4,2, figure=fig)


        gs = fig.add_gridspec(4, 2)
        fig.suptitle(f"Experiment: {os.path.normpath(self.path).split(os.sep)[-1]}")    

        #### Embeddings ####
        ax = fig.add_subplot(gs[:, 0])
        try:
            plot_emb(self.emb_x, self.emb_y, ax = ax, title="Embeddings", mode=mode)
            # ax.set_aspect("equal", adjustable="box")
        except:
            pass

        #### TIG/MLP Loss/Metrics ####
        
        ax = fig.add_subplot(gs[0, 1]) if hasattr(self, "clf_train_loss") else fig.add_subplot(gs[0:2, 1])
        self.plot_curve(name="TIG.loss", ax = ax)

        ax = fig.add_subplot(gs[1, 1]) if hasattr(self, "clf_train_loss") else fig.add_subplot(gs[2:, 1])
        self.plot_curve(name="TIG.metric", ax = ax)

        if hasattr(self, "clf_train_loss") and len(self.clf_train_loss) > 0:
            ax = fig.add_subplot(gs[2, 1])
            self.plot_curve(name="MLP.loss", ax = ax)
            ax = fig.add_subplot(gs[3, 1])
            self.plot_curve(name="MLP.metric", ax = ax)
        # else:
        #     try:
        #         ax = fig.add_subplot(gs[2:, 1])
        #         i = io.BytesIO(self.img["MLP.loss.metric.final.png"])
        #         i = mpimg.imread(i, format='PNG')

        #         ax.imshow(i, interpolation='none', aspect='auto')
        #     except:
        #         pass
        #     ax.axis('off')

        return fig

    def plot_curve(self, name="TIG.loss", ax = None, which=None):
        """
        """
        args = None
        try:
            if name == "TIG.loss":
                model_name = "TIG: "
                title = "Loss" + ("BCE" if "loss" in self.config and self.config["loss"] == "bce" else "JSD")
                args = {
                    "data": {
                        "Train Loss": self.tig_train_loss,
                        "Val Loss": self.tig_val_loss
                        }
                    }
            elif name == "TIG.metric"  and hasattr(self, "tig_train_metrics"):
                model_name = "TIG: "
                title = "Metric"

                args = {
                    "line_mode": np.mean
                    }

                if which is None:
                    args["data"] = {**self.tig_train_metrics, **self.tig_val_metrics}

            elif name == "MLP.loss":
                model_name = "MLP: "
                title = "Loss (Cross Entropy)"

                args = {
                    "data": {
                        "Train Loss": self.clf_train_loss,
                        "Val Loss": self.clf_val_loss
                        }
                    }
            elif name == "MLP.metric" and hasattr(self, "clf_val_metrics") and hasattr(self, "clf_train_metrics"):
                model_name = "MLP: "
                title = "Accuracy (Top-1, Top-5)"

                args = {
                    "data": self.clf_train_metrics,
                    "line_mode": np.mean
                    }
                if hasattr(self, "clf_val_metrics"):
                    args["data"] = {**self.clf_train_metrics, **self.clf_val_metrics}


        except Exception as e:
            print(e)
            if ax is not None:
                ax.axis('off')
            return None
        if args is not None:
            return plot_curve(**args, title=title, model_name=model_name, ax=ax)

    def plot_emb(self, mode="PCA", label=None):
        """
        """
        return plot_emb(self.emb_x, self.emb_y, mode=mode, label = label)

    def plot_class_distr(self):
        fig, ax = plt.subplots(figsize=(7,5))

        unique, counts = np.unique(self.emb_y, return_counts=True)

        ax.bar(unique, counts)

        return ax.figure

    ### Experiment Evaluation ####
    def evaluate_emb(self, plot: bool = False, which: [str] = None):
        """ Evaluate the quality of the embeddings by testing different classifier.

            Paramters:
                plot: bool
                    If true the embeddings with their predictions are plotted.
        """
        if hasattr(self, "clf_val_metrics"):
            if hasattr(self, "log"):
                self.log.info(f"(Pipeline) MLP Accuracy: {np.max(self.clf_val_metrics['val. top-1'])}, {np.max(self.clf_val_metrics['val. top-5'])}")
            else:
                print(f"(Pipeline) MLP Accuracy: {np.max(self.clf_val_metrics['val. top-1'])}, {np.max(self.clf_val_metrics['val. top-5'])}")
    
        #### SVM Classifier ####
        if which is None or "svm" in which:
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
            
            if self.ssh is not True:
                f = Path(f"{self.path}sklearn.evaluation.txt")
                f = open(f"{self.path}sklearn.evaluation.txt", "a")
                f.write(f"SVM Avg. Accuracy (top-1): {acc}\n")
                f.close()

        #### MLP Classifier ####
        if which is None or "mlp" in which:
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
            
            if self.ssh is not True:
                f = Path(f"{self.path}sklearn.evaluation.txt")
                f = open(f"{self.path}sklearn.evaluation.txt", "a")
                f.write(f"MLP Avg. Accuracy (top-1): {acc}\n")
                f.close()

        #### Random Forest Classifier ####
        if which is None or "random forest" in which:
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
            
            if self.ssh is not True:
                f = Path(f"{self.path}sklearn.evaluation.txt")
                f = open(f"{self.path}sklearn.evaluation.txt", "a")
                f.write(f"Random Forest Avg. Accuracy (top-1): {acc}\n")
                f.close()

    def plot_emb_3D(self, gif_path: str = None, plotly= True, label=None):
        """
        """
        pca = PCA(n_components=3, random_state=123)
        x = self.emb_x
        y = self.emb_y

        if label is not None:
            x = x[np.isin(y, label)]
            y = y[np.isin(y, label)]
        x = pca.fit_transform(x)

        if plotly:
            fig = px.scatter_3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], 
              color=y.astype(int))#,  colorscale='Viridis')#colormap=sns.color_palette("Spectral", as_cmap=True))
            fig.show()
            return

        for angle in tqdm(range(0, 360, 2)):
            fig = plt.figure(figsize=(8,8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x[:, 0], x[:, 1], x[:, 2],  c=y.astype(int), cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.view_init(10, angle)
            ax.set_facecolor("white")

            if gif_path is not None:
                create_gif(fig, path=gif_path, name="embeddings_3d.gif")
                plt.close()
            else:
                break

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