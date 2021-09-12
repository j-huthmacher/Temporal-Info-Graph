""" In this file we have some helper functions and classes to load experiment
    in easy-to-use objects (as a experiment API). Those objects directly have the data
    loaded and implement some often used plotting functions.

"""
# pylint: disable=bare-except, line-too-long, broad-except
import io
import json
import os
import pprint
from datetime import datetime
from tqdm import tqdm
from pathlib import Path
import plotly.express as px
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary

from PIL import Image

from paramiko import SSHClient, SSHConfig, AutoAddPolicy, SFTPClient

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import matplotlib.gridspec as gridspec
import seaborn as sns

import sacred

# pylint: disable=import-error
from config.config import log, create_logger
from utils import KINECT_ADJACENCY
from utils.tig_data_set import TIGDataset
from model import TemporalInfoGraph, TemporalInfoGraphLSTM
from baseline import MLP
from model.loss import jensen_shannon_mi, bce_loss, hypersphere_loss
# from evaluation import svc_classify, mlp_classify, randomforest_classify
from visualization.plots import plot_emb, plot_curve
from visualization import create_gif

import warnings
warnings.filterwarnings("ignore")

##########################################################
# API to easily access experiment artifacts on the local #
# machine as well as on the slurm cluster (remote)       #
##########################################################


class Experiment():
    """ Experiment class to easy access experiments and load all experiment aritfacts
        direct into memory from local locations or via ssh.
    """

    def __init__(self, path: str, con: str = "local"):
        """ Initilization.

            Args:
                path: str
                    Path pointing to the experiment folder created by an experiment run.
                con: str
                    Determines the connection type which is used to loade the experiment.
                    Options ["local", "ssh"]
        """
        self.img = {}
        self.path = path
        self.folder = os.path.normpath(self.path).split(os.sep)[0]
        self.name = os.path.normpath(self.path).split(os.sep)[-1]
        self.load = {
            "local": self.local_load,
            "ssh": self.ssh_load
        }

        #### Load Attributes ####
        # Since we might use different ways to loade functions, the easiest way
        # to handle missing data (e.g. because the experiment is still running)
        # is to use try/except.

        # Load Embeddings
        try:
            data = self.load[con]("embeddings", "embeddings_full.npz")
            self.emb_loader = DataLoader(list(zip(data["x"], data["y"])))
        except Exception:
            try:
                data = self.load[con]("embeddings", "embeddings_train.npz")
                self.emb_loader = DataLoader(list(zip(data["x"], data["y"])))
            except Exception:
                pass

        try:
            data = self.load[con]("embeddings", "embeddings_best.npz")
            self.emb_loader_best = DataLoader(list(zip(data["x"], data["y"])))
        except Exception:
            pass

        #### TIG Artifacts ####
        try:
            self.tig_train_loss = self.load[con](
                "loss", "TIG_train_losses.npy")
            self.tig_train_metrics = self.load[con](
                "metric", "TIG_train.metrics.npz")

            self.tig_val_loss = self.load[con]("loss", "TIG_val_losses.npy")
            self.tig_val_metrics = self.load[con](
                "metric", "TIG_val.metrics.npz")
            try:
                self.tig_best = self.load[con]("model", "TIG__best.pt")
            except Exception:
                pass

            self.tig = self.load[con]("model", "TIG_.pt")
        except Exception as e:
            print(e)

        #### TIG Discriminator Scores ####
        try:
            self.tig_discriminator = self.load[con](
                "discriminator", "discriminator.npy")
        except Exception:
            pass

        #### MLP Artifacts #####
        try:
            #### MLP Loss ####
            self.clf_train_loss = self.load[con](
                "loss", "TIG_MLP.train_losses.npy")
            self.clf_val_loss = self.load[con](
                "loss", "TIG_MLP.val_losses.npy")

            #### MLP Metrics ####
            try:
                self.clf_train_metrics = self.load[con](
                    "metric", "TIG_MLP.train.metrics.npz")
                self.clf_val_metrics = self.load[con](
                    "metric", "TIG_MLP.val.metrics.npz")
                if "top-k" in self.clf_val_metrics:
                    self.clf_val_metrics = {}
                    self.clf_val_metrics["val. top-1"] = self.load[con]("metric",
                                                                        "TIG_MLP.train.metrics.npz")["top-k"][:, 0]
                    self.clf_val_metrics["val. top-5"] = self.load[con]("metric",
                                                                        "TIG_MLP.val.metrics.npz")["top-k"][:, 1]
            except Exception:
                # Legacy support
                self.clf_train_metrics = {}
                self.clf_val_metrics = {}
                self.clf_train_metrics["top-1"] = self.load[con](
                    "metric", "TIG_MLP.train.metrics.npy")[:, 0]
                self.clf_train_metrics["top-5"] = self.load[con](
                    "metric", "TIG_MLP.train.metrics.npy")[:, 1]
                self.clf_val_metrics["val. top-1"] = self.load[con](
                    "metric", "TIG_MLP.val.metrics.npy")[:, 0]
                self.clf_val_metrics["val. top-5"] = self.load[con](
                    "metric", "TIG_MLP.val.metrics.npy")[:, 1]

            #### MLP conl ####
            self.classifier = self.load[con]("conl", "TIG_MLP.pt")
        except Exception:
            pass

        try:
            self.config = self.load[con]("config", None)
        except Exception:
            self.config = {}

        if con == "local":
            #### Set Up Logging ####
            self.log = create_logger(fpath=self.config["exp_path"],
                                     name="EXP Logger", suffix="EXP_")

    def ssh_load(self, name: str, file_name: str):
        """ Load artifacts through SSH with SFTP.

            Args:
                name: str
                    Name of the artifact that is loaded.
                    Options ["embeddings", "model", "loss", "metric", "config", "discriminator"]
                file_name: str
                    File name of the artifact on the remote server.
            Return:
                Any: Returns the artifact, e.g. np.ndarray for the loss or a torch.nn.Module
                     for the model artifact.
        """
        slurm, jhost = connect_to_slurm()
        sftp = SFTPClient.from_transport(slurm.get_transport())

        if name == "embeddings":
            f = sftp.open(self.path + file_name)
            f.prefetch()
            f = f.read()
            return np.load(io.BytesIO(f))
        elif name == "model":
            f = sftp.open(self.path + file_name)
            f.prefetch()
            f = f.read()
            return torch.load(io.BytesIO(f))
        elif name == "loss":
            f = sftp.open(self.path + file_name)
            f.prefetch()
            f = f.read()
            return np.load(io.BytesIO(f))
        elif name == "metric":
            f = sftp.open(self.path + file_name)
            f.prefetch()
            f = f.read()
            return np.load(io.BytesIO(f))
        elif name == "config":
            f = sftp.open(f"{self.path}config.json")
            f.prefetch()
            f = f.read()
            config = json.load(io.BytesIO(f))
            config["exp_path"] = self.path
            return config
        elif name == "discriminator":
            f = sftp.open(f"{self.path}config.json")
            f.prefetch()
            f = f.read()
            return np.load(io.BytesIO(f))

        slurm.close()
        jhost.close()

    def local_load(self, name: str, file_name: str):
        """ Load data from local storage

            Args:
                name: str
                    Name of the artifact that is loaded.
                    Options ["embeddings", "model", "loss", "metric", "config", "discriminator"]
                file_name: str
                    File name of the artifact on the remote server.
            Return:
                Any: Returns the artifact, e.g. np.ndarray for the loss or a torch.nn.Module
                     for the model artifact.
        """
        if name == "embeddings":
            return np.load(self.path + file_name, allow_pickle=True)
        elif name == "model":
            return torch.load(self.path + file_name)
        elif name == "loss":
            return np.load(self.path + file_name)
        elif name == "metric":
            return np.load(self.path + file_name)
        elif name == "config":
            with open(f"{self.path}config.json") as file:
                config = json.load(file)
                config["exp_path"] = self.path
            return config
        elif name == "discriminator":
            return np.load(self.path + file_name)

    def show_img(self, name: str):
        """ Function to display loaded images (e.g. plots that were created).

            Args:
                name: str
                    Name of the image that should be displayed.
            Return:
                PIL.Image.Image: Image object of the desired image.
        """
        image = Image.open(io.BytesIO(self.img[name]))
        return image

    def tig_input_shape(self, n_batches: int = 2, n_nodes: int = 36, n_times: int = 300):
        """ Function to extract the input shape to the model in an experiment.

            Args:
                n_batches: int
                    Number of batches.
                n_nodes: int
                    Number of nodes of the graph.
                n_times: int
                    Number of timesteps.
            Return:
                tuple: (# batches, input dimension of the model, # nodes, # timesteps)
        """
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
        except Exception:
            return None

    @property
    def emb_best(self):
        """ Embeddings from the encoder (corresponding to the lowest loss).
        """
        try:
            return self.emb_loader_best.dataset
        except Exception:
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

    @property
    def emb_x_best(self):
        """ Features of the best emebddings (corresponding to lowest loss).
        """
        return np.array(list(np.array(self.emb_best)[:, 0])) if self.emb_best is not None else None

    @property
    def emb_y_best(self):
        """ Labels of the best emebddings (corresponding to lowest loss).
        """
        return np.array(list(np.array(self.emb_best)[:, 1])) if self.emb_best is not None else None

    def __repr__(self):
        """ Text representation to represent the experiment at a glance.
        """

        if "start_time" in self.config:
            runtime = str(datetime.now() - datetime.strptime(self.config["start_time"],
                                                             "%d.%m.%Y %H:%M:%S"))
        else:
            runtime = "Not available!"

        representation = f"Experiment ({os.path.normpath(self.path).split(os.sep)[-1]})\n"
        representation += f"- -- Epoch: {(self.tig_train_loss.shape[0]
                                          if hasattr(self, 'tig_train_loss')
                                          else 'Loss not available.')}\n"
        representation += f"- -- Duration: {(self.config['duration']
                                             if 'duration' in self.config
                                             else 'Not Finished')}\n"
        if 'duration' not in self.config:
            representation += f"--- Runtime: {runtime}\n"
        representation += f"- -- Batchsize: {(self.config['loader']['batch_size']
                                              if 'loader' in self.config
                                              else 'Config Incomplete')}\n"
        representation += f"- -- Learning Rate: {(self.config['encoder_training']['optimizer']['lr']
                                                  if 'encoder_training' in self.config
                                                  else 'Config Incomplete')}\n"
        representation += f"- -- Training set: {(self.config['train_length']
                                                 if 'train_length' in self.config
                                                 else 'Config Incomplete')}\n"
        representation += f"- -- Validation set: {(self.config['val_length']
                                                   if 'val_length' in self.config
                                                   else 'Config Incomplete')}\n"
        representation += f"- -- Last JSD Loss(Train): {(self.tig_train_loss[-1]
                                                         if hasattr(self, 'tig_train_loss')
                                                         else 'Not Finished')}\n"
        representation += f"- -- Last JSD Loss(Val): {(self.tig_val_loss[-1]
                                                       if hasattr(self, 'tig_val_loss')
                                                       else 'Not Finished')}\n"
        representation += f"- -- Embedding Shape: {(self.emb_x.shape
                                                    if self.emb_x is not None
                                                    else 'Not Finished')}\n"

        return representation

    def tensorboard(self, replace: bool = False):
        """ Create Tensor board based on the experiment.

            Args:
                replace: bool
                    Flag to determine if an existing tensorboard should be replaced.
        """
        if not os.path.exists(f"./runs/{self.name}") and not replace:
            writer = SummaryWriter(log_dir=f"./runs/{self.name}")
            writer.add_graph(self.tig.cpu(), torch.rand(
                self.tig_input_shape).cpu())
            for i, val in enumerate(self.tig_train_loss):
                writer.add_scalar('training loss', val, i)

            for i, val in enumerate(self.tig_val_loss):
                writer.add_scalar('val loss', val, i)

            text = pprint.pformat(self.config, indent=1,
                                  width=80).replace("\n", "  \n")
            text = text.replace("     ", "&nbsp;")
            writer.add_text("config", text)

    #### Experiment Visualization ####
    def plot_dash(self, figsize: tuple = (14, 7), mode: str = "PCA"):
        """ Plots the embeddings and the loss of the TIG and downstream classifier.

            Args:
                figsize: tuple
                    Defines the size of the matplot figure.
                mode: str
                    Determines the mode which is used to plot high dinmensional embeddings.
            Returns:
                matplotlib.figure.Figure: Figure of the created plot.
        """
        plt.rcParams['figure.constrained_layout.use'] = True
        fig = plt.figure(figsize=figsize, constrained_layout=True)
        # gs = gridspec.GridSpec(4,2, figure=fig)

        gs = fig.add_gridspec(4, 2)
        fig.suptitle(
            f"Experiment: {os.path.normpath(self.path).split(os.sep)[-1]}")

        #### Embeddings ####
        ax = fig.add_subplot(gs[:, 0])
        try:
            plot_emb(self.emb_x, self.emb_y, ax=ax,
                     title="Embeddings", mode=mode)
            # ax.set_aspect("equal", adjustable="box")
        except Exception:
            pass

        #### TIG/MLP Loss/Metrics ####
        ax = fig.add_subplot(gs[0, 1]) if hasattr(
            self, "clf_train_loss") else fig.add_subplot(gs[0:2, 1])
        self.plot_curve(name="TIG.loss", ax=ax)

        ax = fig.add_subplot(gs[1, 1]) if hasattr(
            self, "clf_train_loss") else fig.add_subplot(gs[2:, 1])
        self.plot_curve(name="TIG.metric", ax=ax)

        if hasattr(self, "clf_train_loss") and len(self.clf_train_loss) > 0:
            ax = fig.add_subplot(gs[2, 1])
            self.plot_curve(name="MLP.loss", ax=ax)
            ax = fig.add_subplot(gs[3, 1])
            self.plot_curve(name="MLP.metric", ax=ax)

        return fig

    def plot_curve(self, name: str = "TIG.loss", ax: matplotlib.axes.Axes = None,
                   which: [str] = None):
        """ Function to plot different types of curves, e.g. the loss curve.

            Args:
                name: str
                    Defines which curve should be plotted. The name follows the
                    forma "Model.Curve".
                    Options: ["TIG.loss", "TIG.metric", "MLP.loss", "MLP.metric"]
                    Hint: The MLP curves are generally corresponds to the classifier, if one
                    exists in the experiment.
                ax: matplotlib.axes.Axes
                    Axes for the plot. If None a new axis is created.
                which: [str]
                    Array of strings that identfies the metrics.
                    General options are ["accuracy", "auc", "prec", "top-1", "top-5"].
                    However the options depend on how the experiment is customized.
            Return:
                matplotlib.figure.Figure: Figure of the plot.
        """
        # Plot arguments
        args = None
        try:
            if name == "TIG.loss":
                model_name = "TIG: "
                title = "Loss" + ("BCE"
                                  if "loss" in self.config and self.config["loss"] == "bce"
                                  else "JSD")
                # Handover the data that is plotted.
                args = {
                    "data": {
                        "Train Loss": self.tig_train_loss
                    }
                }

                # Add validation loss if it exists. One could also plot the intermediate
                # results during training.
                if hasattr(self, "tig_val_loss"):
                    args["data"]["Val Loss"] = self.tig_val_loss

            elif name == "TIG.metric" and hasattr(self, "tig_train_metrics"):
                model_name = "TIG: "
                title = "Metric"

                # Define which type of aggregation is added to the plot as an 
                # single number indicator.
                args = {
                    "line_mode": np.mean
                }

                if which is None:
                    args["data"] = {**self.tig_train_metrics}
                else:
                    args["data"] = {}
                    for metric in which:
                        metric_name = "Train " + metric.capitalize()
                        args["data"][metric_name] = self.tig_train_metrics[metric]

                        if hasattr(self, "tig_val_metrics"):
                            metric_name = "Val " + metric.capitalize()
                            args["data"][metric_name] = self.tig_val_metrics["val. " + metric]

            elif name == "MLP.loss":
                model_name = "MLP: "
                title = "Loss (Cross Entropy)"

                args = {
                    "data": {
                        "Train Loss": self.clf_train_loss,
                        "Val Loss": self.clf_val_loss
                    }
                }
            elif name == "MLP.metric" and hasattr(self, "clf_train_metrics"):
                model_name = "MLP: "
                title = "Accuracy (Top-1, Top-5)"

                args = {
                    "data": {**self.clf_train_metrics},
                    "line_mode": np.mean
                }
                if hasattr(self, "clf_val_metrics"):
                    args["data"] = {**self.clf_train_metrics,
                                    **self.clf_val_metrics}
            else:
                print("name not known")

        except Exception as e:
            print(e)
            if ax is not None:
                ax.axis('off')
            return None
        if args is not None:
            return plot_curve(**args, title=title, model_name=model_name, ax=ax)

    def plot_emb_best(self, **kwargs: dict):
        """ Function to plot the "best" embeddings.

            Best in this context mean the embeddings causing the lowest loss during training.

            Args:
                **kwargs: dict
                    Arguments for the plot_emb() function.
            Return:
                matplotlib.figure.Figure: Figure of the plot.
        """
        return plot_emb(self.emb_x_best, self.emb_y_best, **kwargs)

    def plot_emb(self, x: Any = None, y: Any = None, **kwargs: dict):
        """ Function to plot the latest embeddings of an experiment.

            Args:
                x: Any
                    Either np.ndarray or torch.Tensor that contains the features.
                y: Any
                    Either np.ndarray or torch.Tensor that contains the corresponding labels
                    for the given features.
                **kwargs: dict
                    Arguments for the plot_emb() function.
            Return:
                matplotlib.figure.Figure: Figure of the plot.            
        """
        if x is None and y is None:
            return plot_emb(self.emb_x, self.emb_y, **kwargs)
        else:
            return plot_emb(x, y, **kwargs)

    def plot_class_distr(self):
        """ Function to plot the distribution of classes in the experiment.
        """
        fig, ax = plt.subplots(figsize=(7, 5))

        unique, counts = np.unique(self.emb_y, return_counts=True)

        ax.bar(unique, counts)

        return ax.figure

    def plot_emb_3D(self, gif_path: str = None, plotly: bool = True, label: [int] = None):
        """ Function to plot the latest embeddings in 3D.

            Args:
                gif_path: str
                    Path where the gif is stored if theplot is not created with plotly.
                plotly: bool
                    Flag to determine whether plotly is used to create the plot.
                label: [int]
                    List of selected labels that are considered for the plot.
            Return:
                None
        """
        pca = PCA(n_components=3, random_state=123)
        if self.emb_x is not None:
            x = self.emb_x
            y = self.emb_y
        else:
            x = self.emb_x_best
            y = self.emb_y_best

        spectral_cmap = sns.color_palette("Spectral", as_cmap=True)
        spectral_rgb = []
        norm = mpl.colors.Normalize(vmin=0, vmax=255)

        if label is not None:
            x = x[np.isin(y, label)]
            y = y[np.isin(y, label)]
        x = pca.fit_transform(x)

        if plotly:
            fig = px.scatter_3d(x=x[:, 0], y=x[:, 1], z=x[:, 2], color=y.astype(int))
            fig.show()
            return

        for angle in tqdm(range(0, 360, 2)):
            fig = plt.figure(figsize=(8, 8))
            ax = fig.add_subplot(111, projection='3d')

            ax.scatter(x[:, 0], x[:, 1], x[:, 2], c=y.astype(int),
                       cmap=sns.color_palette("Spectral", as_cmap=True))
            ax.view_init(10, angle)
            ax.set_facecolor("white")

            if gif_path is not None:
                create_gif(fig, path=gif_path, name="embeddings_3d.gif")
                plt.close()
            else:
                break


####################
# Helper Functions #
####################

def connect_to_slurm():
    """ Helper function to connect to a slurm cluster.

        The connection is established by the log in data and adresses written
        in ".slurm.json". One have to provide the following values.
        {
            "username": Username used for slurm,
            "password": Password to connect to slurm,
            "key_filename": File name of the key file (for the jump host),
            "jhost": Jump host,
            "slurm_host": Actual adress of the slurm machine
        }
    """
    with open(".slurm.json") as file:
        cfg_json = json.load(file)

    jhost = SSHClient()
    jhost.set_missing_host_key_policy(AutoAddPolicy())
    config = {
        "username": cfg_json["username"],
        "key_filename": cfg_json["key_filename"],
    }
    jhost.connect(cfg_json["jhost"], **config)

    jhost_transport = jhost.get_transport()
    local_addr = (cfg_json["jhost"], 22)  # edited#
    dest_addr = (cfg_json["slurm_host"], 22)  # edited#
    jchannel = jhost_transport.open_channel(
        "direct-tcpip", dest_addr, local_addr)

    slurm = SSHClient()
    slurm.set_missing_host_key_policy(AutoAddPolicy())
    slurm.connect(cfg_json["slurm_host"], username=cfg_json["username"],
                  password=cfg_json["password"], sock=jchannel)

    return slurm, jhost
