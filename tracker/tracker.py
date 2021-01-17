"""
    @author: jhuthmacher
"""
import io
import os
from typing import Any
from pathlib import Path
from datetime import datetime
import shutil
import json
import warnings
from shutil import copyfile
import sys
import psutil

import torch
import pymongo
import gridfs
import numpy as np
import matplotlib.pyplot as plt

from sacred import Experiment
from sacred.observers import MongoObserver

# pylint: disable=import-error
from model import MLP, TemporalInfoGraph
from model import get_negative_expectation as neg_exp
from model import get_positive_expectation as pos_exp
from config.config import log, formatter, logging
from visualization import create_gif, plot_eval, plot_heatmaps, plot_heatmap, plot_loss_metric
from visualization import plot_curve
from data import KINECT_ADJACENCY

#### "Global" Access to Tracker ####
tracker = None

# TODO: This should be part of the solver.
acc = {}
e_pos = {}
e_neg = {}
mem_used = []
gpu_mem_used = []


class Tracker(object):
    """ Tracker class to do organized and structured tracking.

        In general the tracker works like this that it access class/function attributes during the
        execution of a procedure and store them locally or remotely. This means to track a specific
        value it has to exists as accessble variable somewhere.
    """

    def __init__(self, ex_name: str = None, sacred_cfg: dict = {}, config: dict = {},
                 local: bool = True, local_path: str = None, track_decision: bool = False):
        """ Initialization of the tracker.

            Paramter:
                ex_name: str
                    Experiment name. If you use the remote tracking by sacred you can either define
                    the name by this paramter or by the sacred_cfg.
                sacred_cfg: dict
                    Configuration for remote tracking with sacred.
                    Definition (with type and default values):
                    {
                        "ex_name": str = None,
                        "db": str = "temporal_info_graph",
                        "interactive": booo = False,
                        "db_url": str = None
                    }
                config: dict
                    Tracker configuration (merged with class attributes!).
                local: bool
                    Flag to decide if the tracking is done locally or remotely by sacred.
                local_path: str
                    Local path where the (intermediate) results are stored. If it is not provided
                    a folder called 'output' is created in the working directory.
                track_decision: bool
                    Flag to decide if the intermediate results, such as the learned decision
                    boundaries, should be visualized in each epoch.
        """
        super().__init__()

        #### Set default configuration ####
        self.sacred_cfg = {**{
            "ex_name": ex_name,
            "db": "temporal_info_graph",
            "interactive": False,
            "db_url": None
        }, **sacred_cfg}

        #### Make tracker obj globally available ####
        global tracker
        tracker = self

        if (self.sacred_cfg is not None and
            self.sacred_cfg["ex_name"] is not None and
                self.sacred_cfg["db_url"] is not None):
            #### Do remote tracking with sacred ####
            client = pymongo.MongoClient(self.sacred_cfg["db_url"])

            self.ex_name = self.sacred_cfg["ex_name"]
            self.ex = Experiment(self.sacred_cfg["ex_name"],
                                 interactive=self.sacred_cfg["interactive"])
            self.ex.logger = log
            self.ex.observers.append(MongoObserver(
                client=client, db_name=self.sacred_cfg["db"]))
            self.observer = self.ex.observers[0]
            self.run_collection = self.ex.observers[0].runs

            self.db_url = self.sacred_cfg["db_url"]
            self.db = client[self.sacred_cfg["db"]]
            self.interactive = self.sacred_cfg["interactive"]
            self.id = None
            self.run = None
        else:
            #### Local Tracking ####
            self.ex = None

        #### Set local attributes ####
        self.tag = ""
        self.local = local
        self.track_decision = track_decision

        #### Prepare local tracking ####
        self.date = datetime.now().strftime("%d%m%Y_%H%M")

        self.checkpoint_dict = {
            "model_params": {},
            "optim_params": {},
        }

        self.local_path = f"./output/{self.date}_{ex_name}/" if local_path is None else local_path
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

        #### Adapt Logger ####
        f_name = log.handlers[1].baseFilename
        copyfile(f_name, f"{self.local_path}{f_name.split(os.sep)[-1]}")

        fh = logging.FileHandler(
            f"{self.local_path}{f_name.split(os.sep)[-1]}")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        log.addHandler(fh)

        log.removeHandler(log.handlers[1])
        log.addHandler(fh)

        self.save_nth = 100

        # Assign model config
        self.__dict__ = {**self.__dict__, **config}

        self.cfg = {}

    @property
    def model(self):
        """ Property to return the model that is tracked.

            Return:
                torch.nn.Module: PyTorch model that is tracked (and trained).
        """
        if hasattr(self, "solver") or self.solver is None:
            ValueError(
                "Solver object not provided! I.e. the code to train the model isn't there.")
        if hasattr(self.solver, "model") or self.solver.model is None:
            ValueError("Solver contains no model!")
        return self.solver.model

    @property
    def checkpoint(self):
        """ Checkpoint of the training/testing procedure.
        """
        try:
            return torch.load(f"{self.local_path}checkpoints/checkpoint.pt")
        except Exception as e:  # pylint: disable=broad-except,unused-variable
            # Checkpoint doesn't exists
            return None

    # TODO: This doesn't work yet!
    # pylint: disable=unused-argument
    def new_ex(self, ex_name: str):
        """ Function to create a new experiment with an already initialized tracker object.

            Paramter:
                ex_name: str
                    New experiment name.
        """
        # self.ex_name = ex_name
        # self.ex = Experiment(ex_name, interactive=self.interactive)
        # self.ex.observers.append(MongoObserver(url=self.db_url, db_name=self.db))
        # self.observer = self.ex.observers[0]
        # self.run_collection = self.ex.observers[0].runs
        warnings.warn("Not implemented yet! Nothing happened.")

    def track(self, mode: Any, cfg: dict = None):
        """ General tracking function that distributes the tracking task depending on the mode.

            Paramters:
                mode: Callable or str
                    Mode can be either a function or a string. In case of an function the track
                    function starts the experiment with the handovered function as main (sacred).
                    If it is a string the experiment is already started and the function delegate
                    the tracking task to the corresponding function.
                    E.g. you can handover solver.train() to track the trianing.
                cfg: dict
                    Configuration of the experiment.
        """
        if callable(mode):
            #### Mode corresponds to an function ####
            self.cfg = cfg
            self.cfg["start_time"] = datetime.now().strftime(
                "%d.%m.%Y %H:%M:%S")

            Path(self.local_path).mkdir(parents=True, exist_ok=True)

            with open(f'{self.local_path}/config.json', 'w') as fp:
                json.dump(self.cfg, fp)
            
            if self.ex is not None:
                self.ex.main(mode(self, cfg))
                self.ex.run()
            else:
                # Local execution
                mode(self, cfg)()

            self.cfg["end_time"] = datetime.now().strftime("%d.%m.%Y %H:%M:%S")
            self.cfg["duration"] = str(datetime.strptime(self.cfg["end_time"],
                                                         "%d.%m.%Y %H:%M:%S") -
                                       datetime.strptime(self.cfg["start_time"],
                                                         "%d.%m.%Y %H:%M:%S"))

            if self.local:
                self.track_locally()

        elif mode == "epoch":
            self.track_epoch()
        elif mode == "training":
            self.track_train()  # Not used yet!
        elif mode == "validation":
            self.track_validation()  # Not used yet!
        elif mode == "evaluation":
            self.track_evaluation()  # Not used yet!
        elif mode == "loss":
            self.track_loss()

    #### (Individual) Tracker Functions #####
    def track_loss(self):
        """ Function to track the loss calculation.
        """
        if "memory" in self.cfg["visuals"]:
            #### Check Memory Consumption ####
            mem = psutil.virtual_memory()
            mem_used.append(mem.used/1024**3)

            fig, ax = plt.subplots(2, 1, figsize=(15, 6))
            ax[0].plot(mem_used)

            ax[0].axhline(mem.total/1024**3, ax[0].get_xlim()[0],
                        ax[0].get_xlim()[1], linestyle="--", linewidth=1)

            if len(mem_used) > len(self.solver.train_loader):
                x = np.arange(len(self.solver.train_loader), len(
                    mem_used), len(self.solver.train_loader))
                ymax = [ax[0].get_ylim()[1]] * len(x)
                ymin = [ax[0].get_ylim()[0]] * len(x)
                if len(x) > 20:
                    x = x[::3]
                    ymax = ymax[::3]
                    ymin = ymin[::3]
                elif len(x) > 500:
                    x = x[:500:3]
                    ymax = ymax[:500:3]
                    ymin = ymin[:500:3]
                ax[0].vlines(x, ymin, ymax, label="epoch",
                            linestyles=":", linewidth=1)

            ax[0].set_ylabel("GB")
            ax[0].set_title("Memory Usage")

            #### Plot GPU Memory ####
            mem = torch.cuda.memory_stats()["allocated_bytes.all.current"]
            total_mem_gpu = torch.cuda.get_device_properties(
                0).total_memory/1024**3
            gpu_mem_used.append(mem/1024**3)

            ax[1].plot(gpu_mem_used)

            ax[1].axhline(total_mem_gpu, ax[1].get_xlim()[0],
                        ax[1].get_xlim()[1], linestyle="--", linewidth=1)

            if len(gpu_mem_used) > len(self.solver.train_loader):
                x = np.arange(len(self.solver.train_loader), len(
                    gpu_mem_used), len(self.solver.train_loader))
                ymax = [ax[1].get_ylim()[1]] * len(x)
                ymin = [ax[1].get_ylim()[0]] * len(x)
                if len(x) > 20:
                    x = x[::3]
                    ymax = ymax[::3]
                    ymin = ymin[::3]
                elif len(x) > 500:
                    x = x[:500:3]
                    ymax = ymax[:500:3]
                    ymin = ymin[:500:3]
                ax[1].vlines(x, ymin, ymax, label="epoch",
                            linestyles=":", linewidth=1)

            ax[1].set_ylabel("GB")
            ax[1].set_title("GPU Memory Usage")

            fig.savefig(self.local_path+"MEM_USAGE.png")
            plt.close('all')

        if isinstance(self.solver.model, TemporalInfoGraph):
            loss_fn = self.solver.loss_fn

            #### Track statistical values of the loss ####
            f = Path(f"{self.local_path}loss.stats.samples.csv")
            if not f.is_file():
                f = open(f"{self.local_path}loss.stats.samples.csv", "a")
                f.write(
                    "num_pos_samples,num_neg_samples,ratio,num_graphs,num_nodes\n")
                f.write(f"{loss_fn.pos_samples.numel()},")
                f.write(f"{loss_fn.neg_samples.numel()},")
                f.write(
                    f"{(loss_fn.neg_samples.numel())/(loss_fn.pos_samples.numel() + loss_fn.neg_samples.numel())},")
                f.write(f"{loss_fn.num_graphs},")
                f.write(f"{loss_fn.num_nodes}\n")
            else:
                f = open(f"{self.local_path}loss.stats.samples.csv", "a")
                f.write(f"{loss_fn.pos_samples.numel()},")
                f.write(f"{loss_fn.neg_samples.numel()},")
                f.write(
                    f"{(loss_fn.neg_samples.numel())/(loss_fn.pos_samples.numel() + loss_fn.neg_samples.numel())},")
                f.write(f"{loss_fn.num_graphs},")
                f.write(f"{loss_fn.num_nodes}\n")
            f.close()
            
            if "discriminator" in self.cfg["visuals"]:
                f = Path(f"{self.local_path}loss.stats.expectations.csv")
                if not f.is_file():
                    f = open(f"{self.local_path}loss.stats.expectations.csv", "a")
                    f.write("E_neg,E_pos,overall_loss\n")
                    f.write(
                        f"{loss_fn.E_neg},{loss_fn.E_pos},{loss_fn.E_neg - loss_fn.E_pos}\n")
                else:
                    f = open(f"{self.local_path}loss.stats.expectations.csv", "a")
                    f.write(
                        f"{loss_fn.E_neg},{loss_fn.E_pos},{loss_fn.E_neg - loss_fn.E_pos}\n")
                f.close()

                #### Reconstruct JSD MI matrix ####
                # TODO: Improve this code
                E = torch.diag(pos_exp(loss_fn.pos_samples,
                                    average=False).mean(dim=1))
                E[:loss_fn.num_graphs-1, 1:] += torch.triu(
                    neg_exp(loss_fn.neg_samples, average=False)
                    .reshape(loss_fn.num_graphs, -1, loss_fn.num_nodes)
                    .mean(dim=2), 0)[:-1]
                E[1:, :loss_fn.num_graphs-1] += torch.tril(
                    neg_exp(loss_fn.neg_samples, average=False)
                    .reshape(loss_fn.num_graphs, -1, loss_fn.num_nodes)
                    .mean(dim=2), -1)[1:]

                #### Unsupervised accuracy ####
                # pylint: disable=line-too-long
                labels = torch.block_diag(
                    *[torch.ones(loss_fn.num_nodes) for _ in range(loss_fn.num_graphs)])

                #### Prediction ####
                # yhat_norm = F.sigmoid(loss_fn.discr_matr)
                yhat_norm = torch.sigmoid(loss_fn.discr_matr)

                if "sigmoid" in self.cfg["visuals"]:
                    Path(self.local_path + "sigmoid/").mkdir(parents=True, exist_ok=True)
                    plot_heatmap(yhat_norm.detach().numpy()).savefig(
                        self.local_path + f"sigmoid/{self.solver.epoch}.sigmoid.batch{self.solver.batch}.png")

                yhat_norm[yhat_norm > 0.5] = 1
                yhat_norm[yhat_norm <= 0.5] = 0

                if self.solver.batch not in acc:
                    acc[self.solver.batch] = [
                        (yhat_norm == labels).sum() / torch.numel(labels)]
                else:
                    acc[self.solver.batch].append(
                        (yhat_norm == labels).sum() / torch.numel(labels))

                if self.solver.batch not in e_pos:
                    e_pos[self.solver.batch] = [self.solver.loss_fn.E_pos]
                    e_neg[self.solver.batch] = [self.solver.loss_fn.E_neg]
                else:
                    e_pos[self.solver.batch].append(self.solver.loss_fn.E_pos)
                    e_neg[self.solver.batch].append(self.solver.loss_fn.E_neg)

                #### Plot Discriminator values over time ####
                num_graphs = loss_fn.num_graphs
                num_nodes = loss_fn.num_nodes
                args = {
                    "matrices": [
                        loss_fn.discr_matr.reshape(
                            num_graphs, num_graphs, num_nodes)
                        .mean(dim=2).squeeze().detach().numpy(),
                        pos_exp(loss_fn.pos_samples,
                                average=False).detach().numpy(),
                        neg_exp(loss_fn.neg_samples, average=False).reshape(
                            num_graphs, -1, num_nodes)
                        .mean(dim=2).detach().numpy(),
                        E.detach().numpy()
                    ],
                    "xlabels": ["Graphs", "Nodes (Same Graph)", "Nodes (Different Graph)", "Graphs"],
                    "ylabels": ["Graphs", "Graphs", "Graphs", "Graphs"],
                    "cbar_titles": ["Discriminator Score [mean]", "E_pos", "E_neg [mean]", "E [mean]"],
                    "ticks": [(self.cfg["loader"]["batch_size"], self.cfg["loader"]["batch_size"])],
                    "im_args": [
                        {"title": f"Discriminator Score - Epoch: {self.solver.epoch} (Batch: {self.solver.batch})"},
                        {"title": f"Positive Samples - Epoch: {self.solver.epoch} (Batch: {self.solver.batch})"},
                        {"title": f"Negative Samples - Epoch: {self.solver.epoch} (Batch: {self.solver.batch})"},
                        {"title": f"Expectation merged - Epoch: {self.solver.epoch} (Batch: {self.solver.batch})"}
                    ],
                    "loss_cfg": {
                        "data": {
                            "TIG Train Loss (JSD MI)": self.solver.train_batch_losses[self.solver.batch::len(self.solver.train_loader)],
                            r"$\mathbb{E}_{pos}$": e_pos[self.solver.batch],
                            r"$\mathbb{E}_{neg}$": e_neg[self.solver.batch],
                            r"$Loss (plain)$": np.array(e_neg[self.solver.batch]) - np.array(e_pos[self.solver.batch])

                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"],
                        "title": "Loss"
                    },
                    "metric_cfg": {
                        "data": {
                            "Accuracy": acc[self.solver.batch],
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"],
                        "title": "Accuracy",
                        "line_mode": np.max
                    }
                }

                fig = plot_heatmaps(**args)
                plt.close()

                # TODO: Save single intermediate plot to monitor tracking.
                # Path(self.local_path + f"JS-MI/discriminator.batch{self.solver.batch}.png").mkdir(parents=True, exist_ok=True)
                # fig.savefig(self.local_path + f"JS-MI/discriminator.batch{self.solver.batch}.png")

                #### Store plots and/or create gifs ####
                path = self.local_path + \
                    f"JS-MI/discriminator.batch{self.solver.batch}.files"
                if "intermediate_gif" in self.cfg and self.cfg["intermediate_gif"]:
                    create_gif(fig, path=self.local_path +
                            f"JS-MI/discriminator.batch{self.solver.batch}.gif")
                elif self.solver.epoch == self.solver.train_cfg["n_epochs"] - 1:
                    # Last epoch create gif
                    create_gif(path, path=self.local_path +
                            f"JS-MI/discriminator.batch{self.solver.batch}.gif")
                    shutil.rmtree(path)
                else:
                    Path(path).mkdir(parents=True, exist_ok=True)
                    fig.savefig(
                        path+f"/{self.solver.epoch}.discriminator.batch{self.solver.batch}.png")

    def track_epoch(self):
        """ Function that manages the tracking per epoch (called from the solver).
        """
        #### Plots & Animations ####
        if isinstance(self.solver.model, MLP) and self.track_decision:
            if ("classification" in self.cfg["visuals"] or 
                ("classification.last" in self.cfg["visuals"] and
                 self.solver.epoch == self.solver.train_cfg["n_epochs"] - 1)):
                #### Track epoch for MLP ####
                emb_x = np.array(
                    self.solver.train_loader.dataset, dtype=object)[:, 0]
                emb_y = np.array(
                    self.solver.train_loader.dataset, dtype=object)[:, 1]

                if self.solver.encoder is not None:
                    device = self.solver.encoder.device
                    self.solver.encoder = self.solver.encoder.cpu()
                    emb_x = self.solver.encoder(np.array(list(emb_x)).transpose(
                        (0, 3, 2, 1)), KINECT_ADJACENCY)[0]
                    self.solver.encoder = self.solver.encoder.to(device)

                args = {
                    "emb_cfg": {
                        "x": emb_x,
                        "y": emb_y,
                        "clf": self.solver.model,
                        "mode": "PCA"
                    },
                    "loss_cfg": {
                        "data": {
                            "MLP Train Loss": self.solver.train_losses,
                            "MLP Val Loss": self.solver.train_losses,
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    },
                    "metric_cfg": {
                        "data": {
                            "MLP Top-1 Acc.": np.array(self.solver.train_metrics)[:, 0],
                            "MLP Top-5 Acc.": np.array(self.solver.train_metrics)[:, 1],
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    }
                }

                if hasattr(self.solver, "val_metrics") and len(self.solver.val_metrics) != 0:
                    args["metric_cfg"]["data"]["MLP Val Top-1 Acc."] = np.array(
                        self.solver.val_metrics)[:, 0]
                    args["metric_cfg"]["data"]["MLP Val Top-5 Acc."] = np.array(
                        self.solver.val_metrics)[:, 1]

                #### Plot embeddings with loss/metric curve ####
                fig = plot_eval(**args,
                                title=f"MLP Decision Boundary - Epoch: {self.solver.epoch}",
                                model_name="MLP")
                plt.close()

                #### Store plots and/or create gif ####
                path = self.local_path + "MLP.decision.boundaries.gif.files/"
                name = "MLP.decision.boundaries"
                self.save_plot(fig, path, name)

            if "loss" in self.cfg["visuals"] and "metric" in self.cfg["visuals"]:
                args = {
                    "loss_cfg": {
                        "data": {
                            "MLP Train Loss": self.solver.train_losses,
                            "MLP Val Loss": self.solver.val_losses,
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    },
                    "metric_cfg": {
                        "data": {
                            "MLP Top-1 Acc.": np.array(self.solver.train_metrics)[:, 0],
                            "MLP Top-5 Acc.": np.array(self.solver.train_metrics)[:, 1],
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    }
                }
                fig = plot_loss_metric(**args, title="Loss & Metric", model_name="MLP")
                plt.close()
                path = self.local_path+"MLP.loss.metric.gif.files/"
                name = "MLP.loss.metric"
                self.save_plot(fig, path, name)

            else:
                if "loss" in self.cfg["visuals"]:
                    args = {
                        "data": {
                            "MLP Train Loss": self.solver.train_losses,
                            "MLP Val Loss": self.solver.train_losses,
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    }
                    fig = plot_curve(**args, title="Loss")
                    plt.close()
                    path = self.local_path+"MLP.loss.gif.files/"
                    name = "MLP.loss"
                    self.save_plot(fig, path, name)

                if "metric" in self.cfg["visuals"]:
                    args =  {
                            "data": {
                            "MLP Top-1 Acc.": np.array(self.solver.train_metrics)[:, 0],
                            "MLP Top-5 Acc.": np.array(self.solver.train_metrics)[:, 1],
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    }
                    fig = plot_curve(**args, title="Metric")
                    plt.close()
                    path = self.local_path+"MLP.metric.gif.files/"
                    name = "MLP.metric"
                    self.save_plot(fig, path, name)

        if isinstance(self.solver.model, TemporalInfoGraph) and self.track_decision:
            if "pca" in self.cfg["visuals"] or "tsne" in self.cfg["visuals"]:
                #### Track epoch for TIG Encoder ####
                with torch.no_grad():
                    device = self.solver.model.device
                    self.solver.model = self.solver.model.cpu()
                    emb_x = np.array(
                        list(np.array(self.solver.train_loader.dataset, dtype=object)[:, 0]))
                    emb_x = self.solver.model(emb_x.transpose(
                        (0, 3, 2, 1)), KINECT_ADJACENCY)[0]
                    self.solver.model = self.solver.model.to(device)

                emb_y = np.array(
                    self.solver.train_loader.dataset, dtype=object)[:, 1]

                args = {
                    "emb_cfg": {
                        "x": emb_x,
                        "y": emb_y,
                        "mode": "" if emb_x.shape[1] <= 2 else "PCA"
                    },
                    "loss_cfg": {
                        "data": {
                            "TIG Train Loss (JSD MI)": self.solver.train_losses,
                            "TIG Val Loss (JSD MI)": self.solver.train_losses,
                        },
                        "n_epochs": self.solver.train_cfg["n_epochs"]
                    }
                }

            if "pca" in self.cfg["visuals"]:
                #### Plot embeddings with loss/metric curve (PCA or plain) ####
                fig = plot_eval(**args,
                                title=f"TIG Embeddings {args['emb_cfg']['mode']} - Epoch: {self.solver.epoch}",
                                model_name="TIG")
                plt.close()
                path = self.local_path + "TIG.embeddings.pca.gif.files/"
                name = "TIG.embeddings.pca"
                self.save_plot(fig, path, name)

            if "tsne" in self.cfg["visuals"]:
                #### Plot embeddings with loss/metric curve (t-SNE) ####
                args["emb_cfg"]["mode"] = "TSNE"
                fig = plot_eval(**args,
                                title=f"TIG Embeddings (TSNE) - Epoch: {self.solver.epoch}",
                                model_name="TIG")
                plt.close()

                path = self.local_path+"TIG.embeddings.tsne.gif.files/"
                name = "TIG.embeddings.tsne"
                self.save_plot(fig, path, name)

            if "loss" in self.cfg["visuals"]:
                args = {
                    "data": {
                        "TIG Train Loss (JSD MI)": self.solver.train_losses,
                        "TIG Val Loss (JSD MI)": self.solver.val_losses,
                    },
                    "n_epochs": self.solver.train_cfg["n_epochs"]
                }
                fig = plot_curve(**args, title="Loss")
                plt.close()
                path = self.local_path+"TIG.loss.gif.files/"
                name = "TIG.loss"
                self.save_plot(fig, path, name)

        #### Plain values and Python objects ####
        if self.ex is not None:
            #### Remote Tracking ####
            if self.solver.epoch % self.save_nth == 0:
                self.track_checkpoint()

            self.ex.log_scalar(f"{self.tag}loss.epoch.train",
                               self.solver.train_losses[self.solver.epoch], self.solver.epoch)
            if hasattr(self.solver, "train_metric"):
                self.ex.log_scalar(f"{self.tag}train.top1",
                                   self.solver.train_metric[0])
                self.ex.log_scalar(f"{self.tag}train.top5",
                                   self.solver.train_metric[1])

        # TODO: Check if this block is still needed.
        # if self.solver.phase == "validation" and self.ex is not None:
        #     self.ex.log_scalar(f"{self.tag}loss.epoch.val",
        #                        self.solver.val_losses[self.solver.epoch], self.solver.epoch)

        #     if hasattr(self.solver, "val_metric"):
        #         self.ex.log_scalar(f"{self.tag}val.top1",
        #                            self.solver.val_metric[0])
        #         self.ex.log_scalar(f"{self.tag}val.top5",
        #                            self.solver.val_metric[1])

    def track_train(self):
        """ Function that manage the tracking per trainings batch (called from the solver).
        """
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None:
            self.ex.log_scalar(f"{self.tag}.loss.batch.train.{self.solver.epoch}",
                               self.solver.train_batch_losses[self.solver.batch], self.solver.batch)

    def track_validation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        # pylint: disable=no-member
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None:
            self.ex.log_scalar(f"{self.tag}.loss.batch.val.{self.solver.epoch}",
                               self.solver.val_batch_losses[self.solver.batch], self.solver.batch)

    def track_evaluation(self):
        """ Function that manage the tracking per validation batch (called from the solver).
        """
        # pylint: disable=no-member
        if hasattr(self, "intermediate_tracking") and not self.intermediate_tracking:
            return
        if self.ex is not None:
            self.ex.log_scalar(f"{self.tag}test.top1", self.solver.metric[0])
            self.ex.log_scalar(f"{self.tag}test.top5", self.solver.metric[1])

    def track_traning(self, train: callable):
        """ Function that initiate the the tracking of the training.

            Paramter:
                train: callable
                    Trainings function that will be executed.
        """
        def inner(cfg, encoder=None):
            # Extract solver
            # pylint: disable=attribute-defined-outside-init
            self.solver = train.__self__

            #### Pre Training ####
            cfg["train_size"] = '{:.2f}GB'.format(sys.getsizeof(self.solver.train_loader.dataset) / 1024**3)
            cfg["val_size"] = '{:.2f}GB'.format(sys.getsizeof(self.solver.val_loader.dataset) / 1024**3)
            try:
                cfg["train_length"] = len(self.solver.train_loader.dataset)
                cfg["val_length"] = len(self.solver.val_loader.dataset)
            except:
                pass

            Path(self.local_path).mkdir(parents=True, exist_ok=True)

            with open(f'{self.local_path}/config.json', 'w') as fp:
                json.dump(cfg, fp)

            train(cfg, track=self.track, encoder=encoder)

            #### After the training is done ####
            if self.ex is not None:
                #### LOGGING ####
                self.log_config(f"{self.tag}optimzer",
                                str(self.solver.optimizer))
                self.log_config(f"{self.tag}train_cfg",
                                str(self.solver.train_cfg))
                self.log_config(f"{self.tag}train_size", str(
                    len(self.solver.train_loader.dataset)))
                self.log_config(f"{self.tag}train_batch_size", str(
                    self.solver.train_loader.batch_size))
                self.log_config(f"{self.tag}model", str(self.solver.model))

                buffer = io.BytesIO()
                torch.save(self.solver.model, buffer)

                self.add_artifact(buffer.getvalue(), name=f"{self.ex_name}.pt")

            if self.solver.phase == "validation":
                self.log_config(f"{self.tag}val_size", str(
                    len(self.solver.val_loader.dataset)))
                self.log_config(f"{self.tag}val_batch_size", str(
                    self.solver.val_loader.batch_size))

            self.track_locally()

        return inner

    def track_testing(self, test: callable):
        """ Function that initiate the the tracking of the testing.

            Paramter:
                test: callable
                    Testing function that will be executed.
        """
        def inner(*args):
            # Extract solver
            # pylint: disable=attribute-defined-outside-init
            self.solver = test.__self__
            test(*args, track=self.track)

            if self.ex is not None:
                if self.solver.test_loader is not None:
                    self.log_config(f"{self.tag}test_cfg",
                                    str(self.solver.test_cfg))
                    self.log_config(f"{self.tag}test_size", str(
                        np.array(self.solver.test_loader.dataset).shape))
                    self.log_config(f"{self.tag}test_batch_size", str(
                        self.solver.test_loader.batch_size))

                self.log_config(f"{self.tag}model", str(self.solver.model))

        return inner

    def track_locally(self):
        """ Function to track everything after the training/testing is done locally.
        """
        Path(self.local_path).mkdir(parents=True, exist_ok=True)

        if hasattr(self, "solver"):
            with open(f'{self.local_path}/config.json', 'w') as fp:
                json.dump({
                    **self.cfg,
                    **{
                        "val_data_size": len(self.solver.val_loader.dataset) if hasattr(self.solver, "val_loader") else "-",
                        "train_data_size": len(self.solver.train_loader.dataset),
                        "exec_dir": os.getcwd(),
                        "optimzer": str(self.solver.optimizer),
                    }}, fp)

            np.save(f'{self.local_path}/TIG_{self.tag}train_losses.npy',
                    self.solver.train_losses)

            if hasattr(self.solver, "train_metric"):
                np.save(f"{self.local_path}/TIG_{self.tag}train.metrics.npy",
                        self.solver.train_metrics)

            np.save(f'{self.local_path}/TIG_{self.tag}val_losses.npy',
                    self.solver.val_losses)

            if hasattr(self.solver, "val_metric"):
                np.save(f"{self.local_path}/TIG_{self.tag}val.metrics.npy",
                        self.solver.val_metrics)

            #### Test / Evaluation Metrics ####
            if hasattr(self.solver, "metric"):
                np.save(f'{self.local_path}/TIG_{self.tag}top1.npy',
                        self.solver.metric[0])
                np.save(f'{self.local_path}/TIG_{self.tag}top5.npy',
                        self.solver.metric[1])

            torch.save(
                self.model, f'{self.local_path}/TIG_{self.tag.replace(".", "")}.pt')
            log.info(f"Experiment stored at '{self.local_path}")

    def track_checkpoint(self):
        """ Function to track and manage checkpoints of the training.
        """
        self.checkpoint_dict = {
            **self.checkpoint_dict,
            **{
                'epoch': self.solver.epoch,
                'model_state_dict': self.solver.model.state_dict(),
                'optimizer_state_dict': self.solver.optimizer.state_dict(),
                'loss': self.solver.val_losses[-1] if len(self.solver.val_losses) > 0 else "-",
            }
        }

        #### Remote ####
        buffer = io.BytesIO()
        torch.save(self.checkpoint, buffer)

        self.add_artifact(buffer.getvalue(), name="checkpoint.pt")

        #### Local Checkpoint ####
        path = self.local_path + "checkpoints/"
        Path(path).mkdir(parents=True, exist_ok=True)

        torch.save(self.checkpoint_dict, f"{path}checkpoint.pt")

        log.info("Checkpoint stored")

    def save_plot(self, fig, gif_path, name):
        #### Save Plots ####
        if self.solver.epoch == 0:
            fig.savefig(self.local_path + f"{name}.initial.png")
            fig.savefig(self.local_path + f"{name}.final.png")
        else:
            fig.savefig(self.local_path+f"{name}.final.png")

        if gif_path is not None:
            if "intermediate_gif" in self.cfg and self.cfg["intermediate_gif"]:
                create_gif(fig, path=self.local_path + f"{name}.gif")
            elif self.solver.epoch == self.solver.train_cfg["n_epochs"] - 1:
                # Last epoch create gif
                fig.savefig( gif_path+f"/{self.solver.epoch}.{name}.png")
                create_gif(gif_path, path=self.local_path + f"{name}.gif")
                shutil.rmtree(gif_path) # Remove single frames
            else:
                Path(gif_path).mkdir(parents=True, exist_ok=True)
                fig.savefig( gif_path+f"/{self.solver.epoch}.{name}.png")

    ##############################################
    # Custom Tracking Function (Locally/Remotly) #
    ##############################################

    def log_tensor(self, array: np.array, name: str = "tensor"):
        """ Function to log/track a tensor.
        """
        Path(self.local_path).mkdir(parents=True, exist_ok=True)
        np.save(f"{self.local_path}/{name}.npy", array)

    def log_config(self, name: str, cfg: object):
        """ Manually track configuration to data base.

            Paramters:
                name: str
                    Name of the configuration that will be tracked.
                cfg: object
                    The configuration that should be stored in the DB.
                    For instance, a dictionary or a simple string.
        """
        if self.ex is None:
            return

        if self.id is None:
            raise ValueError("Experiment ID is not set!")

        self.observer.run_entry["config"][name] = cfg
        self.observer.save()

    def add_artifact(self, file: bytes, name: str):
        """ Tracks an object.

            Paramters:
                file: bytes
                    File that should be tracked as artifact.
                name: str
                    Name of the artifact.
        """
        if self.ex is None:
            return

        db_filename = "artifact://{}/{}/{}".format(
            self.observer.runs.name, self.id, name)

        result = self.db["fs.files"].find_one(
            {"name": self.observer.fs.delete(db_filename)})
        if "_id" in result:
            self.observer.fs.delete(result['_id'])

        file_id = self.observer.fs.put(file, filename=db_filename)

        self.observer.run_entry["artifacts"].append(
            {"name": name, "file_id": file_id})
        self.observer.save()

    def load_model(self, run_id: int, name: str, db_url: Any = None, mode: str = "torch"):
        """ Function to load a model from the tracking server.

            You can either use the configuration (e.g. DB connection) from the tracker,
            if it is initialized or you provide the DB URL to create a new connection
            to the DB.

            Paramters:
                run_id: int
                    ID of the run from which you want to load the model.
                name: str
                    File name of the model
                db_url: str
                    DB url that is used to establish a new connection to the DB.
                mode: str
                    To determine if the loaded model is returned as PyTorch object or 
                    as a plain bytes object.
            Return:
                torch.nn.Module or bytes object.
        """

        if hasattr(self, "observer") and self.observer is not None and db_url is None:
            runs = self.run_collection
            fs = self.observer.fs
        else:
            client = pymongo.MongoClient(db_url)
            fs = gridfs.GridFS(client.temporal_info_graph)
            runs = client.temporal_info_graph.runs

        run_entry = runs.find_one(
            {'_id': run_id, "artifacts": {"$elemMatch": {"name": name}}})

        if mode == "torch":
            buff = fs.get(run_entry['artifacts'][0]['file_id']).read()
            buff = io.BytesIO(buff)
            buff.seek(0)
            return torch.load(buff)
        else:
            return fs.get(run_entry['artifacts'][0]['file_id']).read()
