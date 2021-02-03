"""
    @author: jhuthmacher
"""
from typing import Callable

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm, trange
import numpy as np
from sklearn import metrics

# pylint: disable=import-error
from visualization import create_gif, plot_emb_pred
from data import KINECT_ADJACENCY
from model.loss import jensen_shannon_mi, bce_loss

#### Local/Tracking Config ####
# pylint: disable=import-error
import config.config as cfg
from config.config import log

from sacred import Experiment

# pylint: disable=too-many-instance-attributes


class Solver():
    """ The Solver class implements several functions to provide a common training
        and testing procedure.
    """

    #pylint: disable=dangerous-default-value
    def __init__(self, model: nn.Module, dataloader: [DataLoader],
                 loss_fn: Callable[[torch.Tensor,
                                    torch.Tensor], torch.Tensor] = None,
                 train_cfg: dict = {}, test_cfg: dict = {}):
        """ Initialization of the Solver.

            Paramters:
                model: torch.nn.Module
                    Model that should be optimized.
                dataloader: [DataLoader]
                    List of data loaders for training, validation and testing.
                    We distinguish by the length of the list:
                    1 element:  dataloader[0] = train_loader
                    2 elements: dataloader[0] = train_loader
                                dataloader[1] = validation_loader
                    3 elements: dataloader[0] = train_loader
                                dataloader[1] = validation_loader
                                dataloader[2] = test_loader
                    Optimzer used to train the model.
                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
                    Function calculating the loss of the model.
        """
        #### Train configuration ####
        self.train_cfg = {
            "n_epochs": 10,
            "log_nth": 0,
            "optimizer_name": "Adam",
            "optimizer": {
                "lr": 1e-3,
                "weight_decay": 1e-3
            },
            "verbose": False
        }
        self.train_cfg = {**self.train_cfg, **train_cfg}

        #### Test configuration ####
        self.test_cfg = {
            "n_batches": 1,
        }
        self.test_cfg = {**self.test_cfg, **test_cfg}

        #### load check point #####
        # TODO: Check if this is working.
        if isinstance(model, tuple):
            # Model is a checkpoint
            checkpoint = model[0]
            modelClass = model[1]
            optimClass = model[2] if len(model) > 2 else optim.Adam

            self.model = modelClass(**checkpoint['model_params'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

            optim_params = (self.train_cfg["optimizer"]
                            if checkpoint['optim_params'] == {}
                            else checkpoint['optim_params'])
            self.optimizer = optimClass(
                self.model.parameters(), **optim_params)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.epoch = checkpoint['epoch']
        else:
            self.model = model

        self.loss_fn = jensen_shannon_mi if loss_fn is None else loss_fn

        if dataloader is not None:
            # Can be None for testing
            self.train_loader = dataloader[0]
            self.val_loader = dataloader[1] if len(
                dataloader) == 2 or len(dataloader) == 3 else None
            self.test_loader = dataloader[2] if len(dataloader) == 3 else None

        # Object storage to track the performance
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []  # TODO: Useless?

        self.val_metrics = {}  # np.array([])
        self.train_metrics = {}  # np.array([])
        self.predictions = np.array([])
        self.labels = np.array([])

        self.val_pred = np.array([])
        self.val_label = np.array([])
        self.train_pred = np.array([])
        self.train_label = np.array([])

        self.epoch = 0

    def train_step(self, batch_x, batch_y, encoder=None):
        """ Train step function to reduce memory footprint.
        """
        # The train loader returns dim (batch_size, frames, nodes, features)
        # TODO: Check if it is needed
        try:
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1)
            else:
                batch_x = torch.tensor(
                    batch_x, dtype=torch.long).permute(0, 3, 2, 1)
        except:  # pylint: disable=bare-except
            # For testing MLP on iris
            if not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x, dtype=torch.float32)

        self.optimizer.zero_grad()

        if encoder is not None:
            #### Downstream Task ####
            self.yhat, _ = encoder.to(self.model.device)(batch_x)
            self.yhat = self.model(self.yhat)
        else:
            #### TIG Prediction ####
            self.yhat = self.model(batch_x)

        if isinstance(self.yhat, tuple):
            loss = self.loss_fn(*self.yhat)
        else:
            if not isinstance(batch_y, torch.Tensor):
                batch_y = torch.tensor(
                    batch_y, dtype=torch.long).to(self.model.device)
            else:
                batch_y = batch_y.type(
                    "torch.LongTensor").to(self.model.device)

            loss = self.loss_fn(self.yhat, batch_y)

            #### EVALUATION DURING TRAINING ####
            self.yhat_idx = torch.argsort(self.yhat, descending=True)
            self.train_pred = np.vstack([self.train_pred, self.yhat_idx.detach().cpu(
            ).numpy()]) if self.train_pred.size else self.yhat_idx.detach().cpu().numpy()
            self.train_label = np.append(
                self.train_label, batch_y.detach().cpu().numpy())

        if loss.isnan():
            # Loss can be nan when the batch only contains a single sample!
            return

        self.train_batch_losses.append(torch.squeeze(loss).item())
        loss.backward()
        if "gradient_clipping" in self.train_cfg and (type(self.train_cfg["gradient_clipping"]) == int or float):
            clip_grad_norm_(self.model.parameters(),
                            self.train_cfg["gradient_clipping"])
        self.optimizer.step()

    def val_step(self, batch_x, batch_y, encoder=None):
        """ Validation step function to reduce memory footprint.
        """

        # The train loader returns dim (batch_size, frames, nodes, features)
        # TODO: Check if it is needed
        try:
            if isinstance(batch_x, torch.Tensor):
                batch_x = batch_x.type("torch.FloatTensor").permute(0, 3, 2, 1)
            else:
                batch_x = torch.tensor(
                    batch_x, dtype=torch.float32).permute(0, 3, 2, 1)
        except:
            # For testing MLP on iris
            if not isinstance(batch_x, torch.Tensor):
                batch_x = torch.tensor(batch_x, dtype=torch.float32)

        if encoder is not None:
            #### Downstream Task ####
            self.yhat, _ = encoder(batch_x)
            self.yhat = self.model(self.yhat)
        else:
            #### TIG Prediction ####
            self.yhat = self.model(batch_x)

        if isinstance(self.yhat, tuple):
            loss = self.loss_fn(*self.yhat)
        else:
            if not isinstance(batch_y, torch.Tensor):
                batch_y = torch.tensor(
                    batch_y, dtype=torch.float32).to(self.model.device)
            else:
                batch_y = batch_y.type(
                    "torch.LongTensor").to(self.model.device)

            loss = self.loss_fn(self.yhat, batch_y)

            #### EVALUATION DURING VALIDATION ####
            self.yhat_idx = torch.argsort(self.yhat, descending=True)
            self.val_pred = np.vstack([self.val_pred, self.yhat_idx.detach().cpu(
            ).numpy()]) if self.val_pred.size else self.yhat_idx.detach().cpu().numpy()
            self.val_label = np.append(
                self.val_label, batch_y.detach().cpu().numpy())

        if loss.isnan():
            # Loss can be nan when the batch only contains a single sample!
            return
        self.val_batch_losses.append(torch.squeeze(loss).item())

        # if callable(track):
        #     track("validation")

        # # lr_scheduler.step()

    def train(self, train_config: dict = None, track: object = None,
              optimizer: torch.optim.Optimizer = None, encoder: nn.Module = None):
        """ Training procedure.

            Paramters:
                train_config: dict
                    Dictionary to adapt the trainings configuration. If nothing is provided
                    the config defined in the initialization is used.
                track: Tracker (optional)
                    Tracker object that takes care about the tracking.
                optimizer: torch.optim.Optimizer (optional)
                    PyTorch optimizer instance that is used for the training. If nothing is
                    provided the defined optimizer from the initialization is used.
                encoder: nn.Module
                    PyTorch model that encodes the data. Needed when the downstream model is
                    trained.
        """

        if train_config is not None:
            # Merges custom config with default config!
            self.train_cfg = {**self.train_cfg, **train_config}

        self.encoder = encoder
        self.optimizer = (getattr(optim,  self.train_cfg["optimizer_name"])(
            self.model.parameters(),
            **self.train_cfg["optimizer"],
        )
            if optimizer is None
            else optimizer)

        #### Learning Rate Decay #####
        # decayRate = 0.95
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)

        self.train_batch_losses = []

        #### Training & Validation ####
        start_epoch = self.epoch
        pbar = trange(start_epoch, self.train_cfg["n_epochs"],
                      disable=(not self.train_cfg["verbose"]),
                      desc=f'Epochs ({self.model.__class__.__name__})')
        for self.epoch in pbar:
            # We store all losses and use a mask to access the right loss per epoch.
            # self.train_batch_losses = []
            self.val_batch_losses = []

            #### Training ####
            self.model.train()
            train_batch_metric = {}
            self.phase = "train"
            for self.batch, (batch_x, self.batch_y) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), leave=False,
                                                                      disable=False, desc=f'Trai. Batch (Epoch: {self.epoch})')):
                self.train_step(batch_x, self.batch_y, encoder)

                #### Track Evaluation ####
                if isinstance(self.yhat, tuple):
                    yhat_norm = torch.sigmoid(
                        self.loss_fn.discr_matr).detach().numpy()
                    yhat_norm[yhat_norm > 0.5] = 1
                    yhat_norm[yhat_norm <= 0.5] = 0

                    evaluate(
                        yhat_norm, self.loss_fn.mask.detach().numpy(), mode="accuracy")

                    acc = evaluate(
                        yhat_norm, self.loss_fn.mask.detach().numpy(), mode="accuracy")
                    prec = evaluate(
                        yhat_norm, self.loss_fn.mask.detach().numpy(), mode="precision")
                    auc = evaluate(
                        yhat_norm, self.loss_fn.mask.detach().numpy(), mode="auc")

                    train_batch_metric["accuracy"] = (train_batch_metric["accuracy"] + [acc]
                                                      if "accuracy" in train_batch_metric
                                                      else [acc])
                    train_batch_metric["precision"] = (train_batch_metric["precision"] + [prec]
                                                       if "precision" in train_batch_metric
                                                       else [prec])
                    train_batch_metric["auc"] = (train_batch_metric["auc"] + [auc]
                                                 if "auc" in train_batch_metric
                                                 else [auc])
                else:
                    top_k = evaluate(self.train_pred, self.train_label)
                    train_batch_metric["top-1"] = (train_batch_metric["top-1"] + [top_k[0]]
                                                   if "top-1" in train_batch_metric
                                                   else [top_k[0]])
                    train_batch_metric["top-5"] = (train_batch_metric["top-5"] + [top_k[1]]
                                                   if "top-5" in train_batch_metric
                                                   else [top_k[1]])
                    self.train_pred = np.array([])
                    self.train_label = np.array([])

                if encoder is None and callable(track):
                    track("train_step")

                # if callable(track):
                #     track("training")

            for key in train_batch_metric:
                metric = np.mean(train_batch_metric[key])
                self.train_metrics[key] = (self.train_metrics[key] + [metric]
                                           if key in self.train_metrics
                                           else [metric])
            train_batch_metric = {}

            self.phase = "validation"
            if self.val_loader is not None:
                # For disabling the gradient calculation -> Performance advantages
                with torch.no_grad():
                    self.val_pred = np.array([])
                    self.val_label = np.array([])
                    val_batch_metric = {}
                    #### Validate ####
                    # self.model.eval()
                    for self.batch, (batch_x, self.batch_y) in enumerate(tqdm(self.val_loader, disable=False, leave=False,
                                                                              desc=f'Vali. Batch (Epoch: {self.epoch})')):
                        self.val_step(batch_x, self.batch_y, encoder)

                        if isinstance(self.yhat, tuple):
                            yhat_norm = torch.sigmoid(
                                self.loss_fn.discr_matr).detach().numpy()
                            yhat_norm[yhat_norm > 0.5] = 1
                            yhat_norm[yhat_norm <= 0.5] = 0

                            evaluate(
                                yhat_norm, self.loss_fn.mask.detach().numpy(), mode="accuracy")

                            acc = evaluate(
                                yhat_norm, self.loss_fn.mask.detach().numpy(), mode="accuracy")
                            prec = evaluate(
                                yhat_norm, self.loss_fn.mask.detach().numpy(), mode="precision")
                            auc = evaluate(
                                yhat_norm, self.loss_fn.mask.detach().numpy(), mode="auc")

                            val_batch_metric["accuracy"] = (val_batch_metric["accuracy"] + [acc]
                                                            if "accuracy" in val_batch_metric
                                                            else [acc])
                            val_batch_metric["precision"] = (val_batch_metric["precision"] + [prec]
                                                             if "precision" in val_batch_metric
                                                             else [prec])
                            val_batch_metric["auc"] = (val_batch_metric["auc"] + [auc]
                                                       if "auc" in val_batch_metric
                                                       else [auc])
                        else:
                            top_k = evaluate(self.val_pred, self.val_label)
                            val_batch_metric["top-1"] = (val_batch_metric["top-1"] + [top_k[0]]
                                                           if "top-1" in val_batch_metric
                                                           else [top_k[0]])
                            val_batch_metric["top-5"] = (val_batch_metric["top-5"] + [top_k[1]]
                                                           if "top-5" in val_batch_metric
                                                           else [top_k[1]])
                            self.val_pred = np.array([])
                            self.val_label = np.array([])

                    for key in val_batch_metric:                        
                        metric = np.mean(val_batch_metric[key])
                        key = f"val. {key}"
                        self.val_metrics[key] = (self.val_metrics[key] + [metric]
                                                   if key in self.val_metrics
                                                   else [metric])
                    val_batch_metric = {}

                    # TODO: CHeck this
                    self.val_losses.append(np.mean(self.val_batch_losses) if len(
                        self.val_batch_losses) > 0 else 0)
                    self.val_batch_losses = []

            idx = self.epoch * len(self.train_loader)
            self.train_losses.append(np.mean(self.train_batch_losses[idx:idx+len(
                self.train_loader)]) if len(self.train_batch_losses) > 0 else 0)
            # self.train_batch_losses = []
            if ("val. accuracy" in self.val_metrics and
                "val. precision" in self.val_metrics and
                    "val. auc" in self.val_metrics):
                pbar.set_description(f'Epochs ({self.model.__class__.__name__})' +
                                     f'(Val (max) - acc: {"%.2f"%np.max(self.val_metrics["val. accuracy"])},' +
                                     f'prec: {"%.2f"%np.max(self.val_metrics["val. precision"])},' +
                                     f'auc: {"%.2f"%np.max(self.val_metrics["val. auc"])})')

            if callable(track):
                track("epoch")

    def test(self, test_config: dict = None, track: object = None, encoder: nn.Module = None):
        """ Function to test the model.

            Paramters:
                test_config: dict
                    Dictionary to adapt the test configuration. If nothing is provided
                    the config defined in the initialization is used.
                track: Tracker (optional)
                    Tracker object that takes care about the tracking.
                    PyTorch optimizer instance that is used for the training. If nothing is
                    provided the defined optimizer from the initialization is used.
                encoder: nn.Module
                    PyTorch model that encodes the data. Needed when the downstream model is tested.
            Return:
                tuple: tuple with test metrics. 
        """
        # Not used yet
        if test_config is not None:
            # Merges custom config with default config!
            self.test_cfg = {**self.test_cfg, **test_config}

        self.predictions = np.array([])
        self.labels = np.array([])

        with torch.no_grad():

            #### Test ####
            self.model.eval()
            for self.batch, (batch_x, self.batch_y) in enumerate(tqdm(self.test_loader, disable=False, leave=False,
                                                                      desc=f'Test. Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                try:
                    if isinstance(batch_x, torch.Tensor):
                        batch_x = batch_x.type(
                            "torch.FloatTensor").permute(0, 3, 2, 1)
                    else:
                        batch_x = torch.tensor(
                            batch_x, dtype=torch.float32).permute(0, 3, 2, 1)
                except:
                    # For testing MLP on iris
                    batch_x = torch.tensor(batch_x, dtype=torch.float32)

                if encoder is not None:
                    self.yhat, _ = self.model(batch_x)
                    self.yhat = self.model(self.yhat)
                else:
                    self.yhat = self.model(batch_x)

                if isinstance(self.yhat, tuple):
                    # loss = self.loss_fn(*self.yhat)
                    # TODO: "Unsupervised accuracy"
                    pass
                else:
                    self.yhat_idx = torch.argsort(self.yhat, descending=True)

                    self.predictions = np.vstack([self.predictions, self.yhat_idx.detach().cpu(
                    ).numpy()]) if self.predictions.size else self.yhat_idx.detach().cpu().numpy()
                    self.labels = np.append(
                        self.labels, self.batch_y.detach().cpu().numpy())

            self.metric = evaluate(self.predictions, self.labels)

            if callable(track):
                track("evaluation")

            return self.metric

    def __repr__(self):
        """ String representation of the solver.
        """
        representation = f"Solver (model: {self.model.__class__.__name__})\n"
        representation += f"# Model Parameters: {self.model.paramters}\n"

        if len(self.train_losses) == 0:
            # Not trained yet!
            representation += "+-- Not trained yet!"
        else:
            representation += f"+-- Trainings performance (mean): {np.mean(self.train_losses)}\n"
            representation += f"+-- Validation performance (mean): {np.mean(self.val_losses)}\n"
            representation += f"+-- Num. of epochs: {self.train_cfg['n_epochs']}"

        return representation


def evaluate(predictions: np.array, labels: np.array, mode: str = "top-k"):
    """ Function to calculate the evaluation metric.
            Parameters:
                predictions: np.array
                    Array with the predicted values.
                labels: np.array
                    Corresponding ground truth for the predicted values.
                mode: str (not used yet)
                    Mode to decide with evaluation should be used.
            Return:
                tuple: tuple containing the evaluation metrics.
        """
    if mode == "top-k":
        # Here predictions need to be sorted!
        k = 1  # Top-k
        correct = np.sum([l in pred for l, pred in zip(
            labels, np.asarray(predictions)[:, :k])])
        top1 = (correct/len(labels))

        k = 5  # Top-k
        correct = np.sum([l in pred for l, pred in zip(
            labels, np.asarray(predictions)[:, :k])])
        top5 = (correct/len(labels))

        metric = (top1, top5)
    elif mode == "accuracy":
        metric = (predictions.flatten() == labels.flatten()
                  ).sum() / len(labels.flatten())
    elif mode == "precision":
        # Works only for
        TP = ((predictions.flatten() == 1) & (labels.flatten() == 1)).sum()
        FP = ((predictions.flatten() == 1) & (labels.flatten() == 0)).sum()
        metric = TP / (TP+FP) if TP != 0 else 0
    elif mode == "recall":
        # Works only for
        TP = ((predictions == 1) & (labels == 1)).sum()
        FN = ((predictions == 0) & (labels == 0)).sum()
        metric = TP / (TP+FN) if TP != 0 else 0
    elif mode == "auc":
        fpr, tpr, thresholds = metrics.roc_curve(
            labels.flatten(), predictions.flatten())
        metric = metrics.auc(fpr, tpr)

    # accuracy
    return metric
