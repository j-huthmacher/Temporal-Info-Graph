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

# pylint: disable=import-error
from visualization import create_gif, plot_emb_pred
from data import KINECT_ADJACENCY
from model.loss import jensen_shannon_mi

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
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
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
            self.optimizer = optimClass(self.model.parameters(), **optim_params)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.epoch = checkpoint['epoch']
        else:
            self.model = model

        self.loss_fn = jensen_shannon_mi if loss_fn is None else loss_fn

        self.train_loader = dataloader[0]
        self.val_loader = dataloader[1] if len(dataloader) == 2 or len(dataloader) == 3 else None
        self.test_loader = dataloader[2] if len(dataloader) == 3 else None

        # Object storage to track the performance
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []  # TODO: Useless?

        self.val_metrics = []
        self.train_metrics = []
        self.predictions = np.array([])
        self.labels = np.array([])

        self.val_pred = np.array([])
        self.val_label = np.array([])
        self.train_pred = np.array([])
        self.train_label = np.array([])

        self.epoch = 0

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
        for self.epoch in trange(start_epoch, self.train_cfg["n_epochs"],
                                 disable=(not self.train_cfg["verbose"]),
                                 desc=f'Epochs ({self.model.__class__.__name__})'):
            # We store all losses and use a mask to access the right loss per epoch.
            # self.train_batch_losses = []
            self.val_batch_losses = []

            #### Training ####
            self.model.train()
            self.phase = "train"
            for self.batch, (self.batch_x, self.batch_y) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), leave=False,
                                                                      disable=False, desc=f'Trai. Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                # TODO: Check if it is needed
                try:
                    if isinstance(self.batch_x, torch.Tensor):
                        self.batch_x = self.batch_x.type("torch.FloatTensor").permute(0,3,2,1)
                    else:
                        self.batch_x = torch.tensor(self.batch_x, dtype=torch.long).permute(0,3,2,1)
                except:  #pylint: disable=bare-except
                    # For testing MLP on iris
                    if not isinstance(self.batch_x, torch.Tensor):
                        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)            
                
                self.optimizer.zero_grad()

                if encoder is not None:
                    #### Downstream Task ####
                    self.yhat, _ = encoder.to(self.model.device)(self.batch_x, torch.tensor(KINECT_ADJACENCY))
                    self.yhat = self.model(self.yhat)
                else:
                    #### TIG Prediction ####
                    self.yhat = self.model(self.batch_x, torch.tensor(KINECT_ADJACENCY))
                
                if isinstance(self.yhat, tuple):
                    loss = self.loss_fn(*self.yhat)
                else:
                    if not isinstance(self.batch_y, torch.Tensor):
                        self.batch_y = torch.tensor(self.batch_y, dtype=torch.long).to(self.model.device)
                    else:
                        self.batch_y = self.batch_y.type("torch.LongTensor").to(self.model.device)

                    loss = self.loss_fn(self.yhat, self.batch_y)

                    #### EVALUATION DURING TRAINING ####
                    self.yhat_idx = torch.argsort(self.yhat, descending=True)
                    self.train_pred = np.vstack([self.train_pred, self.yhat_idx.detach().cpu().numpy()]) if self.train_pred.size else self.yhat_idx.detach().cpu().numpy()
                    self.train_label = np.append(self.train_label, self.batch_y.detach().cpu().numpy())

                self.train_batch_losses.append(torch.squeeze(loss).item())
                loss.backward()
                if "gradient_clipping" in train_config and (type(train_config["gradient_clipping"]) == int or float):
                    clip_grad_norm_(self.model.parameters(), train_config["gradient_clipping"])
                self.optimizer.step()

                if callable(track):
                    track("loss")

                # if callable(track):
                #     track("training")

            if not isinstance(self.yhat, tuple):
                self.train_metric = self.evaluate(self.train_pred, self.train_label)
                self.train_metrics.append(self.train_metric)
                self.train_pred = np.array([])
                self.train_label = np.array([])

            self.phase = "validation"
            if self.val_loader is not None:
                # For disabling the gradient calculation -> Performance advantages
                with torch.no_grad():
                    self.val_pred = np.array([])
                    self.val_label = np.array([])

                    #### Validate ####
                    self.model.eval()
                    del self.batch_x
                    del self.batch_y
                    for self.batch, (self.batch_x, self.batch_y) in enumerate(tqdm(self.val_loader, disable=False, leave=False,
                                                                              desc=f'Vali. Batch (Epoch: {self.epoch})')):
                        # The train loader returns dim (batch_size, frames, nodes, features)
                        # TODO: Check if it is needed
                        try:
                            if isinstance(self.batch_x, torch.Tensor):
                                self.batch_x = self.batch_x.type("torch.FloatTensor").permute(0,3,2,1)
                            else:
                                self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32).permute(0,3,2,1)
                        except:
                            # For testing MLP on iris
                            if not isinstance(self.batch_x, torch.Tensor):
                                self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)
                            
                        
                        if encoder is not None:
                            #### Downstream Task ####
                            self.yhat, _ = encoder(self.batch_x, torch.tensor(KINECT_ADJACENCY))
                            self.yhat = self.model(self.yhat)
                        else:
                            #### TIG Prediction ####
                            self.yhat = self.model(self.batch_x, torch.tensor(KINECT_ADJACENCY))

                        if isinstance(self.yhat, tuple):
                            loss = self.loss_fn(*self.yhat)
                        else:
                            if not isinstance(self.batch_y, torch.Tensor):
                                self.batch_y = torch.tensor(self.batch_y, dtype=torch.float32).to(self.model.device)
                            else:
                                self.batch_y = self.batch_y.type("torch.LongTensor").to(self.model.device)

                            loss = self.loss_fn(self.yhat, self.batch_y)

                            #### EVALUATION DURING VALIDATION ####
                            self.yhat_idx = torch.argsort(self.yhat, descending=True)
                            self.val_pred = np.vstack([self.val_pred, self.yhat_idx.detach().cpu().numpy()]) if self.val_pred.size else self.yhat_idx.detach().cpu().numpy()
                            self.val_label = np.append(self.val_label, self.batch_y.detach().cpu().numpy())

                        self.val_batch_losses.append(torch.squeeze(loss).item())

                    if not isinstance(self.yhat, tuple):
                        self.val_metric = self.evaluate(self.val_pred, self.val_label)
                        self.val_metrics.append(self.val_metric)
                        self.val_pred = np.array([])
                        self.val_label = np.array([])

                        # if callable(track):
                        #     track("validation")

            # # lr_scheduler.step()            
                self.val_losses.append(np.mean(self.val_batch_losses) if len(self.val_batch_losses) > 0 else 0)
                self.val_batch_losses = []
            
            idx = self.epoch * len(self.train_loader)
            self.train_losses.append(np.mean(self.train_batch_losses[idx:idx+len(self.train_loader)]) if len(self.train_batch_losses) > 0 else 0)
            # self.train_batch_losses = []

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
            for self.batch, (self.batch_x, self.batch_y) in enumerate(tqdm(self.test_loader, disable=False, leave=False,
                                                                      desc=f'Test. Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                try:
                    if isinstance(self.batch_x, torch.Tensor):
                        self.batch_x = self.batch_x.type("torch.FloatTensor").permute(0,3,2,1)
                    else:
                        self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32).permute(0,3,2,1)
                except:
                    # For testing MLP on iris
                    self.batch_x = torch.tensor(self.batch_x, dtype=torch.float32)

                if encoder is not None:
                    self.yhat, _ = self.model(self.batch_x, torch.tensor(KINECT_ADJACENCY))
                    self.yhat = self.model(self.yhat)
                else:
                    self.yhat = self.model(self.batch_x, torch.tensor(KINECT_ADJACENCY))

                if isinstance(self.yhat, tuple):
                    # loss = self.loss_fn(*self.yhat)
                    # TODO: "Unsupervised accuracy"
                    pass
                else:
                    self.yhat_idx = torch.argsort(self.yhat, descending=True)

                    self.predictions = np.vstack([self.predictions, self.yhat_idx.detach().cpu().numpy()]) if self.predictions.size else self.yhat_idx.detach().cpu().numpy()
                    self.labels = np.append(self.labels, self.batch_y.detach().cpu().numpy())

            self.metric = self.evaluate(self.predictions, self.labels)

            if callable(track):
                track("evaluation")

            return self.metric
    
    def evaluate(self, predictions: np.array, labels: np.array, mode: str = "top-k"):
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

        k = 1  # Top-k
        correct = np.sum([l in pred for l, pred in zip(labels, np.asarray(predictions)[:,:k])])
        top1 = (correct/len(labels))

        k = 5  # Top-k
        correct = np.sum([l in pred for l, pred in zip(labels, np.asarray(predictions)[:,:k])])
        top5 = (correct/len(labels))

        # accuracy 
        return top1, top5
    
    # def evaluate_embedding(embeddings, labels, search=True):
    #     labels = preprocessing.LabelEncoder().fit_transform(labels)
    #     x, y = np.array(embeddings), np.array(labels)
    #     # print(x.shape, y.shape)

    #     logreg_accuracies = [logistic_classify(x, y) for _ in range(1)]
    #     # print(logreg_accuracies)
    #     print('LogReg', np.mean(logreg_accuracies))

    #     svc_accuracies = [svc_classify(x,y, search) for _ in range(1)]
    #     # print(svc_accuracies)
    #     print('svc', np.mean(svc_accuracies))

    #     linearsvc_accuracies = [linearsvc_classify(x, y, search) for _ in range(1)]
    #     # print(linearsvc_accuracies)
    #     print('LinearSvc', np.mean(linearsvc_accuracies))

    #     randomforest_accuracies = [randomforest_classify(x, y, search) for _ in range(1)]
    #     # print(randomforest_accuracies)
    #     print('randomforest', np.mean(randomforest_accuracies))

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
