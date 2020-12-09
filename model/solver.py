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

from data import KINECT_ADJACENCY

from model.loss import jensen_shannon_mi

#########################
# Local/Tracking Config #
#########################
import config.config as cfg
from config.config import log

from sacred import Experiment


class Solver(object):
    """ The Solver class implement several functions to provide a common training and testing procedure.
    """

    #pylint: disable=dangerous-default-value
    def __init__(self, model: nn.Module, dataloader: [DataLoader],
                 optimizer: torch.optim.Optimizer = None,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 train_cfg = {}, test_cfg = {}):
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
                optimizer: torch.optim.Optimizer
                    Optimzer used to train the model.
                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
                    Function calculating the loss of the model.
        """
        #### Train configuration ####
        self.train_cfg = {
            "n_epochs": 10,
            "log_nth": 0,
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

        #### load check point

        if isinstance(model, tuple):
            # Model is a checkpoint
            checkpoint = model[0]
            modelClass = model[1]
            optimClass = model[2] if len(model) > 2 else optim.Adam

            self.model = modelClass(**checkpoint['model_params'])
            self.model.load_state_dict(checkpoint['model_state_dict'])

            optim_params = self.train_cfg["optimizer"] if checkpoint['optim_params'] == {} else checkpoint['optim_params']
            self.optimizer = optimClass(self.model.parameters(), **optim_params)
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

            self.epoch = checkpoint['epoch']
        else:
            self.model = model
            self.optimizer = (optim.Adam(
                    model.parameters(), 
                    **self.train_cfg["optimizer"],
                ) 
                if optimizer is None 
                else optimizer)

        self.loss_fn = jensen_shannon_mi if loss_fn is None else loss_fn

        self.train_loader = dataloader[0]
        self.val_loader = dataloader[1] if len(dataloader) == 2 or len(dataloader) == 3 else None
        self.test_loader = dataloader[2] if len(dataloader) == 3 else None

        # Object storage to track the performance
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []  # Useless?

        self.val_metrics = []
        self.predictions = np.array([])
        self.labels = np.array([])

        self.val_pred = np.array([])
        self.val_label = np.array([])

        self.epoch = 0

    def train(self, train_config: dict = None, track = None):
        """ Training method from the solver.

            Paramters:
                train_config: dict
                    Dictionary to adapt the trainings configuration.
        """

        if train_config is not None:
            # Merges custom config with default config!
            self.train_cfg = {**self.train_cfg, **train_config}

        
        # decayRate = 0.95
        # lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=decayRate)
        
        #########################
        # Training & Validation #
        #########################
        start_epoch = self.epoch
        for self.epoch in trange(start_epoch, self.train_cfg["n_epochs"],
                                 disable=(not self.train_cfg["verbose"]),
                                 desc=f'Epochs ({self.model.__class__.__name__})'):
            self.train_batch_losses = []
            self.val_batch_losses = []

            #########
            # Train #
            #########
            self.model.train()
            for self.batch, (batch_x, batch_y) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), leave=False,
                                                                 disable=False, desc=f'Trai. Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                try:
                    batch_x = torch.tensor(batch_x.astype("float64")).permute(0,3,2,1)
                except:
                    # For testing MLP on iris
                    batch_x = torch.tensor(batch_x.astype("float64"))            
                
                self.optimizer.zero_grad() # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch

                # For each action/dynamic graph in the batch we get the gbl and lcl representation
                # yhat = gbl, lcl
                yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                if isinstance(yhat, tuple):
                    loss = self.loss_fn(*yhat)
                else:
                    loss = self.loss_fn(yhat, torch.tensor(batch_y, dtype=torch.long).to(self.model.device))

                self.train_batch_losses.append(torch.squeeze(loss).item())
                loss.backward()
                clip_grad_norm_(self.model.parameters(), 1)
                self.optimizer.step()

                # if callable(track):
                #     track("training")

            if self.val_loader is not None:
                # For disabling the gradient calculation -> Performance advantages
                with torch.no_grad():
                    self.val_pred = np.array([])
                    self.val_label = np.array([])
                    ############
                    # Validate #
                    ############
                    self.model.eval()
                    for self.batch, (batch_x, batch_y) in enumerate(tqdm(self.val_loader, disable=False, leave=False,
                                                                         desc=f'Vali. Batch (Epoch: {self.epoch})')):
                        # The train loader returns dim (batch_size, frames, nodes, features)
                        try:
                            batch_x = torch.tensor(batch_x.astype("float32")).permute(0,3,2,1)
                        except:
                            # For testing MLP on iris
                            batch_x = torch.tensor(batch_x.astype("float32"))   
                        
                        # For each action/dynamic graph in the batch we get the gbl and lcl representation
                        # yhat = gbl, lcl
                        yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                        if isinstance(yhat, tuple):
                            loss = self.loss_fn(*yhat)
                        else:
                            loss = self.loss_fn(yhat, torch.tensor(batch_y, dtype=torch.long).to(self.model.device))
                            
                            #### EVALUATION DURING VALIDATION ####
                            yhat_idx = torch.argsort(yhat, descending=True)
                            self.val_pred = np.vstack([self.val_pred, yhat_idx.cpu().numpy()]) if self.val_pred.size else yhat_idx.cpu().numpy()
                            self.val_label = np.append(self.val_label, batch_y)

                        self.val_batch_losses.append(torch.squeeze(loss).item())

                    if not isinstance(yhat, tuple):
                        self.val_metric = self.evaluate(self.val_pred, self.val_label)

                        self.val_metrics.append(self.val_metric )

                        
                        # if callable(track):
                        #     track("validation")

            # lr_scheduler.step()

            self.train_losses.append(np.mean(self.train_batch_losses) if len(self.train_batch_losses) > 0 else 0)
            self.val_losses.append(np.mean(self.val_batch_losses) if len(self.val_batch_losses) > 0 else 0)

            if callable(track):
                track("epoch")
            
    def test(self, test_config: dict = None, model: nn.Module = None, test_loader = None,
             track = None):
        """ Function to test a model, i.e. calculate evaluation metrics.
        """
        if model is not None:
            self.model = model
        if test_loader is not None:
            self.test_loader = test_loader

        self.predictions = np.array([])
        self.labels = np.array([])

        with torch.no_grad():
        
            #### TEST ####
            self.model.eval()
            for self.batch, (batch_x, batch_y) in enumerate(tqdm(self.test_loader, disable=False, leave=False,
                                                                 desc=f'Test. Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                try:
                    batch_x = torch.tensor(batch_x.astype("float64")).permute(0,3,2,1)
                except:
                    # For testing MLP on iris
                    batch_x = torch.tensor(batch_x.astype("float64"))
                        
                # For each action/dynamic graph in the batch we get the gbl and lcl representation
                # yhat = gbl, lcl
                # yhat stochastic vector
                yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                if isinstance(yhat, tuple):
                    # loss = self.loss_fn(*yhat) 
                    # TODO: Evaluation of TIG model?
                    pass
                else:
                    yhat_idx = torch.argsort(yhat, descending=True)
                    
                    self.predictions = np.vstack([self.predictions, yhat_idx.cpu().numpy()]) if self.predictions.size else yhat_idx.cpu().numpy()
                    self.labels = np.append(self.labels, batch_y)

                    # self.predictions.append(yhat_idx.numpy())
                    # self.labels.append(batch_y)

                    # pred_labels = torch.stack([yhat, batch_y], 1)

                    # if self.results is None:
                    #     self.results = pred_labels
                    # else:
                    #     self.results = torch.cat((self.results, pred_labels))    

            self.metric = self.evaluate(self.predictions, self.labels)

            if callable(track):
                track("evaluation")           

            return self.metric
    
    def evaluate(self, predictions, labels, mode="top-k"):
        """
        """

        k = 1  # Top-k
        correct = np.sum(np.isin(labels, np.asarray(predictions)[:,:k]))
        top1 = (correct/len(labels))

        k = 5  # Top-k
        correct = np.sum(np.isin(labels, np.asarray(predictions)[:,:k]))
        top5 = (correct/len(labels))

        # accuracy 
        return top1, top5

    def __repr__(self):
        """ String representation
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
