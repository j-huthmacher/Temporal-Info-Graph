"""
    @author: jhuthmacher
"""
from typing import Callable

import torch
import torch.optim as optim
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
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

    def __init__(self, model: nn.Module, dataloader: [DataLoader],
                 optimizer: torch.optim.Optimizer = None,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = None,
                 pad_length=None,
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
        #######################
        # Train configuration #
        #######################
        self.train_cfg = {
            "n_epochs": 10,
            "log_nth": 0,
            "learning_rate": 1e-4,
            "verbose": False
        }
        self.train_cfg = {**self.train_cfg, **train_cfg}

        ######################
        # Test configuration #
        ######################
        self.test_cfg = {
            "n_batches": 1,
        }
        self.test_cfg = {**self.test_cfg, **test_cfg}

        self.pad_length = pad_length
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.train_cfg["learning_rate"]) if optimizer is None else optimizer
        self.loss_fn = jensen_shannon_mi if loss_fn is None else loss_fn

        self.train_loader = dataloader[0]
        self.val_loader = dataloader[1] if len(dataloader) == 2 else None
        self.test_loader = dataloader[2] if len(dataloader) == 3 else None

        # Object storage to track the performance
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []  # Useless?
        self.predictions = np.array([])
        self.labels = np.array([])

    def train(self, train_config: dict = None, track = None):
        """ Training method from the solver.

            Paramters:
                train_config: dict
                    Dictionary to adapt the trainings configuration.
        """

        if train_config is not None:
            # Merges custom config with default config!
            self.train_cfg = {**self.train_cfg, **train_config}
        
        #########################
        # Training & Validation #
        #########################
        for self.epoch in tqdm(range(self.train_cfg["n_epochs"]),
                               disable=(not self.train_cfg["verbose"]),
                               desc='Epochs'):
            self.train_batch_losses = []
            self.val_batch_losses = []

            #########
            # Train #
            #########
            self.model.train()
            for self.batch, (batch_x, batch_y) in enumerate(tqdm(self.train_loader, total=len(self.train_loader), leave=False,
                                                                 disable=False, desc=f'Train Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                batch_x = torch.tensor(batch_x.astype("float64")).permute(0,3,2,1)                
                
                # For each action/dynamic graph in the batch we get the gbl and lcl representation
                # yhat = gbl, lcl
                yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                if isinstance(yhat, tuple):
                    loss = self.loss_fn(*yhat)
                else:
                    loss = self.loss_fn(yhat, torch.tensor(batch_y, dtype=torch.long).to(self.model.device))

                self.train_batch_losses.append(torch.squeeze(loss).item())
                loss.backward()
                # clip_grad_norm(model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                # if callable(track):
                #     track("training")

            if self.val_loader is not None:
                # For disabling the gradient calculation -> Performance advantages
                with torch.no_grad():
                    ############
                    # Validate #
                    ############
                    self.model.eval()
                    for self.batch, (batch_x, batch_y) in enumerate(tqdm(self.val_loader, disable=False, leave=False,
                                                                         desc=f'Val Batch (Epoch: {self.epoch})')):
                        # The train loader returns dim (batch_size, frames, nodes, features)
                        batch_x = torch.tensor(batch_x.astype("float64")).permute(0,3,2,1)
                        
                        # For each action/dynamic graph in the batch we get the gbl and lcl representation
                        # yhat = gbl, lcl
                        yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                        if isinstance(yhat, tuple):
                            loss = self.loss_fn(*yhat)
                        else:
                            loss = self.loss_fn(yhat, torch.tensor(batch_y).to(self.model.device))

                        self.val_batch_losses.append(torch.squeeze(loss).item())
                        
                        # if callable(track):
                        #     track("validation")

            self.train_losses.append(np.mean(self.train_batch_losses))
            self.val_losses.append(np.mean(self.val_batch_losses))

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
                                                                 desc=f'Test Batch (Epoch: {self.epoch})')):
                # The train loader returns dim (batch_size, frames, nodes, features)
                batch_x = torch.tensor(batch_x.astype("float64")).permute(0,3,2,1)                
                        
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
