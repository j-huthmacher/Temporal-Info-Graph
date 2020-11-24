"""
    @author: jhuthmacher
"""
from typing import Callable

import torch
import torch.optim as optim
from torch import nn
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

        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.train_cfg["learning_rate"]) if optimizer is None else optimizer
        self.loss_fn = jensen_shannon_mi if loss_fn is None else loss_fn

        self.train_loader = dataloader[0]
        self.val_loader = dataloader[1] if len(dataloader) == 2 else None
        self.test_loader = dataloader[2] if len(dataloader) == 3 else None

        # Object storage to track the performance
        self.train_losses = []
        self.val_losses = []
        self.test_losses = []

        


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
            for self.batch, batch_ in enumerate(tqdm(self.train_loader, total=len(self.train_loader), disable=True)):
                # Build plain batches
                batch_x = []
                for graph in range(batch_.num_graphs):
                    # Per graph extract feature matrix from batch
                    batch_x.append(torch.tensor(batch_.x[batch_.batch[batch_.batch==graph]]))
                
                batch_x = torch.stack(batch_x).permute(0,3,2,1)
                
                # For each action/dynamic graph in the batch we get the gbl and lcl representation
                # yhat = gbl, lcl
                yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                if isinstance(yhat, tuple):
                    loss = self.loss_fn(*yhat)
                else:
                    loss = self.loss_fn(yhat, batch_.y)

                self.train_batch_losses.append(torch.squeeze(loss).item())
                loss.backward()
                # clip_grad_norm(model.parameters(), 1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if callable(track):
                    track("training")

            if self.val_loader is not None:
                # For disabling the gradient calculation -> Performance advantages
                with torch.no_grad():
                    ############
                    # Validate #
                    ############
                    self.model.eval()
                    for self.batch, (batch_) in enumerate(tqdm(self.val_loader, disable=True)):

                        # Build plain batches
                        batch_x = []
                        for graph in range(batch_.num_graphs):
                            # Per graph extract feature matrix from batch
                            batch_x.append(torch.tensor(batch_.x[batch_.batch[batch_.batch==graph]]))
                        
                        batch_x = torch.stack(batch_x).permute(0,3,2,1)
                        
                        # For each action/dynamic graph in the batch we get the gbl and lcl representation
                        yhat = self.model(batch_x, torch.tensor(KINECT_ADJACENCY))
                
                        if isinstance(yhat, tuple):
                            loss = self.loss_fn(*yhat)
                        else:
                            loss = self.loss_fn(yhat, batch_.y)

                        self.val_batch_losses.append(torch.squeeze(loss).item())
                        
                        if callable(track):
                            track("validation")

            self.train_losses.append(np.mean(self.train_batch_losses))
            self.val_losses.append(np.mean(self.val_batch_losses))

            if callable(track):
                track("epoch")
            
 
    def test(self, test_config: dict = None):
        """ Function to test a model, i.e. calculate evaluation metrics.
        """
        raise NotImplementedError
    
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
