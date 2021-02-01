""" Loss related code

    @author: jhuthmacher
"""
import math

import matplotlib
import torch
from torch import nn
import torch.nn.functional as F

matplotlib.use('Agg')


def bce_loss(enc_global: torch.Tensor, enc_local: torch.Tensor):
    """ Binary cross entropy loss with preceeding discriminator.

        The idea: Train the encoder to distinguish between positive and negative
        samples in a classification set up.

        Parameters:
            enc_local: torch.Tensor
                Represents the encoded node/patch level embedding.
                Dimension (batch_size, embedding_dim, num_nodes).
                I.e. we have per batch entry a matrix where each row corresponds to the embedding
                of the corresponding node.
            enc_global: torch.Tensor
                Represents the encoded graph level/global embedding.
                Dimension (batch_size, embedding_dim)
                I.e. we have per batch entry a single embedding of the corresponding graph.
        Return:
            torch.Tensor: Tensor containing the (batch) loss (mutual information estimate).
    """
    self = bce_loss

    self.b_xent = nn.BCEWithLogitsLoss()

    self.num_graphs = enc_global.shape[0]
    self.num_nodes = enc_local.shape[-1]

    #### Discriminator ####
    # Row wise matrix product! Final dimension: (num_graphs, num_graphs * num_nodes)
    self.discr_matr = torch.bmm(enc_global.repeat(self.num_graphs, 1, 1), enc_local)
    self.discr_matr = self.discr_matr.type("torch.FloatTensor").permute(1,0,2)
    self.discr_matr = self.discr_matr.reshape(self.num_graphs, (self.num_graphs)*self.num_nodes)

    #### Sampling ####
    # Diagonal with blocks 'num_nodes' ones on the diagonal. Labels the date
    # with 1 for positive sample and 0 for negative sample.
    self.mask = torch.block_diag(*[torch.ones(self.num_nodes) for _ in range(self.num_graphs)]).to("cpu")

    return self.b_xent(self.discr_matr, self.mask)

def jensen_shannon_mi(enc_global: torch.Tensor, enc_local: torch.Tensor):
    """ Jensen-Shannon mutual information estimate.

        Parameters:
            enc_local: torch.Tensor
                Represents the encoded node/patch level embedding.
                Dimension (batch_size, embedding_dim, num_nodes).
                I.e. we have per batch entry a matrix where each row corresponds to the embedding
                of the corresponding node.
            enc_global: torch.Tensor
                Represents the encoded graph level/global embedding.
                Dimension (batch_size, embedding_dim)
                I.e. we have per batch entry a single embedding of the corresponding graph.
        Return:
            torch.Tensor: Tensor containing the (batch) loss (mutual information estimate).
    """
    #### For Tracking ####
    self = jensen_shannon_mi

    self.num_graphs = enc_global.shape[0]
    self.num_nodes = enc_local.shape[-1]

    #### Discriminator ####
    # Row wise matrix product! Final dimension: (num_graphs, num_graphs * num_nodes)
    self.discr_matr = torch.bmm(enc_global.repeat(self.num_graphs, 1, 1), enc_local)
    self.discr_matr = self.discr_matr.type("torch.FloatTensor").permute(1,0,2)
    self.discr_matr = self.discr_matr.reshape(self.num_graphs, (self.num_graphs)*self.num_nodes)

    #### Sampling ####
    # Diagonal with blocks 'num_nodes' ones on the diagonal
    self.mask = torch.block_diag(*[torch.ones(self.num_nodes) for _ in range(self.num_graphs)]).to("cpu")

    self.pos_samples = self.discr_matr[self.mask == 1].reshape(self.num_graphs, self.num_nodes)
    self.neg_samples = self.discr_matr[self.mask == 0].reshape(self.num_graphs,
                                                          (self.num_graphs - 1) * self.num_nodes)

    #### Expectation Calculation ####
    # Normalize with the size of the batch to get the batch loss.
    self.E_pos = get_positive_expectation(self.pos_samples, average=False).sum()
    self.E_pos = (self.E_pos / (self.num_nodes * self.num_graphs)) 
    self.E_neg = get_negative_expectation(self.neg_samples, average=False).sum()
    self.E_neg = (self.E_neg / (self.num_nodes * (self.num_graphs - 1) * self.num_graphs))

    #### Actual Loss Calculation ####
    return self.E_neg - self.E_pos


def get_positive_expectation(p_samples: torch.Tensor, measure: str = "JSD", average: bool = True):
    """Computes the positive part of a divergence / difference.

        @source: https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/losses.py

        Paramters:
            p_samples: torch.Tensor
                Tensor of dimension (num_graphs, num_nodes*num_pos_pairs).
                I.e. the tensor contains per column i the positive samples (stacked together if
                multiple samples) of the correspondings graph i.
            measure: str
                Determines wich method is used for calculating the positive part of a divergence.
            average: bool
                Determines if the results should be averaged over samples.
        Returns:
            torch.Tensor: Dimension (num_graphs, num_nodes*num_pos_pairs) if not averaged.
            If average == True the dimension is (num_graphs, 1)
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Ep = - F.softplus(-p_samples)
    elif measure == 'JSD':
        Ep = log_2 - F.softplus(- p_samples)
    elif measure == 'X2':
        Ep = p_samples ** 2
    elif measure == 'KL':
        Ep = p_samples + 1.
    elif measure == 'RKL':
        Ep = -torch.exp(-p_samples)
    elif measure == 'DV':
        Ep = p_samples
    elif measure == 'H2':
        Ep = 1. - torch.exp(-p_samples)
    elif measure == 'W1':
        Ep = p_samples
    else:
        ValueError(f"Measurement method {measure} not found!")

    if average:
        return Ep.mean()
    else:
        return Ep


def get_negative_expectation(n_samples: torch.Tensor, measure: str ="JSD", average: bool = True):
    """Computes the negative part of a divergence / difference.

        @source: https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/losses.py

        Paramters:
            p_samples: torch.Tensor
                Tensor of dimension (num_graphs, num_nodes*num_pos_pairs).
                I.e. the tensor contains per column i the negative samples (stacked together if
                multiple samples) of the correspondings graph i.
            measure: str
                Determines wich method is used for calculating the negative part of a divergence.
            average: bool
                Determines if the results should be averaged over samples.
        Returns:
            torch.Tensor: Dimension (num_graphs, num_nodes*num_pos_pairs) if not averaged.
            If average == True the dimension is (num_graphs, 1)
    """
    log_2 = math.log(2.)

    if measure == 'GAN':
        Eq = F.softplus(-n_samples) + n_samples
    elif measure == 'JSD':
        Eq = F.softplus(-n_samples) + n_samples - log_2
    elif measure == 'X2':
        Eq = -0.5 * ((torch.sqrt(n_samples ** 2) + 1.) ** 2)
    elif measure == 'KL':
        Eq = torch.exp(n_samples)
    elif measure == 'RKL':
        Eq = n_samples - 1.
    # elif measure == 'DV':
    #     Eq = log_sum_exp(n_samples, 0) - math.log(n_samples.size(0))
    elif measure == 'H2':
        Eq = torch.exp(n_samples) - 1.
    elif measure == 'W1':
        Eq = n_samples
    else:
        ValueError(f"Measurement method {measure} not found!")

    if average:
        return Eq.mean()
    else:
        return Eq
