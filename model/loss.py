"""
    @author: jhuthmacher
"""

import torch
import torch.nn.functional as F

import math


def jensen_shannon_mi(enc_local: torch.Tensor, enc_global:torch.Tensor):
    """ Jensen-Shannon mutual information estimate.

        Parameters:
            enc_local: torch.Tensor
                Represents the encoded node/patch level embedding. Dimension (batch_size, embedding_dim, num_nodes).
                I.e. we have per batch entry a matrix where each row corresponds to the embedding of the corresponding node.
            enc_global: torch.Tensor
                Represents the encoded graph level/global embedding. Dimension (batch_size, embedding_dim)
                I.e. we have per batch entry a single embedding of the corresponding graph.
        Return:
            torch.Tensor: Tensor containing the loss (mutual information estimate) with dimension (batch_size, 1). 
            I.e. one loss (mi estimate) per graph.
    """

    num_graphs = enc_global.shape[0]
    num_nodes = enc_local.shape[-1]
    
    #################
    # Discriminator #
    #################
    # Row wise dot product!
    # I.e. we get for each node and for each graph a vector witht node size, which can be interpreted as our discriminator vector!
    # Output: Per batch entry we get a disciriminator vector (from the formular T(h, H), but T is not a neural net yet)
    # yhat contains negative and positive samples!
    yhat = torch.bmm(enc_local.permute(0, 2, 1), enc_global.view(enc_global.shape[0], enc_global.shape[1],1))
    # yhat = yhat.t() # Transpose to get the same dimensions as in InfoGraph (nodes x graphs)
    
    ############
    # Sampling #
    ############
    yhat_matr = torch.flatten(yhat).repeat(num_graphs,1,1).reshape(num_graphs, num_graphs, num_nodes)

    # Create unit matrix to mask positive and negative samples
    unit = torch.eye(num_graphs, num_graphs).reshape((num_graphs, num_graphs, 1))
    unit = unit.repeat(1, 1, num_nodes)

    pos_samples = yhat_matr[unit == 1].reshape(num_graphs, num_nodes) # equivalent to torch.diagonal(yhat_matr)
    neg_samples = yhat_matr[unit == 0].reshape(num_graphs, (num_graphs-1)*num_nodes) # Checked!

    # Batch loss, i.e. aggregate and normalize
    E_pos = get_positive_expectation(pos_samples, average=False).sum()
    E_pos = E_pos / num_nodes
    E_neg = get_negative_expectation(neg_samples, average=False).sum()
    E_neg = E_neg / (num_nodes * (num_graphs - 1))

    return E_neg - E_pos


def get_positive_expectation(p_samples: torch.Tensor, measure: str = "JSD", average: bool = True):
    """Computes the positive part of a divergence / difference.

        @source: https://github.com/fanyun-sun/InfoGraph/blob/master/unsupervised/losses.py

        Paramters:
            p_samples: torch.Tensor
                Tensor of dimension (num_graphs, num_nodes*num_pos_pairs). I.e. the tensor contains per column i
                the positive samples (stacked together if multiple samples) of the correspondings graph i.
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
                Tensor of dimension (num_graphs, num_nodes*num_pos_pairs). I.e. the tensor contains per column i
                the negative samples (stacked together if multiple samples) of the correspondings graph i.
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
    elif measure == 'DV':
        Eq = log_sum_exp(n_samples, 0) - math.log(n_samples.size(0))
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