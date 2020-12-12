"""
    Implementation of a the Temporal Info Graph.

    @author: j-huthmacher
"""
from typing import Any

import torch
import torch.nn as nn

import numpy as np
from math import floor

from torchsummary import summary

import config.config as cfg
from config.config import log

class TemporalConvolution(nn.Module):
    """ Building block for doing temporal convolutions.

        Reference implementation: https://github.com/open-mmlab/mmskeleton/blob/master/mmskeleton
    """

    def __init__(self, c_in: int, c_out: int = 1, kernel: Any = 1,
                 weights: Any = None, activation: str = "leakyReLU", dropout = 0.5):
        """ Initialization of the temporal convolution, which is represented by a simple 2D convolution.

            Parameters:
                c_in: int
                    Number of input channel, i.e. number of features.
                c_out: int
                    Output dimension of the feature map, i.e. numbers of features of the feature map.
                kernel: int or tuple
                    Kernel dimension. To maintain the node dimension the first kernel dimension has 
                    to be 1! With the second dimension you can control how many time steps you
                    will consider.
                weights: torch.Tensor
                    Tensor containing weights of the kernel. E.g. for testing, not for production!
                    If weights are provided, the bias in the convolutional layer is disabled.
                activation: str
                    Determines which activation function is used. Options: ['leakyReLU', 'ReLU']
        """
        super().__init__()

        # Set paramters
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.weights = weights
        
        if isinstance(self.kernel, int):
            self.kernel = (1, self.kernel)

        self.conv = nn.Conv2d(self.c_in,
                              self.c_out,
                              kernel_size=self.kernel,
                              bias=(self.weights == None))

        if weights is not None:            
            self.conv.weight = torch.nn.Parameter(self.weights)

        #### REGULARIZATION ####
        self.bn1 = nn.BatchNorm2d(self.c_in)
        self.bn2 = nn.BatchNorm2d(self.c_out)
        self.dropout = nn.Dropout(dropout)

        #### ACTIVATIOn ####
        if activation == "leakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = None
        
        #### TO DEVICE #####
        self.conv = self.conv.to(self.device)
    
    @property
    def device(self):
        return next(self.parameters()).device
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X: torch.Tensor):
        """ Foward operation of the temporal convolution.

            torch.nn.Conv2d input dim. (N, C_in, H_in, W_in)

            Parameters:
                X: torch.Tensor
                    Input feature matrix. Dimension (batch, features, time, nodes)

            Return:
                torch.Tensor: Dimension (batch, features, time, nodes)
        """
        X = self.bn1(X)
        X = self.conv(X)
        X = self.bn2(X)
        X = self.dropout(X)

        return X if self.activation is None else self.activation(X)
    
    def convShape(self, dim_in: tuple):
        """ Calculate the output dimension of a convolutional layer.

            Parameter:
                dim_in: tuple
                    Input dimension of the data. Here we use the scheme of axis, i.e. input
                    dim follows the form (x,y).
                    dim_in[0] = x; dim_in[1] = y 
        """

        nominatorH = (dim_in[0] + (2 * self.conv.padding[0]) - (self.conv.dilation[0] * 
                      (self.conv.kernel_size[0] - 1)) - 1)

        nominatorW = (dim_in[1] + (2 * self.conv.padding[1]) - (self.conv.dilation[1] * 
                      (self.conv.kernel_size[1] - 1)) - 1)

        return (floor((nominatorH/self.conv.stride[0]) + 1),
                floor((nominatorW/self.conv.stride[1]) + 1))

        
class SpectralConvolution(nn.Module):
    """ Building block for doing spectral convolutions on a graph.

        Reference implementation: https://github.com/open-mmlab/mmskeleton/blob/master/mmskeleton
    """

    def __init__(self, c_in: int , c_out: int = 2, kernel: Any = 1,
                 method: str = "gcn", weights: Any = None, activation: str = "leakyReLU"):
        """ Initialization of the spectral convolution layer.

            Parameters:
                c_in: int
                    Input channels, i.e. features
                c_out: int
                    Output channels, i.e. output feaures or features of the feature map.
                k: int
                    Spectral kernel size, i.e. k determines the k-hop neighborhood
                method: str
                    Determines which method is used for spectral convolution. 
                    Options ['gcn']
                weights: None or torch.Tensor (optional)
                    Predefined weight matrix, if you want to test/validate the calculation.
                activation: str
                    Determines which activation function is used. Options: ['leakyReLU', 'ReLU']
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel
        self.method = method
        self.weights = weights

        # Weights (or kernel matrix)
        if method == "gcn":
            if self.weights is not None:
                self.W = self.weights
            else:
                self.W = torch.nn.Parameter(torch.rand((self.c_in, self.c_out)))
        
        # Set activation
        if activation == "leakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = None       
    
    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """ Forward pass of the spectral convolution layer.

            Parameters:
                X: torch.Tensor
                    Feature tensor of dimension (batch, features, nodes, time)
                A: torch.Tensor
                    Corresponding adjacency matrix of dimension (nodes, nodes)
            Return:
                torch.Tensor: Convoluted feature tensor (feature map) of 
                dimension (batch, nodes, time, features)
        """
        # Adjacency multiplication in time!
        # TODO: Adapt the indices to omit the permutation before and after.
        self.W = self.W.to(self.device)
        H = torch.einsum("ij,jklm->kilm", [A, X.permute(2, 0, 3, 1)]).to(self.device)
        H = torch.matmul(H, self.W)
        
        return H if self.activation is None else self.activation(H)


class TemporalInfoGraph(nn.Module):
    """ Implementation of the temporal info graph model. 
    """

    def __init__(self, dim_in: tuple, c_in: int, c_out: int = 2, spec_out: int = 2, out: int = 2,
                 tempKernel: Any = 1, activation: str = "leakyReLU", batch_norm: bool = True):
        """ Initilization of the TIG model.

            Parameter:
                dim_in: tuple
                    Represents a tuple with the dimensions of the input data, i.e. height and width.
                    In the temporal graph set up this would corresponds to (nodes, features).
                    Those values are needed to calculate the kernel for the last temporal layer, which 
                    covers the whole input to finally reduce the data to a "single timestamp".
                c_in: int
                    Input channels of the data, i.e. features.
                c_out: int
                    Output channels after the first temporal convolution.
                spec_out: int
                    Spectral output channels, i.e. after the spectral convolution.
                out: int
                    Overall output channels, i.e. after last temporal convoltion.
                tempKernel: int or tuple
                    Kernel for the first temporal convolution. At the moment the kernel of the second 
                    convolution is automatically calculated to convolve always over all remaining time steps.
                activation: str
                    Determines which activation function is used. Options: ['leakyReLU', 'ReLU']                
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.spec_out = spec_out
        self.tempKernel = tempKernel
        self.out = out
        self.dim_in = dim_in 

        self.tempLayer1 = TemporalConvolution(c_in=self.c_in, c_out=self.c_out, kernel=self.tempKernel)
        k2 = max(1, self.tempLayer1.convShape(dim_in)[1])
        self.specLayer1 = SpectralConvolution(c_in=self.c_out, c_out=self.spec_out)
        self.tempLayer2 = TemporalConvolution(c_in=self.spec_out, c_out = self.out, kernel = k2)

        #### REGULARIZATION ####
        self.bn = nn.BatchNorm2d(self.c_in) if batch_norm else None

        #### ACTIVATION #####
        if activation == "leakyReLU":
            self.activation = nn.LeakyReLU()
        elif activation == "ReLU":
            self.activation = nn.ReLU()
        else:
            self.activation = None        

        #### TO DEVICE #####
        self.tempLayer1 = self.tempLayer1.to(self.device)
        self.specLayer1 = self.specLayer1.to(self.device)
        self.tempLayer2 = self.tempLayer2.to(self.device)
    
    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X: torch.Tensor, A: torch.Tensor):
        """ Forward function of the temporl info graph model.

            Consists of a initial temporal convolution, followed by an spectral
            convolution and finalized with another temporal convolution.

            Parameters:
                X: torch.Tensor
                    Input matrix, i.e. feature matrix of the nodes.
                    Dimension (batch, features, nodes, time)
                A: torch.Tensor
                    Adjacency matrix. Dimension (nodes, nodes)

            Return:
                torch.Tensor, torch.Tensor: 
                Dimensions: (batch, nodes), (batch, features, nodes)
        """
        if not isinstance(X, torch.Tensor):
            X = torch.tensor(X, dtype=torch.float32)
        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32)

        # Features could be twice or more!
        X = X.type('torch.FloatTensor').to(self.device)
        A = A.type('torch.FloatTensor').to(self.device)

        if self.bn is not None:
            X = self.bn(X)

        H = self.tempLayer1(X) # Returns: (batch, out_features, nodes, time)

        #### Masking Padding ####
        batch_size, _, num_nodes, num_frames = X.shape
        _, out_feature, _, out_time = H.shape
        pad_idx = torch.repeat_interleave((~X.bool()).all(dim=1).all(dim=1),
                                          repeats=num_nodes, dim=0).view(batch_size, num_nodes, -1)
        batch_size, _, num_nodes, num_frames = X.shape
        H.permute(1,0,2,3)[:, pad_idx[:, :, (num_frames-out_time):]] = 0

        H1 = self.specLayer1(H, A) # Returns: (batch, nodes, time, features)
        Z = self.tempLayer2(H1.permute(0, 3, 1, 2)) # Expects: (batch, features, nodes, time)
        Z = Z if self.activation is None else self.activation(Z)

        # Mean readout: Average each feature over all nodes --> dimension (features, 1)
        # Average over dimension 2, dim 2 are the nodes (We want to aggregate the representation of each node!)
        global_Z = Z.mean(dim=2) # Simple mean readout
        local_Z = Z

        # Alternatively another fully connected feed forward network to encode the embedding

        # Remove "empty" dimensions, i.e. dim = 1
        return torch.squeeze(global_Z, dim=-1), torch.squeeze(local_Z, dim=-1)

    @property
    def num_paramters(self):
        # return(summary(self, (self.c_in, self.dim_in[0], self.dim_in[1])))
        return f"Parameters {sum(p.numel() for p in self.parameters())}"
