""" Implementation of a the Temporal Info Graph.

    @author: j-huthmacher
"""
from typing import Any
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_utils import get_normalized_adj, pad_zero_mask, pad_zero_idx

class ConstZeroLayer(torch.nn.Module):

    def forward(self, x):
        return torch.tensor([0])

class FF(nn.Module):
    """
        @source: InfoGraph
    """
    def __init__(self, input_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, input_dim),
            nn.ReLU()
        )
        self.linear_shortcut = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        return self.block(x) + self.linear_shortcut(x)


class TemporalConvolution(nn.Module):
    """ Building block for doing temporal convolutions.

        Reference implementation: https://github.com/open-mmlab/mmskeleton/blob/master/mmskeleton
    """

    def __init__(self, c_in: int, c_out: int = 1, kernel: Any = 1,
                 weights: Any = None, activation: str = "LeakyReLU", dropout = 0.5,
                 bn_in: bool = False, debug=False, causal=False):
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
        self.debug = debug

        if weights is not None:
            self.kernel = weights.shape[2:]
        elif isinstance(self.kernel, int):
            self.kernel = (1, self.kernel)

        #### Padding for Temporal Dependency #####
        if causal:
            padding = ((self.kernel[0] - 1) // 2, 0)
        else:
            padding = 0

        self.conv = nn.Conv1d(self.c_in,
                              self.c_out,
                              kernel_size=self.kernel[1],
                              padding=padding)

        if weights is not None:
            self.conv.weight = torch.nn.Parameter(self.weights)

        #### REGULARIZATION ####
        self.bn_in = bn_in
        if bn_in:
            self.bn1 = nn.BatchNorm2d(self.c_in)

        self.bn2 = nn.BatchNorm2d(self.c_out)
        self.dropout = nn.Dropout(dropout)

        #### ACTIVATIOn ####
        self.activation = getattr(nn, activation)()

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
                    Input feature matrix. Dimension (batch, features, nodes, times)

            Return:
                torch.Tensor: Dimension (batch, features, time, nodes)
        """

        if not self.debug:
            if hasattr(self, "bn_in") and self.bn_in:
                X = self.bn1(X)

        N, C, V, T = X.shape

        # pad_idx = pad_zero_idx(X.permute(0,3,2,1))

        #### Transform the Data and apply 1D conv ####
        X = X.permute(0, 2, 1, 3).reshape(N * V, C, T)  # ()
        # In: (batch * nodes, features, time), Out: (batch * nodes, c_out, time)
        X = self.conv(X)
        X = X.view(N, V, X.shape[1], -1).permute(0, 2, 1, 3)

        # pad_mask = pad_zero_mask(X.permute(0,3,2,1).shape, pad_idx)
        # X.permute(0,3,2,1)[pad_mask] = 0

        if not self.debug:
            # X = self.bn2(X)
            X = self.dropout(X)

        # X = X.permute(0, 1, 3, 2)

        return X if self.activation is None or self.debug else self.activation(X)

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
                 weights: Any = None, activation: str = "LeakyReLU",
                 debug=False, A = None):
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
        self.weights = weights
        self.debug = debug
        self.A = A

        # Weights (or kernel matrix)
        if self.weights is not None:
            # self.W = torch.nn.Parameter(torch.tensor(self.weights))
            self.W = weights
        else:
            self.W = torch.nn.Parameter(torch.rand((self.c_in, self.c_out)))

        self.bn2 = nn.BatchNorm2d(self.c_out)

        self.activation = getattr(nn, activation)()

    @property
    def device(self):
        """ Device of the model (cpu or cuda)
        """
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X: torch.Tensor, A: torch.Tensor = None):
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
        if A is None:
            A = self.A

        if not isinstance(A, torch.Tensor):
            A = torch.tensor(A, dtype=torch.float32)
        A = A.type('torch.FloatTensor').to(self.device)

        # # Adjacency matrix multiplication in time!
        assert len(self.W.shape) == 2

        H = torch.einsum("lkjm, ji -> lkim", [X, A]) # Dim (batch, features, nodes, time)
        H = torch.matmul(H.permute(0, 3, 2, 1), self.W).permute(0, 3, 2, 1)

        # H = self.bn2(H)

        return H if self.activation is None or self.debug else self.activation(H)


class TemporalInfoGraph(nn.Module):
    """ Implementation of the temporal info graph model.
    """

    def __init__(self, architecture: [tuple] = [(2, 32, 32, 32, 32)],
                 batch_norm: bool = True, A: torch.Tensor = None,
                 discriminator_layer: bool = False, residual: bool = False,
                 edge_weights: bool = False, self_connection=True, dropout=0.5,
                 summary=False, diff_scales= False, multi_scale=False):
        """ Initilization of the TIG model.

            Parameter:
                dim_in: tuple
                    Represents a tuple with the dimensions of the input data, i.e. height and width.
                    In the temporal graph set up this would corresponds to (nodes, features).
                    Those values are needed to calculate the kernel for the last temporal layer,
                    which covers the whole input to finally reduce the data to a "single timestamp".
                architecture: [tuple]
                    [(c_in, c_out, spec_out, out, kernel)]
                c_in: int
                    Input channels of the data, i.e. features.
                c_out: int
                    Output channels after the first temporal convolution.
                spec_out: int
                    Spectral output channels, i.e. after the spectral convolution.
                out: int
                    Overall output channels, i.e. after last temporal convoltion.
                tempKernel: int or tuple
                    Kernel for the first temporal convolution. At the moment the kernel of the
                    second convolution is automatically calculated to convolve always over all
                    remaining time steps.
                activation: str
                    Determines which activation function is used. Options: ['leakyReLU', 'ReLU']                
        """
        super().__init__()

        #### Model Creation #####
        self.edge_weights = nn.ParameterList()
        self.embedding_dim = architecture[-1][-2]
        self.c_in = architecture[0][0]
        self.diff_scales = diff_scales
        self.summary = summary
        self.multi_scale = multi_scale

        if A is not None and self_connection:
            # mask = torch.eye(A.shape[0])
            # A = mask + (1. - mask)*A
            A = get_normalized_adj(A)
            self.register_buffer('A', torch.tensor(A))

        #### Initial Data Normalization ####
        if batch_norm and A is not None:
            # Features * nodes
            self.num_nodes = A.shape[0]
            self.data_norm = nn.BatchNorm1d(self.c_in * self.num_nodes)

        # self.upscale = nn.Sequential(nn.Linear(2, 64), nn.LeakyReLU())
        # self.specConvIn = SpectralConvolution(c_in=64, c_out=64)
        # self.specConvOut = SpectralConvolution(c_in=64, c_out=64)

        self.layers = []
        self.res_layers = []
        self.batch_norms = []
        self.concat_dim = 0

        for layer in architecture:
            if len(layer) == 5:
                c_in, c_out, spec_out, out, kernel = layer
                tempConv1 = TemporalConvolution(c_in=c_in, c_out=c_out,
                                                kernel=kernel, dropout=dropout)
                specConv = nn.Sequential(SpectralConvolution(c_in=c_out, c_out=spec_out, A=self.A))
                                        #  SpectralConvolution(c_in=spec_out, c_out=spec_out, A=self.A))
                # specConv = nn.Identity()
                tempConv2 = TemporalConvolution(c_in=spec_out, c_out=out, kernel=kernel, dropout=dropout)
            else:
                c_in, c_out, out, kernel = layer
                tempConv1 = TemporalConvolution(c_in=c_in, c_out=c_out, kernel=kernel, dropout=dropout)
                # specConv = SpectralConvolution(c_in=c_out, c_out=out)#, weights=self.spectral_weights)
                specConv = nn.Identity()
                tempConv2 = nn.Identity()
            
            self.layers.append(nn.Sequential(tempConv1, specConv, tempConv2, nn.LeakyReLU()))

            #### Residual Layer ####
            if (c_in != out) and residual:
                res_kernel = (1, (kernel*2)-1) if len(layer) == 5 else (1, kernel)
                residual = nn.Sequential(nn.Conv2d(c_in, out, kernel_size=res_kernel),
                                         nn.BatchNorm2d(out))
                # residual = nn.Identity() # Feed the raw input to the end
                self.res_layers.append(residual)
            else:
                self.res_layers.append(ConstZeroLayer())

            #### Edge Paramter ####
            if A is not None:
                if edge_weights:
                    self.edge_weights.append(nn.Parameter(torch.ones(self.A.size())))
                else:
                    self.edge_weights.append(nn.Parameter(torch.ones(self.A.size()), requires_grad=False))
            else:
                self.edge_weights.append(nn.Parameter(torch.tensor(1), requires_grad=False))

            #### Batch Norm Layer ####
            self.batch_norms.append(nn.BatchNorm2d(out))

            #### Determines Concat Dim ####
            self.concat_dim += out

        self.batch_norms = nn.Sequential(*self.batch_norms)
        self.convLayers = nn.Sequential(*self.layers)
        self.res_layers = nn.Sequential(*self.res_layers)

        #### Determine Output Dimension ####
        with torch.no_grad():
            input_size = (self.c_in, self.num_nodes, 300)
            xhat = torch.zeros(input_size).unsqueeze_(dim=0)
            _, self.out_ch, _, self.out_dim = self.convLayers(xhat).shape

        #### Discriminator FF ####
        self.discriminator_layer = discriminator_layer
        if discriminator_layer:
            self.global_ff = FF(self.embedding_dim)
            self.local_ff = FF(self.embedding_dim)

        #### Fully Connected Layer ####
        if self.diff_scales:
            self.concat_readout = nn.Conv1d(self.concat_dim, self.embedding_dim, kernel_size=1)
            self.out_dim = self.concat_dim

        
        self.fc = nn.Conv2d(self.out_dim, 1, kernel_size=1)

        #### Global Readout ####
        self.readout = nn.Sequential(nn.Linear(36*256, self.embedding_dim))

        # self.readout = nn.Sequential(nn.Linear(self.emb_dim * 36, self.emb_dim))
        # self.readout = nn.Identity() #nn.Sequential(nn.Linear(256 * 36, 256))

        # self.bn = nn.BatchNorm2d(self.out_ch)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, X: torch.Tensor):
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

        # Features could be twice or more!
        X = X.type('torch.FloatTensor').to(self.device)
        N, C, V, T = X.shape

        #### Data Normalization ####
        if hasattr(self, "data_norm"):            
            X = X.reshape(N, V * C, T)  # Flatten to create large feature matrix
            X = self.data_norm(X)
            # Convert back to original format
            X = X.reshape(N, C, V, T)

        concat_local = []
        concat_scales = []

        for layer, e_weight, bn, resLayer in zip(self.convLayers, self.edge_weights, self.batch_norms, self.res_layers):
            # Expected layer = [(resLayer), tempConv1, specConv, tempConv2, activation]
            tempConv1, specConv, tempConv2, activation = layer[-4], layer[-3], layer[-2], layer[-1]

            res = resLayer(X).to(self.device)

            # X = self.specConvIn(X, self.A)
            X = tempConv1(X)  # (batch_size, ch_out, nodes, time)
            # X = self.specConvOut(X, self.A)
            for spec in specConv:
                X = spec(X, self.A * e_weight) # (batch_size, ch_out, nodes, time)
            # X = specConv[2](X, self.A * e_weight) # (batch_size, ch_out, nodes, time)
            X = tempConv2(X)  # (batch_size, ch_out, nodes, time)
            X = bn(X)  # stabilize the inputs to nonlinear activation functions.
            X = activation(X + res)

            if self.diff_scales:
                # Consider representations at different scale
                # concat_local.append(F.avg_pool2d(X, (1, X.shape[-1])))
                concat_local.append(F.avg_pool2d(X, (1, X.shape[-1])))
            if self.multi_scale:
                # In: (batch_size, ch_out, nodes, time) Out: (batch_size, 1, ch_out, nodes)
                concat_scales.append(torch.squeeze(F.avg_pool2d(X, (1, X.shape[-1])).permute(0,3,1,2)))

        if self.diff_scales:
            Z = self.concat_readout(torch.squeeze(torch.cat(concat_local, 1)))
        else:
            ### Fully Connected Layer ####
            # In: (batch_size, time, ch_out, nodes), Out: (batch_size, 1, ch_out, nodes))
            Z = torch.squeeze(self.fc(X.permute(0,3,1,2)))
            if self.multi_scale:
                Z = torch.cat([Z, *concat_scales], dim=2)

            # (batch_size, ch_out, nodes, time)	        # (batch_size, ch_out, nodes, time)
            # Z = X.mean(dim=3)

            # Alternatively one can pool over the time dimension, e.g. by averaging.
            # Z = torch.squeeze(F.avg_pool2d(X, (1, X.shape[-1])), dim=-1)
            # X = x.view(N, M, -1, 1, 1).mean(dim=1)

        if self.discriminator_layer:
            avgZ = self.readout(Z.permute(0,2,1).reshape(-1, Z.shape[2])).view(Z.shape[0], -1)
            global_Z = self.global_ff(avgZ) #Z.mean(dim=2))
            local_Z = self.local_ff(self.reshape_3d_2d(Z))
            local_Z = self.reshape_2d_3d(local_Z, Z.shape)
        else:
            # Z = F.tanh(Z)
            # In: (batch_size, ch_out, nodes) Out: (batch_size, ch_out)
            global_Z = torch.squeeze(F.avg_pool2d(Z, (1, Z.shape[-1])), dim=-1)
            local_Z = Z

            # global_Z = self.readout(Z.reshape(Z.shape[0], -1))#.view(Z.shape[0], -1)
            # self.readout(Z.view(N, -1))#.view(Z.shape[0], -1)
            # local_Z = Z#.permute(0, 2, 1)

        # Remove "empty" dimensions
        return torch.squeeze(global_Z, dim=-1), torch.squeeze(local_Z, dim=-1)

    def reshape_3d_2d(self, x):
        """
        """
        return x.permute(0, 2, 1).reshape(x.shape[0] * x.shape[2], x.shape[1])

    def reshape_2d_3d(self, x, shape):
        """
        """
        return x.reshape(shape[0], shape[2], shape[1]).permute(0,2,1)

    @property
    def num_paramters(self):
        """ Number of paramters of the model.
        """
        # return(summary(self, (self.c_in, self.dim_in[0], self.dim_in[1])))
        return f"Parameters {sum(p.numel() for p in self.parameters())}"
