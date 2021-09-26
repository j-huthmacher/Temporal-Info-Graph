""" Implementation of a the Temporal Info Graph.
"""
# pylint: disable=not-callable, undefined-loop-variable, too-many-instance-attributes
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.data_utils import get_normalized_adj

class ConstZeroLayer(torch.nn.Module):
    """ Neural net layer that returns constantly a zero.
    """
    # pylint: disable=unused-argument, no-self-use
    def forward(self, x: torch.Tensor):
        """ Forward pass of the layer.

            Args:
                x: torch.Tensor
                    This input is not used since the layer returns always a 0.
            Return:
                torch.Tensor: A 1x1 tensor with a 0 as value.
        """
        return torch.tensor([0])


class FF(nn.Module):
    """ Traditional Feed forward network with the following architecture.

        Linear -> ReLU -> Linear -> ReLU -> Linear -> ReLU
    """
    def __init__(self, input_dim: int):
        """
            Args:
                input_dim: int
                    Input dimenstion of the layer. All intermediate layer will have
                    the same dimension.
        """
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

    def forward(self, x: torch.Tensor):
        """ Forward pass of the network.

            After all layers another simple linear layer is added to the output.

            Args:
                x: torch.Tensor
                    Input tensor.
            Return:
                torch.Tensor: Output tensor with the dimension as the input.
        """
        return self.block(x) + self.linear_shortcut(x)


class TemporalConvolution(nn.Module):
    """ Implementation of a temporal convolution layer.
    """

    def __init__(self, c_in: int, c_out: int = 1, kernel: Any = 1):
        """
            Args:
                c_in: int
                    Number of input channel.
                c_out: int
                    Number of output channels.
                kernel: Any 
                    The temporal kernal that is applied, it defines how many time stamps
                    are consider in one convolution. Either represented by an intenger
                    or a tuple, in case it is an integer the first dimension of the 2D Kernel 
                    is one.                
        """
        super().__init__()

        # Set paramters
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel

        if isinstance(self.kernel, int):
            self.kernel = (1, self.kernel)

        # The right padding is used to get a proper temporal convolution
        padding = ((self.kernel[1] - 1) // 2, 0)
        self.tcn = nn.Sequential(
            nn.Conv2d(
                self.c_in,
                self.c_out,
                (self.kernel[1], 1),
                (1, 1),
                padding,
            ),
            nn.Dropout(0.5),
        )

        # To device
        # self.conv = self.conv.to(self.device)

        self.relu = nn.LeakyReLU()

    def forward(self, X: torch.Tensor):
        """ Forward pass of the layer.

            Args:
                X: torch.Tensor
                    Input with dim (batch, features, time, nodes) for the layer.
            Return:
                torch.Tensor: Output of the layer.
        """

        # In: (batch, features, time, nodes)
        X = self.tcn(X)

        return X


class SpectralConvolution(nn.Module):
    """ Implementation of the spectral convolution.
    """
    def __init__(self, c_in: int, c_out: int = 2, activation: str = "LeakyReLU",
                 A: torch.Tensor = None):
        """
            Args:
                c_in: int
                    Number of input channels.
                c_out: int
                    Number of output channels.
                activation: str
                    Activation function for the final output. Not used yet.
                A: torch.Tensor
                    Adjacency matrix.
        """
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.A = A

        self.conv = nn.Conv2d(self.c_in,
                              self.c_out,
                              kernel_size=(1, 1))

        # Not used!
        self.activation = getattr(nn, activation)()

    def forward(self, X: torch.Tensor, A: torch.Tensor = None):
        """ Forward pass.

            Args:
                X: torch.Tensor
                    Input tensor.
                A: torch.Tensor (optional)
                    Adjacency matrix. Normally the adjacency is set via the
                    initialization.
            Return:
                torch.Tensor: Output tensor.
        """
        if A is None:
            A = self.A

        X = self.conv(X)

        # This is how the spectral convolution is working
        X = torch.einsum('nctv,vw->nctw', (X, A))

        return X.contiguous()


class TemporalInfoGraph(nn.Module):
    """ Implementation of the Temporal Info Graph model.
    """
    def __init__(self, architecture: [int] = None, A: torch.Tensor = None,
                 mode: str = "encode", edge_weights: bool = True,
                 multi_scale: bool = False, num_classes: int = 50):
        """
            Args:
                architecture: list
                    List of integers that defines the architecture of the model.
                    The Model has always the structure: TempConv -> SpecConv -> TempConv
                    It is defined as follows:
                        architecture[0] = Input channels of first TempConv
                        architecture[1] = Output channels of the first TempConv
                                          (i.e. also input of SpecConv)
                        architecture[2] = Output channels of the SpecConv
                                          (2nd layer, i.e. also the input of the 2nd TempConv)
                        architecture[3] = Final output dimension
                A: torch.Tensor
                    Adjacency matrix.
                mode: str
                    The mode how the model is applied, either "encode" when the model shout output
                    encodings or "classify" when the model should output class labels.
                edge_weights: bool
                    Falg to determine if edge weights are learned.
                multi_scale: bool
                    Falg to determine if the discrimination should consider encoding of multiple
                    scales, i.e. from different layers.
                num_classes: int
                    Number of classes in the data set. This is only needed in mode "classify".
        """

        super().__init__()

        self.mode = mode
        self.multi_scale = multi_scale

        A = torch.tensor(get_normalized_adj(A), dtype=torch.float32).to("cpu")
        self.num_nodes = A.shape[0]
        self.register_buffer('A', A)

        if architecture is None:
            # Fits in 4 GB Memory
            # Default Plain architecture without EW (edge weights) and MS (multi scale)
            architecture = [
                (2, 32, 32, 128),
                (128, 128, 128, 256)
            ]

        # Initial Data Normalization
        in_ch = architecture[0][0]
        self.data_norm = nn.BatchNorm1d(in_ch * A.shape[0])

        self.layers = []
        self.res_layers = []
        self.bn_layers = []
        self.ms_layers = []


        for c_in, c_out, spec_out, out in architecture:

            tempConv1 = TemporalConvolution(c_in=c_in, c_out=c_out, kernel=9)
            specConv = SpectralConvolution(c_in=c_out, c_out=spec_out, A=self.A)
            tempConv2 = TemporalConvolution(c_in=spec_out, c_out=out, kernel=9)

            self.layers.append(nn.Sequential(
                tempConv1,
                nn.BatchNorm2d(c_out),
                nn.LeakyReLU(),
                specConv,
                nn.BatchNorm2d(spec_out),
                nn.Dropout(0.5),
                nn.LeakyReLU(),
                tempConv2,
                nn.LeakyReLU()
            ))

            if c_in == out:
                self.res_layers.append(nn.Identity())
            else:
                self.res_layers.append(nn.Conv2d(c_in, out, kernel_size=1))

            # Final batch normalization after sandwich layer
            self.bn_layers.append(nn.BatchNorm2d(out))

            if multi_scale:
                self.ms_layers.append(nn.Linear(out, architecture[-1][-1]))
            else:
                self.ms_layers.append(None)

        self.convLayers = nn.Sequential(*self.layers)
        self.res_layers = nn.Sequential(*self.res_layers)
        self.bn_layers = nn.Sequential(*self.bn_layers)
        self.ms_layers = nn.Sequential(*self.ms_layers)

        if edge_weights:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size()))
                for i in self.convLayers
            ])
        else:
            self.edge_importance = [1] * len(self.convLayers)

        self.relu = nn.LeakyReLU()

        if self.mode == "classify":
            self.fcn = nn.Conv2d(architecture[-1][-1], num_classes, kernel_size=1)
        else:
            self.fcn = nn.Sequential(nn.Conv2d(out, out, kernel_size=1),
                                     nn.Dropout(0.5),
                                     nn.BatchNorm2d(out))

    # pylint: disable=inconsistent-return-statements
    def forward(self, X: torch.Tensor):
        """ Forward pass.

            Args:
                X: torch.Tensor
            Return:
                torch.Tensor: Output tensor.
        """

        # Split multiple persons in one frame into separate samples
        N, C, T, V, M = X.size()
        X = X.permute(0, 4, 3, 1, 2).contiguous()
        X = X.view(N * M, V * C, T)

        # Normalize the data
        X = self.data_norm(X)
        X = X.view(N, M, V, C, T)
        X = X.permute(0, 1, 3, 4, 2).contiguous()
        X = X.view(N * M, C, T, V)

        X = self.relu(X)

        concat_scales = []

        if not hasattr(self, "edge_importance"):
            self.edge_importance = [nn.Identity()] * len(self.convLayers)
        if not hasattr(self, "ms_layers"):
            self.ms_layers = [nn.Identity()] * len(self.convLayers)

        # Actual forward pass.
        # In: (batch_size, features, time, nodes)
        for layer, res_layer, bn, E, ms_layer in zip(self.convLayers, self.res_layers,
                                                     self.bn_layers, self.edge_importance,
                                                     self.ms_layers):
            res = res_layer(X)
            X = layer[:3](X)
            X = layer[3](X, self.A * E)
            X = layer[4:](X)
            X = bn(X + res)
            X = self.relu(X)

            if self.multi_scale:
                # In: (batch * persons, features, time, nodes)
                # Out:(batch * persons, features, 1, nodes)
                Z = F.avg_pool2d(X, (X.size()[2], 1))
                # In: (batch * persons, features, 1, nodes)
                # Out: (batch, features, nodes)
                Z = torch.squeeze(Z.view(N, M, -1, V).mean(dim=1))
                # In: (batch, features, nodes)
                # Out: (batch, outfeatures, nodes)
                Z = ms_layer(Z.view(-1, Z.shape[1])).view(Z.shape[0], -1, Z.shape[2])

                concat_scales.append(Z)

        if self.mode == "encode":
            # Average over time
            # In: (batch * persons, features, time, nodes)
            # Out:(batch * persons, features, 1, nodes)
            X = self.fcn(X)
            X = F.avg_pool2d(X, (X.size()[2], 1))

            # Average over persons
            # In: (batch, persons, features, 1, nodes)
            # Out:(batch, 1, features, 1, nodes)
            X = X.view(N, M, -1, V).mean(dim=1)

            return torch.squeeze(X.mean(dim=-1)), torch.cat([X, *concat_scales], dim=2)

        elif self.mode == "classify":
            # Average over nodes and time
            # In: (batch * persons, features, time, nodes), Out:(batch * persons, features, 1, 1)
            X = F.avg_pool2d(X, X.size()[2:])

            # Average over persons
            # In: (batch, persons, features, 1, nodes), Out:(batch, 1, features, 1, 1)
            X = X.view(N, M, -1, 1, 1).mean(dim=1)

            # In: (batch, 1, features, 1, 1), Out:(batch, num classes, 1, 1, 1)
            X = self.fcn(X)

            # Squeezing alternative
            return X.view(X.size(0), -1)
