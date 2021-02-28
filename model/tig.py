""" Implementation of a the Temporal Info Graph.

    @author: j-huthmacher
"""
from typing import Any
from math import floor

import torch
import torch.nn as nn
import torch.nn.functional as F

from data.data_utils import get_normalized_adj, pad_zero_mask, pad_zero_idx

from model.st_gcn_aaai18.stgcn_graph import Graph

class ConstZeroLayer(torch.nn.Module):
    def forward(self, x):
        return torch.tensor([0])

class FF(nn.Module):
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

    def __init__(self, c_in: int, c_out: int = 1, kernel: Any = 1):

        super().__init__()

        # Set paramters
        self.c_in = c_in
        self.c_out = c_out
        self.kernel = kernel

        if isinstance(self.kernel, int):
            self.kernel = (1, self.kernel)
        
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

        #### TO DEVICE #####
        # self.conv = self.conv.to(self.device)

        self.relu = nn.LeakyReLU()

    def forward(self, X: torch.Tensor):

        # In: (batch, features, time, nodes)
        X = self.tcn(X)

        return X


class SpectralConvolution(nn.Module):
    def __init__(self, c_in: int , c_out: int = 2, activation: str = "LeakyReLU",A = None):
        super().__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.A = A

        self.conv = nn.Conv2d(self.c_in,
                              self.c_out,
                              kernel_size=(1, 1))

        self.activation = getattr(nn, activation)()

    def forward(self, X: torch.Tensor, A: torch.Tensor = None):

        if A is None:
            A = self.A

        X = self.conv(X)

        X = torch.einsum('nctv,vw->nctw', (X, A))

        return X.contiguous()


class TemporalInfoGraph(nn.Module):
    def __init__(self, architecture: list = None, A: torch.Tensor = None,
                 mode: str = "encode", edge_weights: bool = True,
                 multi_scale: bool = False, num_classes: int=50):

        super().__init__()

        self.mode = mode
        self.multi_scale = multi_scale

        A = torch.tensor(get_normalized_adj(A), dtype=torch.float32).to("cuda")
        self.num_nodes = A.shape[0]
        self.register_buffer('A', A)

        #### Initial Data Normalization ####
        self.data_norm = nn.BatchNorm1d(2*18)

        self.layers = []
        self.res_layers = []
        self.bn_layers = []
        self.ms_layers = []

        if architecture is None:
            # Fits in 4 GB Memory
            architecture = [
                (2, 32, 64, 64),
                (64, 128, 128, 256)
            ]

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
            self.fcn = nn.Conv2d(256, num_classes, kernel_size=1)
        else:
            self.fcn = nn.Conv2d(out, out, kernel_size=1)

    def forward(self, X: torch.Tensor):

        N, C, T, V, M = X.size()
        X = X.permute(0, 4, 3, 1, 2).contiguous()
        X = X.view(N * M, V * C, T)
        X = self.data_norm(X)
        X = X.view(N, M, V, C, T)
        X = X.permute(0, 1, 3, 4, 2).contiguous()
        X = X.view(N * M, C, T, V)

        X = self.relu(X)

        concat_scales = []

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
                # In: (batch * persons, features, time, nodes), Out:(batch * persons, features, 1, nodes)
                Z = F.avg_pool2d(X, (X.size()[2], 1))
                # In: (batch * persons, features, 1, nodes), Out: (batch, features, nodes)
                Z = torch.squeeze(Z.view(N, M, -1, V).mean(dim=1))
                # In: (batch, features, nodes), Out: (batch, outfeatures, nodes)
                Z = ms_layer(Z.view(-1, Z.shape[1])).view(Z.shape[0], -1 ,Z.shape[2])

                concat_scales.append(Z)

        if self.mode == "encode":
            # Average over time
            # In: (batch * persons, features, time, nodes), Out:(batch * persons, features, 1, nodes)
            X = self.fcn(X)
            X = F.avg_pool2d(X, (X.size()[2], 1))
            # Average over persons
            # In: (batch, persons, features, 1, nodes), Out:(batch, 1, features, 1, nodes)
            X = X.view(N, M, -1, V).mean(dim=1)

            return torch.squeeze(X.mean(dim=-1)), torch.cat([X, *concat_scales], dim=2) #torch.squeeze(X)

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

