""" Alternative implementation of the TIG by using an LSTM to process the time dimension.
"""
# pylint: disable=not-callable
import torch
from torch import nn

from model.tig import SpectralConvolution
from utils.data_utils import get_normalized_adj


class TemporalInfoGraphLSTM(nn.Module):
    """ Implementation of the temporal info graph model using a LSTM to model
        the time component.
    """

    def __init__(self, ch_in: int = 2, hidden_size: int = 64, num_layers: int = 2,
                 batch_norm: bool = True, A: torch.Tensor = None,
                 self_connection: bool = True):
        """ Initilization of the TIG model.

            Args:
                c_in: int
                    Input channels of the data, i.e. features.
                hidden_size: int
                    Number of hidden channels in the LSTM.
                num_layers: int
                    Number of layers of the LSTM.
                batch_norm: bool
                    Falg to determine if batch norm is applied.
                A: torch.Tensor
                    Adjacency matrix.
                self_connection: bool
                    Falg to determine if self-connections are used in the architecture.
        """
        super().__init__()

        # Model Creatio
        self.layers = []
        self.edge_weights = nn.ParameterList()

        if A is not None and self_connection:
            A = get_normalized_adj(A)
            self.register_buffer('A', torch.tensor(A))

        #### Initial Data Normalization ####
        if batch_norm and A is not None:
            # Features * nodes
            self.data_norm = nn.BatchNorm1d(ch_in * A.shape[0])

        self.gcn = SpectralConvolution(c_in=ch_in, c_out=64)
        self.gcn2 = SpectralConvolution(c_in=64, c_out=64)

        ch_in = ch_in + 64 + 64

        self.lstm = nn.LSTM(ch_in, hidden_size, num_layers)

        self.readout = nn.Sequential(nn.Linear(A.shape[0], 512), nn.Linear(512, 1))

    def forward(self, X: torch.Tensor):
        """ Forward function of the temporl info graph model.

            Consists of a initial temporal convolution, followed by an spectral
            convolution and finalized with another temporal convolution.

            Args:
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
        # In:  (batch, features, nodes, time)
        # Out: (time, batch * nodes, features)
        N, C, V, T = X.shape
        X = X.type('torch.FloatTensor').to("cuda")

        #### Data Normalization ####
        if hasattr(self, "data_norm"):
            X = X.reshape(N, V * C, T)  # Flatten to create large feature matrix
            X = self.data_norm(X)
            # Convert back to original format
            X = X.reshape(N, C, V, T)

        Z1 = self.gcn(X, self.A)
        Z2 = self.gcn2(Z1, self.A)

        # Concatenate the raw input, 1-hop convolution, and 2-hop convolution
        X = torch.cat([X, Z1, Z2], dim=1)
        N, C, V, T = X.shape
        # In:  (batch, cat_features, nodes, time), Out: (time, batch * nodes, cat_features)
        X = X.permute(3, 0, 2, 1).reshape(T, N * V, C)

        Z, _ = self.lstm(X)

        # Only consider the last prediction. We are not interested in each time step.
        Z = Z[-1].view(N, -1, V)

        # In: (batch_size, ch_out, nodes) Out: (batch_size, ch_out)
        global_Z = self.readout(Z.permute(0, 2, 1).reshape(-1, Z.shape[2])).view(Z.shape[0], -1)
        local_Z = Z

        # Remove "empty" dimensions, i.e. dim = 1
        return torch.squeeze(global_Z, dim=-1), torch.squeeze(local_Z, dim=-1)
