""" Simple MLP for the downstream classification.
"""
# pylint: disable=no-self-use
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Primitive MLP for downstream classification
    """

    def __init__(self, in_dim: int, num_class: int, hidden_layers: [int] = [128],
                 encoder: nn.Module = None):
        """ Initialization of the downstream MLP. Hidden dimension 128.

            Args:
                in_dim: int
                    Number of input features. Has to match the representation dimension.
                num_class: int
                    Number of classes. This defines the output dim of the model.
                hidden_layers: [int]
                    List of integers representing the dimensions and the number of hidden layers.
                encoder: nn.Module
                    Encoder to create hidden representation. To not re-train the encoder the
                    parameters are disabled for the gradient flow.
        """
        super().__init__()

        self.encoder = encoder

        if self.encoder is not None:
            for p in encoder.parameters():
                p.requires_grad = False

            self.encoder = encoder
            self.encoder.eval()

        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.Tanh()]

        for hl in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[hl - 1], hidden_layers[hl]))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_layers[-1], num_class))

        self.layers = nn.Sequential(*layers)

        # To device
        if self.encoder is not None:
            self.encoder = self.encoder.to(self.device)
        self.layers = self.layers.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def init_weights(self, m: nn.Module):
        """ Function to initliaze the weights for better learning.

            Args:
                m: nn.Module
                    The layer for which the weights are initialized.
        """
        # pylint: disable=unidiomatic-typecheck
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.001)

    def forward(self, x: torch.Tensor, A: torch.Tensor = None):
        """ Forward pass.

            Args:
                x: torch.Tensor
                    Input of dimension (batch, emb_dim)
                A: torch.Tensor
                    Adjacency matrix.
            Return:
                torch.Tensor: Dimension (batch, num_class)
        """
        x = x.type('torch.FloatTensor').to(self.device)

        if self.encoder is not None and A is not None:
            with torch.no_grad():
                x = self.encoder(x, A)[0].detach()

        x = self.layers(x)

        return x
