"""
    @author: jhuthmacher
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Primitive MLP for downstream classification
    """

    def __init__(self, in_dim, num_class, encoder: nn.Module):
        """ Initialization of the downstream MLP. Hidden dimension 128.

            Paramters:
                in_dim: int
                    Number of input features. Has to match the representation dimension.
                num_class: int
                    Number of classes. This defines the output dim of the model.
                encoder: nn.Module
                    Encoder to create hidden representation. To not retrain the encoder the 
                    parameters are disabled for the gradient flow.
        """
        super(MLP, self).__init__()

        for p in encoder.parameters():
            p.requires_grad = False

        self.encoder = encoder

        self.layers = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LeakyReLU(),
            nn.Linear(128, num_class),
            nn.Softmax()
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """ Forward function

            Parameters:
                x: torch.Tensor
                    Input of dimension (batch, emb_dim)
            Return:
                torch.Tensor: Dimension (batch, num_class)
        """
        x_enc, _ = self.encoder(x, A)

        x = self.layers(x_enc)

        return x