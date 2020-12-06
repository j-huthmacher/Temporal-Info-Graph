"""
    @author: jhuthmacher
"""
import torch
import torch.nn as nn

class MLP(nn.Module):
    """ Primitive MLP for downstream classification
    """

    def __init__(self, in_dim, num_class, hidden_layers = [128], encoder: nn.Module = None):
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
        super().__init__()

        self.encoder = encoder 
        
        if self.encoder is not None: 
            for p in encoder.parameters():
                p.requires_grad = False

            self.encoder = encoder

        layers = [nn.Linear(in_dim, hidden_layers[0]), nn.ReLU(inplace=True)]

        for hl in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[hl-1], hidden_layers[hl]))
            layers.append(nn.ReLU(inplace=True))
        
        layers.append(nn.Linear(hidden_layers[-1], num_class))
        # layers.append(nn.Softmax(dim=1))  # Normalize over feature dimension.
        # Softmax within CrossEntropy?

        self.layers = nn.Sequential(*layers)

        #### TO DEVICE #####
        if self.encoder is not None: 
            self.encoder = self.encoder.to(self.device)
        self.layers = self.layers.to(self.device)

    @property
    def device(self):
        return next(self.parameters()).device
    
    @property
    def is_cuda(self):
        return next(self.parameters()).is_cuda

    def forward(self, x: torch.Tensor, A: torch.Tensor):
        """ Forward function

            Parameters:
                x: torch.Tensor
                    Input of dimension (batch, emb_dim)
            Return:
                torch.Tensor: Dimension (batch, num_class)
        """
        if self.encoder is not None: 
            x, _ = self.encoder(x, A)

        x = self.layers(x)

        return x