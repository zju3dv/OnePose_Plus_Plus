import math
import torch
from torch import nn
import torch.nn.functional as F


# Position encoding for query image
class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256)):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
        """
        super().__init__()

        max_shape = tuple(max_shape)

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        div_term = torch.exp(
            torch.arange(0, d_model // 2, 2).float()
            * (-math.log(10000.0) / d_model // 2)
        )
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, : x.size(2), : x.size(3)]


# Position encoding for 3D points
class KeypointEncoding_linear(nn.Module):
    """ Joint encoding of visual appearance and location using MLPs """

    def __init__(self, inp_dim, feature_dim, layers, norm_method="batchnorm"):
        super().__init__()
        self.encoder = self.MLP([inp_dim] + list(layers) + [feature_dim], norm_method)
        nn.init.constant_(self.encoder[-1].bias, 0.0)

    def forward(self, kpts, descriptors):
        """
        kpts: B*L*3 or B*L*4
        descriptors: B*C*L
        """
        inputs = kpts  # B*L*3
        return descriptors + self.encoder(inputs).transpose(2, 1)  # B*C*L

    def MLP(self, channels: list, norm_method="batchnorm"):
        """ Multi-layer perceptron"""
        n = len(channels)
        layers = []
        for i in range(1, n):
            layers.append(nn.Linear(channels[i - 1], channels[i], bias=True))
            if i < n - 1:
                if norm_method == "batchnorm":
                    layers.append(nn.BatchNorm1d(channels[i]))
                elif norm_method == "layernorm":
                    layers.append(nn.LayerNorm(channels[i]))
                elif norm_method == "instancenorm":
                    layers.append(nn.InstanceNorm1d(channels[i]))
                else:
                    raise NotImplementedError
                    # layers.append(nn.GroupNorm(channels[i], channels[i])) # group norm
                layers.append(nn.ReLU())
        return nn.Sequential(*layers)