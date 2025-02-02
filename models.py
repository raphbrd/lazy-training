""" Model architectures used in the numerical experiments 

@author: Raphael Bordas

References:

    - Geiger, M., Spigler, S., Jacot, A., & Wyart, M. (2020). Disentangling feature and lazy training in deep neural networks. Journal of Statistical Mechanics: Theory and Experiment, 2020(11), 113301.
"""
import torch
import torch.nn as nn


class LinearLayer(nn.Module):
    """ Modified version of a Linear layer to match the architecture described in Geiger et al. 2020 """

    def __init__(self, input_dim, h, factor=1, activation=None, bias=False, std_init=1):
        """ Modified linear layer (as described in Geiger et al. 2020)
        
        Initialization of the weights follows a Gaussian distribution N(0, std_init**2).

        Parameters
        ----------
        input_dim: int
            dimension of the input (d in the paper)
        h: int
            width of the layer (number of neurons)
        factor: float
            factor to multiply the output of the layer. Default is 1.
        activation: torch.nn.Module
            activation function (sigma in the paper). If None, no activation is performed.
        bias: bool
            whether to use bias in the layer. Default is False (identical to the paper)
        std_init: float
            standard deviation of the Gaussian initialization distribution. Default is 1.
        """
        super().__init__()

        self.layer = nn.Linear(in_features=input_dim, out_features=h, bias=bias)
        self.activation = activation
        self.factor = factor
        self.bias = bias

        self.init_weights(std_init)

    def init_weights(self, std_init):
        nn.init.normal_(self.layer.weight, mean=0, std=std_init)
        if self.bias:
            nn.init.zeros_(self.layer.bias)

    def forward(self, x):
        x = self.factor * self.layer(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class FCNet(nn.Module):
    """ Fully connected neural netword using modified linear layers from Geiger et al. 2020 """

    def __init__(self, input_dim, h, L, activation, bias=False, std_init=1, device="cpu", copy_initial_params=True):
        """ Initialize a fully connected network 
        
        Parameters
        ----------
        input_dim: int
            dimension of the input (d in the paper)
        h: int
            width of the layers (number of neurons). The width is assumed to be constant across all layers
        L: int
            number of hidden layers
        activation: torch.nn.Module
            activation function (sigma in the paper)
        bias: bool
            whether to use bias in the layer. Default is False (identical to the paper)
        std_init: float
            standard deviation of the Gaussian initialisation distribution. Default is 1
        """
        super().__init__()

        # most inputs will be images, which are flatten before the network
        self.flatten = nn.Flatten()

        layers = list()
        for _ in range(L):
            layers.append(LinearLayer(input_dim, h, (1 / input_dim) ** 0.5, activation, bias, std_init))
            input_dim = h

        # last layer handles the output format (dimension = 1)
        layers.append(LinearLayer(h, 1, factor=1 / input_dim, activation=None, bias=bias, std_init=std_init))
        self.layers = nn.Sequential(*layers)

        if copy_initial_params:
            self.initial_params = {name: p.clone().to(device) for name, p in self.named_parameters()}

    def forward(self, x):
        x = self.flatten(x)
        return self.layers(x)
