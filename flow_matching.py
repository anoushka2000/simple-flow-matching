#!/usr/bin/env python

# A minimal example of Flow Matching with Optimal Transport
# Adapted from https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa
# Based on Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, and Matt Le. Flow matching for generative modeling, 2023.
# URL https://arxiv.org/abs/2210.02747.

import torch
import torch.nn as nn

from torch import Tensor
from typing import List
from zuko.utils import odeint


class VectorField(nn.Sequential):
    """
    MLP approximating the vector field i.e, v_{\theta}(t,x)

    Parameters
    __________
    in_features: int
        Size of input (i.e. data dimension + 1 channel for time)
    out_features: int
        Size of output (i.e. data dimension + 1 channel for time)
    """
    def __init__(
        self,
        in_features: int,
        out_features: int,
        hidden_features: List[int] = [64, 64],
    ):
        layers = []

        for a, b in zip(
            [in_features] + hidden_features,
            hidden_features + [out_features],
        ):
            layers.extend([nn.Linear(a, b), nn.ELU()])

        super().__init__(*layers[:-1])


class CNF(nn.Module):
    def __init__(
        self,
        data_dim: int,
        **kwargs,
    ):
        super().__init__()
        # data_dim + 1 channel for time
        self.net = VectorField(1 + data_dim, data_dim, **kwargs)


    def forward(self, t: Tensor, x: Tensor) -> Tensor:
        """
        Predict vector field.

        Parameters
        __________
        t: Tensor
            Time steps.
        x: Tensor
            Batch of locations in data space.    
        """
        t = t[..., None]
        t = t.expand(*x.shape[:-1], -1)
        return self.net(torch.cat((t, x), dim=-1))

    def encode(self, x0: Tensor) -> Tensor:
        """
        Integrate from data (q) to source (p0) distribution.
        """
        return odeint(f=self, x=x0, t0=1.0, t1=0.0, phi=self.parameters())

    def decode(self, z: Tensor) -> Tensor:
        """
        Integrate from noise (p0) to approximated
        data (p1) distribution.
        """
        return odeint(f=self, x=z, t0=0.0, t1=1.0, phi=self.parameters())


class FlowMatchingLoss(nn.Module):
    """
    Optimal Transport (OT) flow matching loss.

    Parameters
    __________
    flow: nn.Module
        Continuous normalizing flow model.
    """

    def __init__(self, velocity_field: nn.Module):
        super().__init__()
        self.velocity_field = velocity_field
        self.sigma_min = 1e-4

    def forward(self, x1: Tensor) -> Tensor:
        """
        Compute Optimal Transport training objective
        for conditional flow matching.
        
        Parameters
        __________
        x1: Tensor
            Batch of training data samples to condition on.
        """

        # sample time t ~ U([0,1])
        t = torch.rand_like(x1[..., 0]).unsqueeze(-1)
        
        # sample noise x0 ~ p0(x0) 
        # where p0 is a standard normal 
        x0 = torch.randn_like(x1)

        # Equation 20
        mu_t_x = t*x1 
        sigma_t_x = (1 - (1 - self.sigma_min) * t)

        # Equation 21
        phi_t_x0 =  sigma_t_x * x0 + mu_t_x 

        # evaluate expectation of L2 norm
        # Equation 22
        u_t = x1 - (1 - self.sigma_min) * x0
        return (self.velocity_field(t.squeeze(-1), phi_t_x0) - u_t).square().mean()
