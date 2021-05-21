import torch.nn as nn

from . import base
from . import functional as F
from  .base import Activation
import torch

class JaccardLoss(base.Loss):

    def __init__(self, eps=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.jaccard(
            y_pr, y_gt,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )

class DiceLoss(base.Loss):

    def __init__(self, eps=1., beta=1., activation=None, ignore_channels=None, **kwargs):
        super().__init__(**kwargs)
        self.eps = eps
        self.beta = beta
        self.activation = Activation(activation)
        self.ignore_channels = ignore_channels

    def forward(self, y_pr, y_gt):
        y_pr = self.activation(y_pr)
        return 1 - F.f_score(
            y_pr, y_gt,
            beta=self.beta,
            eps=self.eps,
            threshold=None,
            ignore_channels=self.ignore_channels,
        )


#######
##

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn import Parameter, Module

class _MSEloss(nn.MSELoss, base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.MSELoss = nn.MSELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.MSELoss(input, target)

class _L1loss(nn.L1Loss, base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.L1Loss = nn.L1Loss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.L1Loss(input, target)

class _BCrossEntropyloss(nn.BCELoss, base.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.BCELoss = nn.BCELoss()

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.BCELoss(input, target)

class MultiTaskLoss(base.Loss, nn.Module):
    """
    https://github.com/oxcsaml2019/multitask-learning/blob/master/multitask-learning/losses.py
    """
    def __init__(self, model, **kwargs): #task1 -> SEG, task2 -> BOUNDARY, task3 -> RECONSTRUCTION
        super().__init__(**kwargs)
        self.model = model
        self.BCE = _BCrossEntropyloss()
        self.L1 = _L1loss()
        print("MTL_learnable: Segmentation + Boundary + Reconstruction")

    def forward(self, targets):
       
       model = self.model 
       segmentation_mask, edge_mask, reconstruction_mask, self.sigma = model.forward(targets[-1])
       
       l1_bce = self.BCE(segmentation_mask, targets[0]) #*2
       l2_bce= self.BCE(edge_mask, targets[1]) #*2
       l3_l1 = self.L1(reconstruction_mask,targets[2])
    
       precision1 = torch.exp(-self.sigma[0])
       loss = torch.sum(precision1 * l1_bce + (self.sigma[0] * self.sigma[0]) , -1)

       precision2 = torch.exp( -self.sigma[1])
       loss += torch.sum(precision2 * l2_bce + (self.sigma[1] * self.sigma[1]), -1)

       precision3 = torch.exp(-self.sigma[2])
       loss += torch.sum(precision3 * l3_l1 + (self.sigma[2] * self.sigma[2]), -1)

       loss = torch.mean(loss)
       
       return loss, segmentation_mask, edge_mask, reconstruction_mask#, self.sigma.data.tolist()



