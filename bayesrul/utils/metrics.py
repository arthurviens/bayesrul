"""
Metrics for evaluation on the test set
"""

import torch
import numpy as np
from math import sqrt


# torch.distributions.normal.cdf ?
def normal_cdf(x, loc, scale):
    return 0.5 * (1 + torch.erf(((x - loc) * scale.reciprocal()) / sqrt(2)))


def p_alphalamba(y_true, y_hat, sigma_hat, alpha=0.05, reduction='mean'):
    a = normal_cdf((1+alpha) * y_true, y_hat, sigma_hat)
    b = normal_cdf((1-alpha) * y_true, y_hat, sigma_hat)
    
    if reduction == 'none':
        return a - b
    elif reduction == 'mean':
        return (a - b).mean()
    elif reduction == 'sum':
        return (a - b).sum()
    else:
        raise RuntimeError(f"Unknown reduction {reduction}.")


def PICP(y_true, y_hat, sigma_hat, sigma_level=1, reduction='mean'):
    assert y_true.shape == y_hat.shape, f"Different shapes {y_true.shape} {y_hat.shape}"
    assert y_true.shape == sigma_hat.shape, f"Different shapes {y_true.shape} {sigma_hat.shape}"
    y = torch.abs(y_true - y_hat)
    y_in = (y <= sigma_level * sigma_hat).float()

    if reduction == 'none':
        return y_in
    elif reduction == 'mean':
        return y_in.mean()
    elif reduction == 'sum':
        return y_in.sum()
    else:
        raise RuntimeError(f"Unknown reduction {reduction}.")


def MPIW(sigma_hat, y_true = None, normalized = False):
    if normalized:
        if y_true is None:
            raise RuntimeError("Provide y_true to normalize MPIW")
        R = y_true.max() - y_true.min()
    else: 
        R = 1
    
    return ((sigma_hat.sum()) * 2) / (len(sigma_hat) * R)
    
# For Hetroscedastic NNs, possibility to explore
# Hua Zhong, Li Xu: An All-Batch Loss for Constructing Prediction Intervals
# Quality Driven Loss

