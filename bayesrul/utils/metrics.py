"""
Metrics for evaluation on the test set
See torchmetrics (MetricTracker)
"""

import torch
from math import sqrt


# torch.distributions.normal.cdf ?
def normal_cdf(x, loc, scale):
    return 0.5 * (1 + torch.erf((x - loc) * scale.reciprocal() / sqrt(2)))


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


