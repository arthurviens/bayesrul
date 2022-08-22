"""
Metrics for evaluation on the test set
"""

import torch
from math import sqrt
from torch.distributions import Normal
from bayesrul.utils.miscellaneous import assert_same_shapes

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

def get_proportion_lists(y_pred, y_std, y_true, num_bins, prop_type='interval'):
    exp_proportions = torch.linspace(0, 1, num_bins, device=y_true.get_device())

    residuals = y_pred - y_true
    normalized_residuals = (residuals.flatten() / y_std.flatten()).reshape(-1, 1)
    dist = Normal(
        torch.tensor([0.0], device=y_true.get_device()),
        torch.tensor([1.0], device=y_true.get_device()),
        )
    if prop_type == 'interval':
        gaussian_lower_bound = dist.icdf(0.5 - exp_proportions / 2.0)
        gaussian_upper_bound = dist.icdf(0.5 + exp_proportions / 2.0)
        
        above_lower = (normalized_residuals >= gaussian_lower_bound)
        below_upper = (normalized_residuals <= gaussian_upper_bound)

        within_quantile = above_lower * below_upper
        obs_proportions = torch.sum(within_quantile, axis=0).flatten() / len(residuals)
    elif prop_type == 'quantile':
        gaussian_quantile_bound = dist.icdf(exp_proportions)
        below_quantile = (normalized_residuals <= gaussian_quantile_bound)
        obs_proportions = torch.sum(below_quantile, axis=0).flatten() / len(residuals)

    return exp_proportions, obs_proportions


def rms_calibration_error(y_pred, y_std, y_true, num_bins=100, prop_type='interval'):
    
    assert_same_shapes(y_pred, y_std, y_true)
    assert y_std.min() >= 0, "Not all values are positive"
    assert prop_type in ['interval', 'quantile']

    exp_props, obs_props = get_proportion_lists(
        y_pred, y_std, y_true, num_bins, prop_type
    )
    
    squared_diff_props = torch.square(exp_props - obs_props)
    rmsce = torch.sqrt(torch.mean(squared_diff_props))

    return rmsce


def nasa_scoring_function(y_true, y_pred):
    diff = y_true - y_pred 
    alpha = torch.zeros_like(diff)
    mask = (diff <= 0)
    alpha = torch.where(mask, 1/8, 1/5)

    return (torch.exp(alpha * torch.abs(diff))).mean()
