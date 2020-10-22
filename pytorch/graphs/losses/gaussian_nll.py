import math
import torch
import torch.nn as nn

def gaussian_nll(mu, log_sigma, noise):
    NLL = torch.sum(log_sigma, 1) + \
    torch.sum(((noise - mu) / (1e-8 + torch.exp(log_sigma))) ** 2, 1) / 2.
    return NLL.mean()
