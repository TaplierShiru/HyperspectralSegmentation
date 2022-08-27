import torch


def scale_invariant_loss(x, y):
    n = x.shape[0]
    d = torch.log(x) - torch.log(y)
    loss = 1. / n * torch.sum(d ** 2) - 1. / (2 * n ** 2) * (torch.sum(d)) ** 2
    return loss
