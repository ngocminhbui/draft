import torch.nn.functional as F
import pdb
import torch

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output,target):
    return F.cross_entropy(output, target)

def cross_entropy_loss_multi(output, target):
    x_voted = output[0]
    x = output[1]

    sum_loss_from_rings = 0
    x = torch.transpose(x, 1,0)
    #import ipdb; ipdb.set_trace()
    for i in range(x.shape[0]):
        sum_loss_from_rings += cross_entropy_loss(x[i], target)
    #ipdb.set_trace()
    return cross_entropy_loss(x_voted,target) + sum_loss_from_rings