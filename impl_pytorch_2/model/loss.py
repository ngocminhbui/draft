import torch.nn.functional as F
import pdb

def nll_loss(output, target):
    return F.nll_loss(output, target)

def cross_entropy_loss(output,target):
    return F.cross_entropy(output, target) 