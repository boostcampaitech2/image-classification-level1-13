import torch.nn.functional as F
import torch.nn as N

def nll_loss(output, target):
    return F.nll_loss(output, target)

def CrossEntropyLoss():
    return N.CrossEntropyLoss()