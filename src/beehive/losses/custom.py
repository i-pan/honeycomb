import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.loss import _WeightedLoss
from torch.autograd import Variable


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):

    def forward(self, p, t):
        return F.binary_cross_entropy_with_logits(p.float(), t.float())


class CrossEntropyLoss(nn.CrossEntropyLoss):

    def forward(self, p, t):
        t = t.view(-1)
        if self.weight:
            return F.cross_entropy(p.float(), t.long(), weight=self.weight.float().to(t.device))
        else:
            return F.cross_entropy(p.float(), t.long())


class OneHotCrossEntropy(_WeightedLoss):

    def __init__(self, weight=None, reduction='mean'):
        super().__init__(weight=weight, reduction=reduction)
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        lsm = F.log_softmax(inputs, -1)

        if self.weight is not None:
            lsm = lsm * self.weight.unsqueeze(0)

        loss = -(targets * lsm).sum(-1)

        if  self.reduction == 'sum':
            loss = loss.sum()
        elif  self.reduction == 'mean':
            loss = loss.mean()

        return loss


class SmoothCrossEntropy(nn.Module):
    
    # From https://www.kaggle.com/shonenkov/train-inference-gpu-baseline
    def __init__(self, smoothing = 0.05):
        super().__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = F.one_hot(target.long(), x.size(1))
            target = target.float()
            logprobs = F.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)
    
            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return F.cross_entropy(x, target.long())


class MixCrossEntropy(nn.Module):

    def forward_train(self, p, t):
        lam = t['lam']
        loss1 = F.cross_entropy(p.float(), t['y_true1'].long(), reduction='none')
        loss2 = F.cross_entropy(p.float(), t['y_true2'].long(), reduction='none')
        loss = lam*loss1 + (1-lam)*loss2
        return loss.mean()

    def forward(self, p, t):
        if isinstance(t, dict) and 'lam' in t.keys():
            return self.forward_train(p, t)
        else:
            return F.cross_entropy(p.float(), t.long())


class DenseCrossEntropy(nn.Module):

    def forward(self, x, target):
        x = x.float()
        target = target.float()
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        loss = -logprobs * target
        loss = loss.sum(-1)
        return loss.mean()


class ArcFaceLoss(nn.Module):

    def __init__(self, s=30.0, m=0.5):
        super().__init__()
        self.crit = DenseCrossEntropy()
        self.s = s
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, logits, labels):
        labels = F.one_hot(labels.long(), logits.size(1)).float().to(labels.device)
        logits = logits.float()
        cosine = logits
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        output = (labels * phi) + ((1.0 - labels) * cosine)
        output *= self.s
        loss = self.crit(output, labels)
        return loss
