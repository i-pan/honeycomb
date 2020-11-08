import torch
import torch.nn as nn
import torch.nn.functional as F

from . import backbones
from .constants import POOLING
from .pooling import GeM, AdaptiveConcatPool2d, AdaptiveAvgPool2d, AdaptiveMaxPool2d


POOL2D_LAYERS = {
    'gem': GeM(p=3.0, dim=2),
    'concat': AdaptiveConcatPool2d(),
    'avg': AdaptiveAvgPool2d(1),
    'max': AdaptiveMaxPool2d(1),
}


class Net2D(nn.Module):

    def __init__(self,
                 backbone,
                 pretrained,
                 num_classes,
                 dropout,
                 backbone_params={},
                 multisample_dropout=False,
                 pool='gem'):
        super().__init__()
        bb = getattr(backbones, backbone)
        self.backbone, dim_feats = bb(pretrained=pretrained, **backbone_params)
        pool_layer = POOLING[backbone]
        setattr(self.backbone, pool_layer, POOL2D_LAYERS[pool])
        self.ms_dropout = multisample_dropout
        self.dropout = nn.Dropout(p=dropout)
        self.linear = nn.Linear(dim_feats, num_classes)

    def forward(self, x):
        features = self.backbone(x)
        features = features.view(features.shape[:2])
        if self.ms_dropout:
            x = torch.mean(
                torch.stack(
                    [self.linear(self.dropout(features)) for _ in range(5)],
                    dim=0,
                ),
                dim=0,
            )
        else:
            x = self.linear(self.dropout(features))
        return x[:,0] if self.linear.out_features == 1 else x



