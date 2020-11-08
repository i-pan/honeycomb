import timm
import torch.nn as nn


class Identity(nn.Module): 

    def forward(self, x): return x


##########
# MixNet #
##########
def mixnet_s(pretrained=True, **kwargs):
    model = timm.models.mixnet_s(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_m(pretrained=True, **kwargs):
    model = timm.models.mixnet_m(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_l(pretrained=True, **kwargs):
    model = timm.models.mixnet_l(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def mixnet_xl(pretrained=True, **kwargs):
    model = timm.models.mixnet_xl(pretrained=pretrained)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


###########
# ResNeSt #
###########
def resnest14(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest14d(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest26(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest26d(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest50(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest50d(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest101(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest101e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest200(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest200e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


def resnest269(pretrained=True, **kwargs):
    model = timm.models.resnest.resnest269e(pretrained=pretrained)
    dim_feats = model.fc.in_features
    model.fc = Identity()
    return model, dim_feats


################
# EfficientNet #
################
def efficientnet_b0(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b0(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b1_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b1(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b1(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b2_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b2(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b2(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b3_pruned(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3_pruned(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def efficientnet_b3(pretrained=True, **kwargs):
    model = timm.models.efficientnet_b3(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b4(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b4(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b5(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b5(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b6(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b6_ns(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b6_ns(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b7(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b7(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_b8(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_b8(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats


def tf_efficientnet_l2(pretrained=True, **kwargs):
    model = timm.models.tf_efficientnet_l2_ns(pretrained=pretrained, **kwargs)
    dim_feats = model.classifier.in_features
    model.classifier = Identity()
    return model, dim_feats