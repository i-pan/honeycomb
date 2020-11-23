import torch
import torch.nn as nn

from timm.models import resnest
from timm.models.resnest import ResNet, ResNestBottleneck, default_cfgs

from ._base import EncoderMixin
from . import _utils as utils

class ResNestEncoder(ResNet, EncoderMixin):

    def __init__(self, out_channels, depth=5, **kwargs):
        super().__init__(**kwargs)
        self._out_channels = out_channels
        self._in_channels = 3
        self._depth = depth

        del self.fc
        del self.global_pool

    def get_stages(self):
        return [
            nn.Identity(),
            nn.Sequential(self.conv1, self.bn1, self.act1),
            nn.Sequential(self.maxpool, self.layer1),
            self.layer2,
            self.layer3,
            self.layer4,
        ]

    def forward(self, x):
        stages = self.get_stages()

        features = []
        for i in range(self._depth + 1):
            x = stages[i](x)
            features.append(x)

        return features

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop("fc.bias", None)
        state_dict.pop("fc.weight", None)
        super().load_state_dict(state_dict, **kwargs)

    def make_dilated(self, stage_list, dilation_list, multi_grid):
        stages = self.get_stages()
        for stage_indx, dilation_rate in zip(stage_list, dilation_list):
            if len(stages[stage_indx]) == 3:
                for mg_ind, mg in enumerate(multi_grid):
                    utils.replace_strides_with_dilation(
                        module=stages[stage_indx][mg_ind],
                        dilation_rate=dilation_rate*mg,
                    )
            elif len(stages[stage_indx]) == 2 and stage_indx > 4:
                for mg_ind, mg in enumerate(multi_grid[:2]):
                    utils.replace_strides_with_dilation(
                        module=stages[stage_indx][mg_ind],
                        dilation_rate=dilation_rate*mg,
                    )                
            else:
                utils.replace_strides_with_dilation(
                    module=stages[stage_indx],
                    dilation_rate=dilation_rate,
                )
            for module in stages[stage_indx]:
                if hasattr(module, 'avd_first') and module.avd_first:
                    module.avd_first = nn.Identity()

                if hasattr(module, 'downsample') and module.downsample:
                    module.downsample[0] = nn.Identity()


def prepare_settings(settings):
    return {
        "mean": settings["mean"],
        "std": settings["std"],
        "url": settings["url"],
        "input_range": (0, 1),
        "input_space": "RGB",
    }


resnest_settings = {
    "block": ResNestBottleneck,
    "stem_type": "deep",
    "avg_down": True,
    "base_width": 64,
    "cardinality": 1,
    "block_args": {
        "radix": 2,
        "avd": True,
        "avd_first": True
    },
    "out_channels": (3, 64, 256, 512, 1024, 2048)
}

resnest_encoders = {

    "resnest14": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest14d"]),
        },
        "params": {**resnest_settings, 
                "layers": [1, 1, 1, 1],
                "stem_width": 32
            },
    },

    "resnest26": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest26d"]),
        },
        "params": {**resnest_settings, 
                "layers": [2, 2, 2, 2],
                "stem_width": 32
            },
    },

    "resnest50": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest50d"]),
        },
        "params": {**resnest_settings, 
                "layers": [3, 4, 6, 3],
                "stem_width": 32
            },
    },

    "resnest101": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest101e"]),
        },
        "params": {**resnest_settings, 
                "layers": [3, 4, 23, 3],
                "stem_width": 64
            },
    },

    "resnest200": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest200e"]),
        },
        "params": {**resnest_settings, 
                "layers": [3, 24, 36, 3],
                "stem_width": 64
            },
    },

    "resnest269": {
        "encoder": ResNestEncoder,
        "pretrained_settings": {
            "imagenet": prepare_settings(default_cfgs["resnest269e"]),
        },
        "params": {**resnest_settings, 
                "layers": [3, 40, 48, 8],
                "stem_width": 64
            },
    },
}
