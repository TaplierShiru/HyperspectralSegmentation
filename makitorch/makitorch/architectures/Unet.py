import torch
import torch.nn as nn
import torch.nn.functional as F


def Unet(in_channels=3, out_channels=1, init_features=32, pretrained=True):
    return torch.hub.load(
        'mateuszbuda/brain-segmentation-pytorch', 
        'unet', 
        in_channels=in_channels,
        out_channels=out_channels,
        init_features=init_features,
        pretrained=pretrained
    )


class UnetWithFeatureSelection(nn.Module):

    def __init__(self, in_channels=3, out_channels=1, init_features=32, pretrained=True):
        super().__init__()
        self.features_selection = nn.Conv2d(
            in_channels=in_channels,
            out_channels=in_channels,
            groups=in_channels,
            kernel_size=1,
            stride=1,
            bias=False,
        )
        self.unet = torch.hub.load(
            'mateuszbuda/brain-segmentation-pytorch', 
            'unet', 
            in_channels=in_channels,
            out_channels=out_channels,
            init_features=init_features,
            pretrained=pretrained
        )

    def forward(self, x):
        out = self.features_selection(x)
        logits = self.unet(out)

        fs_weights = self.features_selection.weight
        return fs_weights, F.sigmoid(logits)
