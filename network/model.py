import torch
from .unet import UNet
from .unet_gray import UNet_gray
from .modeling.deeplab import *

def _load_model():
    raise NotImplementedError

def unet_rgb(channel=3, num_classes=2):
    print('UNet RGB - Channel: {} Classes: {}'.format(channel, num_classes))
    return UNet(n_channels=channel, n_classes=num_classes)

def unet_gray(channel=1, num_classes=2):
    print('UNet GRAY - Channel: {} Classes: {}'.format(channel, num_classes))
    return UNet_gray(n_channels=channel, n_classes=num_classes)

# Not Implemented Error
def unet_pt(channel=1, num_classes=2):
    print('UNet Pretrained - Channel: {} Classes: {}'.format(channel, num_classes))
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                    in_channels=1, out_channels=2, init_features=32, pretrained=True)

def deeplab(channel=1, num_classes=2):
    print('DeepLab - Channel: {} Classes: {}'.format(channel, num_classes))
    return DeepLab(num_classes=num_classes, backbone='resnet', output_stride=16,
                    sync_bn=True, freeze_bn=False)


