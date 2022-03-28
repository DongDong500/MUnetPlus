from .unet import UNet

def _load_model():
    raise NotImplementedError

def unet_rgb(channel=3, num_classes=2):
    print('UNet RGB - Channel: {} Classes: {}'.format(channel, num_classes))
    return UNet(n_channels=channel, n_classes=num_classes)

def unet_gray():
    raise NotImplementedError