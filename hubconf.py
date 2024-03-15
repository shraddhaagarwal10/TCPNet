dependencies = ['torch']
from unet_data_unc import UNet as TCPNet
from bayesian_crossentropy_loss import bayesian_categorical_crossentropy as bcc


def bayesian_categorical_crossentropy(*args, **kwargs):
    return bcc(*args, **kwargs)















