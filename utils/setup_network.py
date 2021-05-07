  
import torch

from models import vgg, efn, vgg2
from utils.checkpoint import load_features, restore
from utils.logger import Logger

nets = {

    'vgg': vgg.Vgg,
    'efn': efn.EfficientNet,
    'vgg2':vgg2.Vgg2
}


def setup_network(hps):

    net = nets[hps['network']]()

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
    