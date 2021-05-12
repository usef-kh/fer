from models import vgg, efn
from utils.checkpoint import restore
from utils.logger import Logger

nets = {
    'vgg': vgg.Vgg,
    'efn': efn.EfficientNet
}


def setup_network(hps):
    net = nets[hps['network']]()

    # Prepare logger
    logger = Logger()
    if hps['restore_epoch']:
        restore(net, logger, hps)

    return logger, net
