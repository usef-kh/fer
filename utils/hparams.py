import os

hps = {
    'network': '',  # which network do you want to train
    'name': '',  # whatever you want to name your run
    'n_epochs': 300,
    'model_save_dir': None,  # where will checkpoints be stored (path created automatically using hps[name])
    'restore_epoch': None,  # continue training from a specific saved point
    'start_epoch': 0,
    'lr': 0.01,  # starting learning rate
    'save_freq': 20,  # how often to create checkpoints
    'drop': 0.1,
    'bs': 64,
}

possible_nets = set(filename.split(".")[0] for filename in os.listdir('models'))


def setup_hparams(args):
    for arg in args:
        key, value = arg.split('=')
        if key not in hps:
            raise ValueError(key + ' is not a valid hyper parameter')
        else:
            hps[key] = value

    # Invalid network check
    if hps['network'] not in possible_nets:
        raise ValueError("Invalid network.\nPossible ones include:\n - " + '\n - '.join(possible_nets))

    # invalid parameter check
    try:
        hps['n_epochs'] = int(hps['n_epochs'])
        hps['start_epoch'] = int(hps['start_epoch'])
        hps['save_freq'] = int(hps['save_freq'])
        hps['lr'] = float(hps['lr'])
        hps['drop'] = float(hps['drop'])
        hps['bs'] = int(hps['bs'])

        if hps['restore_epoch']:
            hps['restore_epoch'] = int(hps['restore_epoch'])
            hps['start_epoch'] = int(hps['restore_epoch'])

        # make sure we can checkpoint regularly or at least once (at the end)
        if hps['n_epochs'] < 20:
            hps['save_freq'] = min(5, hps['n_epochs'])

    except Exception as e:
        raise ValueError("Invalid input parameters")

    # create checkpoint directory
    hps['model_save_dir'] = os.path.join(os.getcwd(), 'checkpoints', hps['name'])

    if not os.path.exists(hps['model_save_dir']):
        os.makedirs(hps['model_save_dir'])

    return hps
