import argparse
import os
import sys
from ast import literal_eval

import numpy as np
import yaml


def generate_parser():
    # experimental parameters
    parser = argparse.ArgumentParser(
        description='diffusion deGibbsing and denoising.')

    # names, directories, type of exp
    parser.add_argument(
        'mode',
        type=str,
        help='mode to run in ("train" or "test")'
    )
    parser.add_argument(
        'exp_type',
        type=str,
        help='experiment type for training ("comp2mag", "comp2mag_pocs", "mag2mag", or "mag2hires")'
    )

    # configuration files
    parser.add_argument(
        '--data_config_file',
        type=str,
        default='./configs/data.yaml',
        help='name of the configuration file with data directories'
    )
    parser.add_argument(
        '--model_config_file',
        type=str,
        default='./configs/model.yaml',
        help='name of the configuration file with default model parameters'
    )
    parser.add_argument(
        '--optim_config_file',
        type=str,
        default='./configs/optim.yaml',
        help='name of the configuration file with parameters for the optimizer parameters'
    )
    parser.add_argument(
        '--transforms_config_file',
        type=str,
        default='./configs/transforms.yaml',
        help='name of the configuration file with parameters for MR simulation'
    )
    parser.add_argument(
        '--losses_file',
        type=str,
        default='~/data/diffusion/imagenet_losses.pkl',
        help='name of file for saving loss values in validation data'
    )
    parser.add_argument(
        '--freq_resp_file',
        type=str,
        default='~/data/diffusion/freq_resp.pkl',
        help='name of file for saving frequency responses'
    )
    parser.add_argument(
        '--cnr_file',
        type=str,
        default='~/data/diffusion/cnrs.pkl',
        help='name of file for saving frequency responses'
    )

    # directories
    parser.add_argument(
        '--exp_dir',
        type=str,
        help='directory for loading/saving models and Tensorboard output'
    )
    parser.add_argument(
        '--train_data_dir',
        type=str,
        help='directory for ImageNet (both training and validation)'
    )
    parser.add_argument(
        '--test_data_dir',
        type=str,
        help='directory for test data'
    )

    # resource parameters
    parser.add_argument(
        '--batch_size',
        type=int,
        help='number of examples for each gradient calculation'
    )

    # hyperparameters
    parser.add_argument(
        '--gibb_size',
        type=str,
        help='size of images with simulated Gibbs artifact'
    )
    parser.add_argument(
        '--orig_big',
        type=str,
        help='size of images prior to Gibbs crop'
    )
    parser.add_argument(
        '--pf_fact',
        type=int,
        help='partial Fourier factor (before division by 8)'
    )
    parser.add_argument(
        '--pe_fact',
        help='phase encoding factor'
    )
    parser.add_argument(
        '--nlayers',
        type=int,
        help='number of layers of the U-Net'
    )
    parser.add_argument(
        '--nchans',
        type=int,
        help='number of filters in U-Net top layer'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='number of epochs for training'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='the learning rate for the optimizer'
    )
    parser.add_argument(
        '--workers_per_gpu',
        type=int,
        help='number of workers for PyTorch data loader'
    )

    return parser


def read_yamls(args, params):
    def recursive_update(d, key, value):
        foundflag = False
        for k, v in d.items():
            if k == key:
                d[k] = value
                foundflag = True
            elif isinstance(v, dict):
                d[k], found = recursive_update(d[k], key, value)
                if found is True:
                    foundflag = True

        return d, foundflag

    try:
        with open(args['data_config_file']) as f:
            data_params = yaml.safe_load(f)
            for key, value in data_params.items():
                if 'dir' in key:
                    data_params[key] = os.path.expanduser(value)
    except:
        sys.exit('data config file {} not found'.format(
            args['data_config_file']))
    try:
        with open(args['model_config_file']) as f:
            model_params = yaml.safe_load(f)
            for key, value in model_params['exp_types'][args['exp_type']].items():
                model_params['model_params'][key] = value
            del model_params['exp_types']
    except:
        sys.exit('model params config file {} not found'.format(
            args['model_config_file']))
    try:
        with open(args['optim_config_file']) as f:
            optim_params = yaml.safe_load(f)
    except:
        sys.exit('optim params config file {} not found'.format(
            args['optim_config_file']))
    try:
        with open(args['transforms_config_file']) as f:
            transforms_params = yaml.safe_load(f)
    except:
        sys.exit('transforms params config file {} not found'.format(
            args['transforms_config_file']))

    params.update({k: v for k, v in data_params.items()})
    params.update({k: v for k, v in model_params.items()})
    params.update({k: v for k, v in optim_params.items()})
    params.update({k: v for k, v in transforms_params.items()})

    for key, value in args.items():
        if not value == None:
            if key == 'gibb_size' or key == 'orig_big':
                value = literal_eval(value)
            if 'dir' in key:
                value = os.path.expanduser(value)
            if key == 'pe_fact':
                value = np.double(value)

            params, found = recursive_update(dict(params), key, value)

            if found == False:
                params[key] = value

    return params
