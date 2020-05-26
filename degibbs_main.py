import os
import pickle
import sys

import torch

from configs.config_init import generate_parser, read_yamls
from data.degibbs_loaders import init_loaders
from degibbs_model import DeGibbs


def run_train(params, model):
    exp_folder_name = '{}/wavelet_unet_imnet_pffact_{}_snrlog0-5_nlayers_{}'.format(
        params['exp_type'],
        params['transform_params']['pf_fact'],
        params['model_params']['nlayers'],
    )

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        pe_fact=params['transform_params']['pe_fact'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )

    model = model.init_optimizer(
        learning_rate=params['training_params']['learning_rate'],
        loss_fn=torch.nn.MSELoss(),
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
    )

    model = model.fit(
        exp_dir=os.path.join(params['exp_dir'], exp_folder_name),
        num_epochs=params['training_params']['num_epochs']
    )

    return(0)


def run_val(params, model):
    exp_folder_name = '{}/wavelet_unet_imnet_pffact_{}_snrlog0-5_nlayers_{}'.format(
        params['exp_type'],
        params['transform_params']['pf_fact'],
        params['model_params']['nlayers'],
    )

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )

    visuals = {}
    val_loss, visuals = model.run_val_loop(
        visuals,
        rnd_seed=5
    )

    print('validation loss: {}'.format(val_loss))

    return(0)


def run_test(params, model):
    exp_folder_name = '{}/wavelet_unet_imnet_pffact_{}_snrlog0-5_nlayers_{}'.format(
        params['exp_type'],
        params['transform_params']['pf_fact'],
        params['model_params']['nlayers'],
    )

    metrics_file = os.path.join(
        params['exp_dir'], exp_folder_name, 'metrics_nonreduce.pkl')

    model = model.load_model(
        model_file=os.path.join(
            params['exp_dir'], exp_folder_name, 'best_model.pt'),
        device=params['device']
    )

    train_loader, val_loader = init_loaders(
        exp_type=params['exp_type'],
        data_dir=params['train_data']['train_data_dir'],
        batch_size=params['training_params']['batch_size'],
        num_workers=params['train_data']['workers_per_gpu'] *
        torch.cuda.device_count(),
        gibb_size=params['transform_params']['gibb_size'],
        orig_big=params['transform_params']['orig_big'],
        pf_fact=params['transform_params']['pf_fact'],
        test=True,
        test_dir=params['train_data']['test_data_dir'],
        dtype=params['dtype']
    )
    model = model.attach_data(
        exp_type=params['exp_type'],
        train_loader=train_loader,
        val_loader=val_loader,
        train_data_dir=params['train_data']['train_data_dir']
    )
    model.epoch = 0
    model.loss_fn = torch.nn.MSELoss()

    visuals = {}
    metrics, visuals = model.run_val_loop(
        visuals,
        rnd_seed=5,
        all_metrics=True
    )

    print(metrics)

    with open(metrics_file, 'wb') as f:
        pickle.dump(metrics, f)

    return(0)


def main():
    parser = generate_parser()
    args = vars(parser.parse_args())

    params = read_yamls(args, dict())

    model_params = params['model_params']
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.double

    params['device'] = device
    params['dtype'] = dtype

    print(params)

    model = DeGibbs(
        nlayers=model_params['nlayers'],
        in_ch=model_params['in_ch'],
        out_ch=model_params['out_ch'],
        nchans=model_params['nchans'],
        comp2mag=model_params['comp2mag'],
        leaky=model_params['leaky'],
        device=torch.device('cpu'),
        dtype=dtype
    )

    if params['mode'] == 'train':
        run_train(params, model)
    elif params['mode'] == 'val':
        run_val(params, model)
    elif params['mode'] == 'test':
        run_test(params, model)

    return(0)


if __name__ == "__main__":
    sys.exit(main())
