import os
import sys

import numpy as np
import torch

from . import transforms
from .imagenet import ImageNet


def worker_init_fn(worker_id):
    """Pytorch worker initialization function."""
    np.random.seed(np.random.get_state()[1][0] + worker_id)


def init_loaders(exp_type, data_dir, batch_size, num_workers, gibb_size,
                 orig_big, pf_fact, pe_fact=1, snr_range=(0, 6), sigma=None,
                 test=False, test_dir=None, dtype=torch.double):
    """Data loader initialization for Gibbs simulation.

    This script initializes data loaders for simulating how properties of the
    FFT manifest as the Gibbs phenomenon. It includes standard data
    augmentation methods (random flipping, cropping etc.), random phase
    simulation, Gibbs cropping, and simulations for partial Fourier.

    This data loader does not simulate Fourier-space apodization that may occur
    in some vendors' reconstruction pipelines. At the point of time of the
    script generation (June, 2019), the effects of vendor apodization remains
    an open research topic.

    Args:
        exp_type (str): A string indicating the experiment type.
        data_dir (str): A string pointing to a directory with training images
            with 'train', 'val', and 'test' subdirectories.
        batch_size (int): The batch size to train with.
        num_workers (int): The number of workers for data loading. More workers
            usually loads faster, but on Intel systems the multiprocessing
            package will sometimes duplicate memory and lead to excessive
            memory usage.
        gibb_size (int or tuple): The size of the image after Gibbs crop.
        orig_big (int or tuple): The size of the image before Gibbs crop.
        pf_fact (double): The partial Fourier acceleration factor.
        pe_fact (double): The phase encoding acceleration factor.
        snr_range (2-length tuple, default: (0, 6)): The minimum and maximum
            SNR of the simulation on a log-2 basis
            (min: 2^snr_range[0], max: 2^snr_range[1]).
        sigma (double, default: None): Noise sigma (overwrites snr_range).
        test (boolean, default: False): Whether to use test set (will replace
            the 'val' data loader).

    Returns:
        train_dataloader, val_dataloader (PyTorch DataLoader): Training and
            validation PyTorch data loaders.
    """

    # set up parameters
    if test:
        train_dir = test_dir
        val_dir = test_dir
    else:
        train_dir = os.path.join(data_dir, 'train/')
        val_dir = os.path.join(data_dir, 'val/')

    if isinstance(gibb_size, int):
        gibb_size = (gibb_size, gibb_size)
    if isinstance(orig_big, int):
        orig_big = (orig_big, orig_big)

    # build the transforms
    transform_set = []

    # random flip
    transform_set.append(transforms.RandFlip(targ_op=True))

    # random transpose
    transform_set.append(transforms.RandTranspose(targ_op=True))

    # phase transformation
    if any(np.array(orig_big) > 256):
        phase_trans = transforms.MriRandPhaseBig(targ_op=True)
    else:
        phase_trans = transforms.MriRandPhase(targ_op=True)
    transform_set.append(phase_trans)

    # random ellipse
    transform_set.append(transforms.MriRandEllipse(targ_op=True))

    # FFT
    transform_set.append(transforms.MriFFT(unitary=True, targ_op=True))

    # Gibbs cropping
    scale = np.prod(np.sqrt(np.array(gibb_size)/np.array(orig_big)))
    transform_set.append(
        transforms.MriCrop(
            crop_sz=gibb_size,
            scale=scale
        )
    )

    # noise transformation
    transform_set.append(transforms.MriNoise(snr_range=snr_range, sigma=sigma))

    # partial Fourier mask
    # no apodization included - may want to reconsider for some cases
    print('pe_fact is {}'.format(pe_fact))
    mask = np.ones(shape=gibb_size)
    pe_max = np.floor(gibb_size[1]*(1 - (1-pe_fact)/2)).astype(np.int)
    print('gibbs pe_max is {}'.format(pe_max))
    num_keep = (pe_max * pf_fact) // 8
    mask[:, (num_keep+1):] = 0
    mask[:, :-pe_max] = 0
    transform_set.append(transforms.MriMask(mask=mask))

    if 'lores' in exp_type:
        mask_lores = np.ones(shape=orig_big)
        pe_max = np.floor(orig_big[1]*(1 - (1-pe_fact)/2)).astype(np.int)
        print('orig pe_max is {}'.format(pe_max))
        num_keep = (pe_max * pf_fact) // 8
        mask_lores[:, (num_keep+1):] = 0
        mask_lores[:, :-pe_max] = 0

        transform_set.append(
            transforms.MriMask(dat_op=False, targ_op=True, mask=mask_lores)
        )

    # inverse FFT
    transform_set.append(transforms.MriInverseFFT(unitary=True, targ_op=True))

    # normalization
    transform_set.append(transforms.MriNormalize(
        percentile_norm='Max', targ_op=True))

    # PF calculation
    if exp_type == 'complex_margosian':
        transform_set.append(
            transforms.MriMargosian(
                cent_size=(gibb_size[0], (num_keep - gibb_size[1]//2))
            )
        )
        pf_flag = True
    elif exp_type == 'complex_pocs':
        transform_set.append(
            transforms.MriPfPocs(
                cent_size=(gibb_size[0], (num_keep - gibb_size[1]//2))
            )
        )
        pf_flag = True
    else:
        pf_flag = False

    # absolute value
    if 'magnitude' in exp_type:
        mag_dat_op = True
    else:
        mag_dat_op = False

    transform_set.append(
        transforms.MriAbsolute(targ_op=True, dat_op=mag_dat_op)
    )

    # resize the target
    transform_set.append(transforms.MriResize(
        output_sz=gibb_size, targ_op=True, dat_op=False))

    # convert to tensor
    transform_set.append(
        transforms.ComplexToTensor(
            pf_flag=pf_flag,
            dtype=dtype
        )
    )

    # compose it all together
    transform_set = transforms.Compose(transform_set)

    # set up data loader
    dataset = ImageNet(
        root_dir=train_dir,
        crop_sz=orig_big,
        transform=transform_set
    )
    val_dataset = ImageNet(
        root_dir=val_dir,
        crop_sz=orig_big,
        transform=transform_set
    )

    print('train dataset size: {}, val_dataset size: {}'.format(
        len(dataset), len(val_dataset)))

    train_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn
    )

    return train_loader, val_loader
