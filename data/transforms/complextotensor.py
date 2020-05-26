import numpy as np
import torch


class ComplexToTensor(object):
    """Convert a set of complex ndarrays to a tensor.

    Args:
        float_flag (boolean, default=False): If true, casts tensors as
            tensor.float, otherwise tensor.double.
        pf_flag (boolean, default=False): Whether to concatenate a standard
            partial Fourier reconstruction as an extra channel dimension.
    """

    def __init__(self, complex_flag=True, dtype=torch.double, pf_flag=False):
        self.complex_flag = complex_flag
        self.dtype = dtype
        self.pf_flag = pf_flag

    def __call__(self, sample):
        target, dat = sample['target'], sample['dat']

        if np.iscomplex(dat).any():
            dat = np.concatenate((np.real(dat), np.imag(dat)))
        if np.iscomplex(target).any():
            target = np.reshape(target, (1,) + target.shape)
            target = np.concatenate((np.real(target), np.imag(target)))
        else:
            target = np.expand_dims(target, 0)

        if 'margosian_image' in sample.keys():
            dat = np.concatenate((dat, sample['margosian_image']), 0)
        if 'pocs_image' in sample.keys():
            dat = np.concatenate(
                (
                    dat,
                    np.real(sample['pocs_image']),
                    np.imag(sample['pocs_image'])
                ),
                0
            )

        return {'target': torch.tensor(target, dtype=self.dtype),
                'dat': torch.tensor(dat, dtype=self.dtype)}
