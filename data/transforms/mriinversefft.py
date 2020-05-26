import numpy as np


class MriInverseFFT(object):
    """Apply the inverse Fourier Transform.

    Args:
        dat_op (boolean, default=True): Whether to iFFT 'dat' array.
        targ_op (boolean, default=False): Whether to iFFT 'target' array.
        unitary (boolean, default=False): Whether to use orthogonal
            norm-preserving iFFTs.
    """

    def __init__(self, dat_op=True, targ_op=False, unitary=False,
                 back_sz=False):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.unitary = unitary
        self.back_sz = back_sz

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                iFFT'd.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays (iFFT'd).
        """
        target, dat = sample['target'], sample['dat']

        if self.targ_op:
            target = np.fft.ifftshift(target, axes=range(0, target.ndim))
            if self.unitary:
                target = np.fft.ifftn(target, axes=range(0, target.ndim),
                                      norm="ortho")
            else:
                target = np.fft.ifftn(target, axes=range(0, target.ndim))
            target = np.fft.fftshift(target, axes=range(0, target.ndim))
            sample['target'] = target

        if self.dat_op:
            dat = np.fft.ifftshift(dat, axes=range(1, dat.ndim))
            if self.unitary:
                dat = np.fft.ifftn(dat, axes=range(1, dat.ndim),
                                   norm="ortho")
            else:
                dat = np.fft.ifftn(dat, axes=range(1, dat.ndim))
            dat = np.fft.fftshift(dat, axes=range(1, dat.ndim))
            sample['dat'] = dat

        return sample
