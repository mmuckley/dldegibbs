import numpy as np
import torch
from scipy.sparse.linalg import svds


class MriFFT(object):
    """Apply the forward Fourier Transform.

    Args:
        dat_op (boolean, default=True): Whether to FFT 'dat' array.
        targ_op (boolean, default=False): Whether to FFT 'target' array.
        unitary (boolean, default=False): Whether to use orthogonal
            norm-preserving FFTs.
        sigflag (boolean, default=True): Whether to compute norm of input
            and add it to sample dictionary. The signal level is scaled by the
            FFT norm change.
    """

    def __init__(self, dat_op=True, targ_op=False, unitary=False,
                 sigflag=True):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.unitary = unitary
        self.sigflag = sigflag

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                FFT'd.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays (FFT'd).
        """
        target, dat = sample['target'], sample['dat']

        if self.targ_op:
            target = np.fft.ifftshift(target, axes=range(0, target.ndim))
            if self.unitary:
                target = np.fft.fftn(target, axes=range(0, target.ndim),
                                     norm="ortho")
            else:
                target = np.fft.fftn(target, axes=range(0, target.ndim))
            target = np.fft.fftshift(target, axes=range(0, target.ndim))
            sample['target'] = target

        if self.dat_op:
            dat = np.fft.ifftshift(dat, axes=range(1, dat.ndim))
            if self.unitary:
                dat = np.fft.fftn(dat, axes=range(1, dat.ndim), norm="ortho")
            else:
                dat = np.fft.fftn(dat, axes=range(1, dat.ndim))
            dat = np.fft.fftshift(dat, axes=range(1, dat.ndim))
            sample['dat'] = dat

        if self.sigflag:
            if 'siglevel' in sample:
                if not self.unitary:
                    sample['siglevel'] = sample['siglevel'] * \
                        np.sqrt(target.size)
            else:
                sample['siglevel'] = np.mean(np.absolute(dat))

        return sample
