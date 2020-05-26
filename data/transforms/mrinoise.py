import numpy as np
import torch


class MriNoise(object):
    """Add complex Gaussian noise.

    dat = dat + ComplexGauss(0, siglevel/snr)

    Args:
        dat_op (boolean, default=True): Whether to add noise to 'dat' array.
        targ_op (boolean, default=False): Whether to add noise to 'target'
            array.
        snr_range (tuple): log2(snr_level), a 2-tuple specifying min and max
            SNR for complex Gaussian noise. Each sample receives its own random
            noise level.
        logflag (boolean, default=True): Use snr_range parameter on a log-2
            scale.
        complex_flag (boolean, default=True): Whether to add complex Gaussian
            noise.
        sigma (double): Noise standard deviation parameter (overwrites
            snr_range).
    """

    def __init__(self, dat_op=True, targ_op=False, snr_range=(0, 6),
                 logflag=True, complex_flag=True, sigma=None):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.snr_range = snr_range
        self.complex_flag = complex_flag
        self.logflag = logflag
        self.sigma = sigma

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' to receive noise.
        Returns:
            sample (dict): a sample with 'target' and 'dat' arrays (noisy).
        """
        target, dat = sample['target'], sample['dat']

        if self.logflag:
            cur_snr = 2**(np.random.uniform(
                low=self.snr_range[0],
                high=self.snr_range[1]))
        else:
            cur_snr = np.random.uniform(
                low=self.snr_range[0],
                high=self.snr_range[1])

        if ('siglevel' in sample):
            siglevel = sample['siglevel']
        else:
            siglevel = np.mean(np.absolute(dat[0, ...]))

        if self.sigma is None:
            sigma = siglevel / cur_snr
        else:
            sigma = self.sigma

        if self.complex_flag:
            if self.dat_op:
                sample['dat'] = dat + sigma*(
                    np.random.normal(size=dat.shape) +
                    1j * np.random.normal(size=dat.shape))
            if self.targ_op:
                sample['target'] = target + sigma*(
                    np.random.normal(size=target.shape) +
                    1j * np.random.normal(size=target.shape))
        else:
            if self.dat_op:
                sample['dat'] = dat + sigma*(
                    np.random.normal(size=dat.shape))
            if self.targ_op:
                sample['target'] = target + sigma*(
                    np.random.normal(size=target.shape))

        return sample
