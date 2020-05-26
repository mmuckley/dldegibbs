import numpy as np
import torch


class MriNormalize(object):
    """Normalization (division)

    Args:
        dat_op (boolean, default=True): Whether to normalize 'dat' array.
        targ_op (boolean, default=False): Whether to normalize 'target'  array.
        percentile_norm (double, default=None): Percentile norm to normalize
            by. If None, then the normalization divides by the mean. If a
            a double, then it the normalization divides by that ordered value.
            If 'Max', then normalization divides by the maximum absolute value.
        gauss (boolean, default=False): A poorly-labeled boolean that subtracts
            the mean and divides by the standard deviation for complex values
            (not recommended.)
        maggauss (boolean, default=False): A poorly-labeled boolean that
            subtracts the magnitude mean and divides by the magnitude standard
            deviation (output remains complex).
    """

    def __init__(self, dat_op=True, targ_op=False, percentile_norm=None,
                 gauss=False, maggauss=False):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.percentile_norm = percentile_norm
        self.gauss = gauss
        self.maggauss = maggauss

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                normalized.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays
                (normalized!).
        """
        target, dat = sample['target'], sample['dat']

        # normalize the data
        if self.percentile_norm is None:
            siglevel = np.mean(np.absolute(dat[0, ...]))
        elif self.percentile_norm is 'Max':
            siglevel = np.max(np.absolute(dat[0, ...]))
        else:
            siglevel = np.percentile(np.absolute(
                dat[0, ...]), self.percentile_norm)

        siglevel = np.max([siglevel, 1e-7])

        if self.maggauss:
            sigmean = np.mean(np.absolute(dat[0, ...]))
            sigstd = np.std(np.absolute(dat[0, ...]))

            if self.dat_op:
                datmag = np.absolute(dat)
                datphase = dat / datmag
                datmag = (dat - sigmean) / sigstd
                sample['dat'] = datmag * datphase
            if self.targ_op:
                targmag = np.absolute(target)
                targphase = target / targmag
                targmag = (target - sigmean) / sigstd
                sample['target'] = targmag * targphase
        elif self.gauss:
            sigmean = np.mean(dat[0, ...])
            sigstd = np.max([np.std(dat[0, ...]), 1e-7])

            if self.dat_op:
                sample['dat'] = (dat - sigmean) / sigstd
            if self.targ_op:
                sample['target'] = (target - sigmean) / sigstd
        else:
            if self.dat_op:
                sample['dat'] = (dat / siglevel)

            if self.targ_op:
                sample['target'] = (target / siglevel)

        return sample
