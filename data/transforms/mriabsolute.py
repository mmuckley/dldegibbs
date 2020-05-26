import time
import numpy as np


class MriAbsolute(object):
    """Take the absolute value.

    Args:
        dat_op (boolean, default=True): Whether to compute absolute value on
            'dat' array.
        targ_op (boolean, default=False): Whether to compute absolute value on
            'target' array.
    """

    def __init__(self, dat_op=True, targ_op=False):
        self.dat_op = dat_op
        self.targ_op = targ_op

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                absolute valued.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays.
        """
        target, dat = sample['target'], sample['dat']

        if self.targ_op:
            sample['target'] = np.absolute(target)

        if self.dat_op:
            sample['dat'] = np.absolute(dat)

        return sample
