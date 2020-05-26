import numpy as np
import torch


class RandFlip(object):
    """Random flipping in each dimension.

    Args:
        dat_op (boolean, default=True): Whether to apply random flipping to
            'dat' array.
        targ_op (boolean, default=False): Whether to apply random flipping
            to 'target' array.
    """

    def __init__(self, dat_op=True, targ_op=False):
        self.dat_op = dat_op
        self.targ_op = targ_op

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' to be randomly
                flipped.
        Returns:
            sample (dict): a sample with 'target' and 'dat' arrays (flipped!).
        """
        target, dat = sample['target'], sample['dat']

        for i in range(len(target.shape)):
            if np.random.randint(2) == 1:
                if self.targ_op:
                    target = np.flip(target, i)
                if self.dat_op:
                    dat = np.flip(dat, i+1)

        sample['target'] = target
        sample['dat'] = dat

        return sample

    def __repr__(self):
        return self.__class__.__name__
