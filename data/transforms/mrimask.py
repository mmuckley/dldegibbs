import numpy as np
import torch


class MriMask(object):
    """ Apply a mask multiplication.

    Args:
        dat_op (boolean, default=True): Whether to mask 'dat' array.
        targ_op (boolean, default=False): Whether to mask 'target' array.
        mask (ndarray): an ndarray specifying the mask.
    """

    def __init__(self, dat_op=True, targ_op=False, mask=None):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.mask = mask

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                masked.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays (masked).
        """
        target, dat = sample['target'], sample['dat']

        if self.mask is None:
            mask = np.zeros(dat[0, ...].shape)
            for i in range(mask.shape[0]):
                if (i % 4 == 0):
                    mask[i, ...] = 1
        else:
            mask = self.mask

        if self.targ_op:
            sample['target'] = np.where(mask, target, 0)
        if self.dat_op:
            mask = np.reshape(mask, (1,) + mask.shape)
            sample['dat'] = np.where(mask, dat, 0)

        sample['mask'] = mask

        return sample
