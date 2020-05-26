import numpy as np
import torch


class RandTranspose(object):
    """Random transpose.

    Args:
        dat_op (boolean, default=True): Whether to apply random transpose to
            'dat' array.
        targ_op (boolean, default=False): Whether to apply random transpose
            to 'target' array.
    """

    def __init__(self, dat_op=True, targ_op=False):
        self.dat_op = dat_op
        self.targ_op = targ_op

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat'
                nparrays to be normalized
        Returns:
            sample (dict): a sample with 'target' and 'dat'
                nparrays (abs)
        """
        target, dat = sample['target'], sample['dat']

        datdims = [0]
        # randomness happens here
        targdims = list(np.random.permutation(np.array(range(
            len(target.shape)))))
        datdims = [x+1 for x in targdims]
        datdims = [0] + datdims

        if self.targ_op:
            target = np.transpose(target, axes=targdims)
        if self.dat_op:
            dat = np.transpose(dat, axes=datdims)

        sample['target'] = target
        sample['dat'] = dat

        return sample

    def __repr__(self):
        return self.__class__.__name__
