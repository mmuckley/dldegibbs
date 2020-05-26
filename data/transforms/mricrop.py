import operator

import numpy as np
import torch


class MriCrop(object):
    """Crop an image to a specified size.

    Args:
        crop_sz (ndarray): the size of the cropped image.
        dat_op (boolean, default=True): Whether to crop 'dat' array.
        targ_op (boolean, default=False): Whether to crop 'target' array.
        scale (double, default=None): Scaling factor (to adjust FFT
            normalization).
    """

    def __init__(self, crop_sz, dat_op=True, targ_op=False, scale=None):
        self.crop_sz = crop_sz
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.scale = scale

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                cropped.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays (cropped).
        """
        target, dat = sample['target'], sample['dat']

        targ_crop_sz = self.crop_sz
        crop_sz = (1,) + self.crop_sz

        if self.targ_op:
            targ_start_ind = tuple(
                map(lambda a, da: (a-da)//2, target.shape, targ_crop_sz))
            targ_end_ind = tuple(
                map(operator.add, targ_start_ind, targ_crop_sz))
            targ_slices = tuple(map(slice, targ_start_ind, targ_end_ind))
            if self.scale:
                target = target * self.scale
            sample['target'] = target[targ_slices]

        if self.dat_op:
            start_ind = tuple(map(lambda a, da: (a-da)//2, dat.shape, crop_sz))
            end_ind = tuple(map(operator.add, start_ind, crop_sz))
            slices = tuple(map(slice, start_ind, end_ind))
            if self.scale:
                dat = dat * self.scale
            sample['dat'] = dat[slices]

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(crop_sz={})'.format(self.crop_sz)
