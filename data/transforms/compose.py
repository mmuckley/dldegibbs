import numpy as np
import torch


class Compose(object):
    """Composes several transforms together.
        Args:
            transforms (list of ``Transform`` objects): list of transforms 
            to compose.
        Example:
            >>> transforms.Compose([
            >>>     transforms.MriNoise(),
            >>>     transforms.ComplexToTensor(),
            >>> ])

        Returns:
            ob (PyTorch transform object): Can be used with PyTorch dataset
                with transform=ob option.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, sample):
        # list conversion (for multiple coils)
        dat = sample['dat']
        dat = np.reshape(dat, (1,) + dat.shape)
        sample['dat'] = dat

        sample['siglevel'] = np.mean(np.absolute(sample['target']))

        # transform array
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string
