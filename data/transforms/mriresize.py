import numpy as np
from skimage.transform import resize


class MriResize(object):
    """Resize an image using scikit transforms.

    The resizing is based on splines. The order of the spline is determined
    by the 'order' parameter.

    Args:
        dat_op (boolean, default=True): Whether to apply residing to 'dat'
            array.
        targ_op (boolean, default=False): Whether to apply resizing to 'target'
            array.
        order (int, default=3): Order of spline (see skimage.transform.resize).
        complex_flag (boolean, default=False): Whether input is complex. If
            true, applies spline interpolation in real and imaginary dimensions
            separately.
    """

    def __init__(self, output_sz, dat_op=True, targ_op=False, order=3,
                 complex_flag=False):
        self.output_sz = output_sz
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.order = order
        self.complex_flag = complex_flag

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                resized.
        Returns:
            sample (dict): a sample with 'target' and 'dat' arrays (resized!).
        """
        target, dat = sample['target'], sample['dat']

        if self.targ_op:
            if self.complex_flag:
                sample['target'] = resize(
                    np.real(target),
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                ) + 1j*resize(
                    np.imag(target),
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                )
            else:
                sample['target'] = resize(
                    target,
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                )

        if self.dat_op:
            dat = np.transpose(dat)
            if self.complex_flag:
                dat = resize(
                    np.real(dat),
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                ) + 1j*resize(
                    np.imag(dat),
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                )
            else:
                dat = resize(
                    dat,
                    self.output_sz,
                    order=self.order,
                    mode='reflect',
                )
            dat = np.transpose(dat)
            sample['dat'] = dat

        return sample

    def __repr__(self):
        return self.__class__.__name__ + '(output_sz={})'.format(self.output_sz)
