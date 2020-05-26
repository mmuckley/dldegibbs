import numpy as np


class MriRandEllipse(object):
    """Apply random ellipse crop.

    Args:
        dat_op (boolean, default=True): Whether to apply random ellipse crop to
            'dat' array.
        targ_op (boolean, default=False): Whether to apply random ellipse crop
            to 'target' array.
        el_range (tuple, default=(0.5, 1)): Scaling parameters for cropping in
            first and second dimension.
        no_el_prob (double, default=0.1): Probability of applying no cropping.
    """

    def __init__(self, dat_op=True, targ_op=False, el_range=(0.5, 1),
                 no_el_prob=0.1):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.el_range = el_range
        self.no_el_prob = no_el_prob

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays to be
                cropped.
        Returns:
            sample (dict): a sample with 'target' and 'dat' arrays (cropped!).
        """
        target, dat = sample['target'], sample['dat']

        if np.random.uniform() < self.no_el_prob:
            return sample

        dims = np.flip(target.shape, 0)
        if len(dims) > 2:
            print(
                'mrirandellipse.py: Not programmed for more than 2 dimensions! Aborting...')
            return sample

        yy, xx = np.meshgrid(range(dims[1]), range(dims[0]), indexing='ij')

        a_parm = np.random.uniform(low=self.el_range[0],
                                   high=self.el_range[1])*dims[0]/2
        b_parm = np.random.uniform(low=self.el_range[0],
                                   high=self.el_range[1])*dims[1]/2

        x0 = np.random.uniform(high=dims[0]) / 2
        y0 = np.random.uniform(high=dims[1]) / 2

        el_field = np.zeros(shape=target.shape)
        el_field[(((yy-(dims[1]/2-y0))/b_parm)**2 +
                  ((xx-x0-(dims[0]/2-x0))/a_parm)**2 <= 1)] = 1

        if self.targ_op:
            target[1-el_field > 0.5] = 0
            sample['target'] = target

        if self.dat_op:
            el_field = np.reshape(el_field, (1,) + el_field.shape)
            dat[1 - el_field > 0.5] = 0
            sample['dat'] = dat

        return sample
