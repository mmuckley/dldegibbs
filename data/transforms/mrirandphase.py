import numpy as np
from scipy.ndimage import gaussian_filter


class MriRandPhase(object):
    """A transform class for applying random phase.

    This initializes and object that applies random phase to a set of input
    images. The class accomplishes this by generating a random cloud of points
    and convolving the points with a Gaussian filter of fixed size - equivalent
    to summing up many Gaussian basis functions.

    The transform is designed to take as input a "sample" object with "dat"
    and "target" keys. Flags determine operations on the data contained in the
    keys.

    Args:
        dat_op (boolean, default: True): A flag on whether to add random phase
            to the "dat."
        targ_op (boolean, default: False): A flag on whether to add random
            phase to the "target."
        nophase_prob (double, default=0.05): Probability of not applying random
            phase.
        num_bfn_per_size_param (double, default=15): A parameter for a Poisson
            distribution that generates the number of basis functions to
            construct for each basis function size.
        bfn_amp_mean_sig (tuple, default=(0, 30)): A tuple specifying the mean
            and standard deviation for the distribution of basis function
            amplitudes.
        num_bfn_per_size_param (double, default=12): A parameter for a Poisson
            distribution that generates the number of different basis function
            sizes.
        bfn_size_amp_sig_min (tuple, default=(8, 3, 1): Parameters for the
            mean, standard deviation, and minimum value of the radial basis
            function sizes.
    """

    def __init__(self, dat_op=True, targ_op=False, nophase_prob=0.01,
                 num_bfn_per_size_param=30, bfn_amp_mean_sig=(0, 5),
                 num_sizes_param=12, bfn_size_amp_sig_min=(50, 30, 10),
                 pi_noprob=0.5):
        self.dat_op = dat_op
        self.targ_op = targ_op
        self.nophase_prob = nophase_prob
        self.num_bfn_per_size_param = num_bfn_per_size_param
        self.bfn_amp_mean_sig = bfn_amp_mean_sig
        self.num_sizes_param = num_sizes_param
        self.bfn_size_amp_sig_min = bfn_size_amp_sig_min
        self.pi_noprob = pi_noprob

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' to receive random
                phase.
        Returns:
            sample (dict): a sample with 'target' and 'dat' arrays with random
                phase.
        """
        target, dat = sample['target'], sample['dat']

        dims = np.flip(target.shape, 0)
        if len(dims) > 2:
            print(
                'mrirandphase.py: Not programmed for more than 2 dimensions! Aborting...')
            return sample

        phase_map = np.zeros(shape=target.shape)
        if np.random.uniform() > self.pi_noprob:
            if np.random.uniform() > 0.5:
                phase_map += 180
            else:
                phase_map -= 180

        gauss_map = np.zeros(shape=target.shape)

        if np.random.uniform() > self.nophase_prob:
            num_sizes = np.random.poisson(self.num_sizes_param)

            for _ in range(num_sizes):
                gauss_map[:] = +0

                num_bfns = np.random.poisson(lam=self.num_bfn_per_size_param)

                bfn_size = np.absolute(
                    np.random.normal(
                        loc=self.bfn_size_amp_sig_min[0],
                        scale=self.bfn_size_amp_sig_min[1]
                    )
                )
                bfn_size = np.maximum(bfn_size, self.bfn_size_amp_sig_min[2])

                bfn_amps = np.random.normal(
                    loc=self.bfn_amp_mean_sig[0],
                    scale=self.bfn_amp_mean_sig[1],
                    size=num_bfns
                ) * (bfn_size**2 * 2 * np.pi)

                bfn_x0s = np.floor(np.random.uniform(
                    high=dims[0], size=num_bfns)).astype(np.int)
                bfn_y0s = np.floor(np.random.uniform(
                    high=dims[1], size=num_bfns)).astype(np.int)

                gauss_map[tuple([bfn_x0s, bfn_y0s])] = bfn_amps

                gauss_map = gaussian_filter(gauss_map, sigma=bfn_size)

                # update the phase map
                phase_map += gauss_map

        if self.targ_op:
            target = target * np.exp(1j * phase_map * np.pi / 180)
            sample['target'] = target

        if self.dat_op:
            phase_map = np.reshape(phase_map, (1,) + phase_map.shape)
            dat = dat * np.exp(1j * phase_map * np.pi / 180)
            sample['dat'] = dat

        return sample

    def __repr__(self):
        return self.__class__.__name__


def main():
    """No arguments, runs a testing script."""
    print('running test script')
    import matplotlib.pyplot as plt
    import time

    phaseob = MriRandPhase(pi_noprob=1)

    in_im = np.ones(shape=(256, 256))

    sample = {
        'dat': in_im,
        'target': in_im
    }

    time_start = time.time()
    out_im = phaseob(sample)['dat'][0, ...]
    time_end = time.time()

    print('sim time: {}'.format(time_end - time_start))

    plt.imshow(np.angle(out_im))
    plt.gray()
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
