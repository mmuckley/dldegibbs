import numpy as np


class MriMargosian(object):
    """Calculate Margosian partial Fourier reconstruction.

    This calculates a Margosian image estimate based on 'dat'. The partial
    Fourier dimension is assumed to be the last.

    Args:
        cent_size (tuple): Sice of k-space center used for phase estimation.
    """

    def __init__(self, cent_size):
        self.cent_size = cent_size

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays, 'dat'
                to receive Margosian estimate.
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays, as
                well as new 'margosian_image'.
        """
        dat = sample['dat']

        nx = dat.shape[1]
        ny = dat.shape[2]

        fftaxes = tuple(range(1, len(dat.shape)))

        phase_est = np.fft.ifftshift(dat, axes=fftaxes)
        phase_est = np.fft.fftn(phase_est, axes=fftaxes)
        phase_est = np.fft.fftshift(phase_est, axes=fftaxes)

        window = np.expand_dims(np.hamming(self.cent_size[0]), 1) * \
            np.expand_dims(np.hamming(self.cent_size[1]), 0)
        ny1 = ny//2 - self.cent_size[1]//2
        ny2 = ny - ny1 - self.cent_size[1]
        window = np.concatenate((
            np.zeros(shape=(nx, ny1)),
            window,
            np.zeros(shape=(nx, ny2))),
            1
        )
        window = np.expand_dims(window, 0)

        ramp = np.flip(
            np.array(range(self.cent_size[1])) / self.cent_size[1], 0)
        ramp = np.concatenate((
            np.ones(shape=(ny1,)),
            ramp,
            np.zeros(shape=(ny2,))
        ), 0)
        ramp = np.expand_dims(np.expand_dims(ramp, 0), 0)

        half_im = ramp * phase_est

        phase_est = window * phase_est

        phase_est = np.fft.ifftshift(phase_est, axes=fftaxes)
        phase_est = np.fft.ifftn(phase_est, axes=fftaxes)
        phase_est = np.fft.fftshift(phase_est, axes=fftaxes)
        phase_est = np.angle(phase_est)

        half_im = np.fft.ifftshift(half_im, axes=fftaxes)
        half_im = np.fft.ifftn(half_im, axes=fftaxes)
        half_im = np.fft.fftshift(half_im, axes=fftaxes)

        marg_im = 2 * np.real(half_im * np.exp(-1j*phase_est))

        sample['margosian_image'] = marg_im

        return sample

    def __repr__(self):
        return self.__class__.__name__
