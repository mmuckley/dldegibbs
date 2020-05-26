import numpy as np


class MriPfPocs(object):
    """Calculate POCs Partial Fourier reconstruction.

    This calculates a POCs partial Fourier image estimate. Only operates on
    'dat'.

    Args:
        cent_size (tuple): Sice of k-space center used for phase estimation.
        niter (int): Number of POCs iterations.
    """

    def __init__(self, cent_size, niter=5):
        self.cent_size = cent_size
        self.niter = niter

    def __call__(self, sample):
        """
        Args:
            sample (dict): a sample with 'target' and 'dat' numpy arrays ('dat'
                to be POCs'd).
        Returns:
            sample (dict): a sample with 'target' and 'dat' numpy arrays, as
                well as new 'pocs_image' array.
        """
        dat = sample['dat']

        nx = dat.shape[1]
        ny = dat.shape[2]

        fftaxes = tuple(range(1, len(dat.shape)))

        pf_cutoff = self.cent_size[1] // 2 + dat.shape[2] // 2

        phase_est = np.fft.ifftshift(dat, axes=fftaxes)
        phase_est = np.fft.fftn(phase_est, axes=fftaxes)
        phase_est = np.fft.fftshift(phase_est, axes=fftaxes)
        kdata = phase_est.copy()

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

        phase_est = window * phase_est

        phase_est = np.fft.ifftshift(phase_est, axes=fftaxes)
        phase_est = np.fft.ifftn(phase_est, axes=fftaxes)
        phase_est = np.fft.fftshift(phase_est, axes=fftaxes)
        phase_est = np.angle(phase_est)

        pocs_im = dat

        for _ in range(self.niter):
            pocs_im = np.absolute(pocs_im) * np.exp(1j*phase_est)

            pocs_im = np.fft.ifftshift(
                np.fft.fftn(
                    np.fft.fftshift(
                        pocs_im,
                        axes=fftaxes
                    ),
                    axes=fftaxes
                ),
                axes=fftaxes
            )

            pocs_im[:, :, :pf_cutoff] = kdata[:, :, :pf_cutoff]

            pocs_im = np.fft.ifftshift(
                np.fft.ifftn(
                    np.fft.fftshift(
                        pocs_im,
                        axes=fftaxes
                    ),
                    axes=fftaxes
                ),
                axes=fftaxes
            )

        sample['pocs_image'] = pocs_im

        return sample
