import numpy as np
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio


class MetricEval(object):
    @staticmethod
    def mse(gt, pred):
        """ Compute Mean Squared Error (MSE) """
        return np.mean((gt - pred) ** 2)

    @staticmethod
    def nmse(gt, pred):
        """ Compute Normalized Mean Squared Error (NMSE) """
        return np.linalg.norm(gt - pred) ** 2 / np.linalg.norm(gt) ** 2

    @staticmethod
    def psnr(gt, pred):
        """ Compute Peak Signal to Noise Ratio metric (PSNR) """
        return peak_signal_noise_ratio(gt, pred, data_range=gt.max())

    @staticmethod
    def ssim(gt, pred):
        """ Compute Structural Similarity Index Metric (SSIM). """
        return structural_similarity(gt, pred, data_range=gt.max())
