import os

import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as torchtransforms
from torch.utils.data import Dataset


class ImageNet(Dataset):
    """ImageNet dataset.

    Args:
        root_dir (str): Path to ImageNet root directory.
        crop_sz (int): Size to crop loaded images to (passed to 
            torchtransforms.RandomResizedCrop).
        transform (list, default=None): List of transforms to apply to loaded
            images.
    """

    def __init__(self, root_dir, crop_sz=256, transform=None):
        self.crop_sz = crop_sz
        self.root_dir = root_dir
        self.transform = transform
        self.datloader = datasets.ImageFolder(
            self.root_dir,
            torchtransforms.Compose([
                torchtransforms.RandomResizedCrop(
                    size=np.max(self.crop_sz),
                    scale=(1.0, 1.0),
                    ratio=(1.0, 1.0)
                )
            ])
        )
        self.dat_norm_fact = 255  # actually 189.1

    def __len__(self):
        num = len(self.datloader)
        return num

    def __getitem__(self, index):
        y = self.datloader[index]

        y = np.array(y[0])

        y = 0.2989 * y[..., 0] + 0.5870 * y[..., 1] + 0.1140 * y[..., 2]
        y = np.squeeze(y)
        y = y / self.dat_norm_fact
        y = y.astype(dtype=np.complex)

        for dim, dimsize in enumerate(y.shape):
            slicelist = []
            if not dimsize == self.crop_sz[dim]:
                crop1 = (dimsize - self.crop_sz[dim]) // 2
                crop2 = dimsize - self.crop_sz[dim] - crop1

                slicelist.append(slice(crop1, -crop2, None))

        y = y[tuple(slicelist)]

        sample = {'target': y, 'dat': np.copy(y)}

        if self.transform:
            sample = self.transform(sample)

        return sample
