""" A collection of neural network utilities for MRI in PyTorch.

Functions include:
    fftshift - applies n/2 circulant shift on batched PyTorch tensors
    ifftshift - applies n/2 circulant inverse shift on batched PyTorch
        tensors
    back_fft - applies fftshift(ifft(ifftshift(input))) in PyTorch
    dwt2d - applies 2D discrete wavelet transform
    idwt2d - applies inverse 2D discrete wavelet transform
"""

import numpy as np
import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class roll_fftshift(nn.Module):
    def __init__(self, axis=0):
        super(roll_fftshift, self).__init__()
        self.axis = axis

    def forward(self, x):
        dim_size = x.size(self.axis)

        shift = dim_size//2

        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = x.narrow(self.axis, 0, dim_size - shift)
        after = x.narrow(self.axis, after_start, shift)
        return torch.cat([after, before], self.axis)


class roll_ifftshift(nn.Module):
    def __init__(self, axis=0):
        super(roll_ifftshift, self).__init__()
        self.axis = axis

    def forward(self, x):
        dim_size = x.size(self.axis)

        shift = (dim_size+1)//2

        after_start = dim_size - shift
        if shift < 0:
            after_start = -shift
            shift = dim_size - abs(shift)

        before = x.narrow(self.axis, 0, dim_size - shift)
        after = x.narrow(self.axis, after_start, shift)
        return torch.cat([after, before], self.axis)


class fftshift(nn.Module):
    def __init__(self, ndims, startdim=2):
        super(fftshift, self).__init__()
        self.startdim = startdim
        self.ndims = ndims
        self.roll_list = nn.ModuleList()

        for dim in range(self.startdim, ndims):
            self.roll_list.append(roll_fftshift(axis=dim))

    def forward(self, x):
        for cur_roll in self.roll_list:
            x = cur_roll(x)
        return x


class ifftshift(nn.Module):
    def __init__(self, ndims, startdim=2):
        super(ifftshift, self).__init__()
        self.startdim = startdim
        self.ndims = ndims
        self.roll_list = nn.ModuleList()

        for dim in range(self.startdim, ndims):
            self.roll_list.append(roll_ifftshift(axis=dim))

    def forward(self, x):
        for cur_roll in self.roll_list:
            x = cur_roll(x)
        return x


class back_fft(nn.Module):
    def __init__(self, ndims, startdim=2):
        super(back_fft, self).__init__()
        self.ndims = ndims
        self.startdim = startdim
        self.ishift = ifftshift(ndims=ndims, startdim=startdim)
        self.fshift = fftshift(ndims=ndims, startdim=startdim)

    def forward(self, x):
        x = self.ishift(x)

        if (self.ndims == 4):
            x = x.permute(0, 2, 3, 1)
        elif (self.ndims == 5):
            x = x.permute(0, 2, 3, 4, 1)
        elif (self.ndims == 6):
            x = x.permute(0, 2, 3, 4, 5, 1)
        else:
            raise ValueError('ndim = %d not supported!' % self.ndims)

        x = torch.ifft(x, self.ndims - 2)

        if (self.ndims == 4):
            x = x.permute(0, 3, 1, 2)
        elif (self.ndims == 5):
            x = x.permute(0, 4, 1, 2, 3)
        elif (self.ndims == 6):
            x = x.permute(0, 2, 3, 4, 5, 1)
        else:
            raise ValueError('ndim = %d not supported!' % self.ndims)

        x = self.fshift(x)

        return x


class dwt2d(nn.Module):
    def __init__(self, channels=1, wtype='haar', levels=1):
        super(dwt2d, self).__init__()
        if wtype is 'haar':
            wt = pywt.Wavelet(wtype)
            dec_hi = torch.DoubleTensor(wt.dec_hi[::-1])
            dec_lo = torch.DoubleTensor(wt.dec_lo[::-1])
        else:
            print('filter type not found!')

        self.register_buffer(
            'filters',
            torch.stack(
                [
                    dec_lo.unsqueeze(0)*dec_lo.unsqueeze(1),
                    dec_lo.unsqueeze(0)*dec_hi.unsqueeze(1),
                    dec_hi.unsqueeze(0)*dec_lo.unsqueeze(1),
                    dec_hi.unsqueeze(0)*dec_hi.unsqueeze(1)
                ] * channels
            ).unsqueeze(1)
        )

        self.channels = channels
        self.levels = levels

    def forward(self, x, levels=-1):
        if levels is -1:
            levels = self.levels

        h = x.size(2)
        w = x.size(3)
        x = torch.nn.functional.conv2d(
            x,
            self.filters,
            stride=2,
            groups=self.channels)

        if levels > 1:
            x[:, :1] = self.forward(x, levels-1)

        return x


class idwt2d(nn.Module):
    def __init__(self, channels=1, wtype='haar', levels=1, maxgroup=False):
        super(idwt2d, self).__init__()
        if wtype is 'haar':
            wt = pywt.Wavelet(wtype)
            rec_hi = torch.DoubleTensor(wt.rec_hi)
            rec_lo = torch.DoubleTensor(wt.rec_lo)
        else:
            print('filter type not found!')

        self.register_buffer(
            'inv_filters',
            torch.stack(
                [
                    rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1),
                    rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1),
                    rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1),
                    rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)
                ] * channels
            ).unsqueeze(1)
        )

        self.maxgroup = maxgroup
        self.channels = channels
        self.levels = levels

    def forward(self, x, levels=-1):
        if levels is -1:
            levels = self.levels

        h = x.size(2)
        w = x.size(3)

        if levels > 1:
            x[:, :1] = self.forward(x, levels-1)

        if self.maxgroup:
            out_chans = self.inv_filters.shape[0]
        else:
            out_chans = self.channels

        x = torch.nn.functional.conv_transpose2d(
            x,
            self.inv_filters,
            stride=2,
            groups=out_chans)

        return x
