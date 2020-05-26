import torch
import torch.nn as nn
import torch.nn.functional as F

from . import utils


class UNetWavelet(nn.Module):
    """A U-Net with wavelet downsampling/upsampling.

    This code implements a U-Net in PyTorch, which is presumably useful for
    things such as:
        - classification
        - denoising
        - Gibbs artifact removal

    Example usage:
        unet = UNetWavelet(nlayers=10)

    Based on the SIAM paper by Ye et al.

    Args:
        nlayers (int, default=10): Number of U-Net layers.
        in_ch (int, default=2): Number of input channels. Typically 2 for
            real/imaginary, 1 for magnitude images.
        out_ch (int, default=2): Number of output channels, see in_ch.
        top_filtnum (int, default=64): Number of channels output by first layer.
        n_classes (int, default=False): Number of classes (not tested).
        resid (boolean, default=True): If True, then applies a residual skipped
            connection over the entire network.
        wave_concat (boolean, default=True): If true, during upsampling the
            network concates wavelet channels rather than applying the wavelet
            adjoint. This was the approach in the paper of Ye, and was found
            empirically to reduce checkerboarding.
        comp2mag (boolean, default=False): If True, applies a complex magnitude
            operation prior to returning the result.
        leaky (boolean, default=False): If True, use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, nlayers=10, in_ch=2, out_ch=2, ndims=2, top_filtnum=64,
                 n_classes=False, resid=True, wave_concat=True, comp2mag=False,
                 leaky=False):
        super(UNetWavelet, self).__init__()

        if not ndims == 2:
            print('NOT TESTED FOR ANYTHING OTHER THAN 2D')

        self.ndims = ndims
        self.resid = resid
        self.wave_concat = wave_concat
        self.comp2mag = comp2mag
        self.leaky = leaky

        self.inconv = InConv(in_ch, top_filtnum, ndims=ndims, leaky=leaky)
        lastout = top_filtnum

        self.nlayers = nlayers

        self.downlayers = nn.ModuleList()
        self.wavedown = nn.ModuleList()
        for _ in range(0, (nlayers-2)//2-1):
            self.wavedown.append(utils.dwt2d(channels=lastout))
            self.downlayers.append(
                DoubleConv(lastout, lastout*2, ndims=ndims, leaky=leaky)
            )
            lastout = lastout*2

        self.wavedown.append(utils.dwt2d(channels=lastout))
        self.downlayers.append(MidConv(lastout, ndims=ndims, leaky=leaky))

        self.waveup = nn.ModuleList()
        self.uplayers = nn.ModuleList()
        for _ in range((nlayers-2)//2-1):
            if self.wave_concat is False:
                self.waveup.append(utils.idwt2d(channels=lastout))
                self.uplayers.append(
                    ConcatDoubleConv(
                        lastout*2,
                        lastout//2,
                        ndims=ndims,
                        leaky=leaky
                    )
                )
            else:
                self.waveup.append(utils.idwt2d(
                    channels=lastout, maxgroup=True))
                self.uplayers.append(
                    ConcatDoubleConv(
                        lastout*5,
                        lastout//2,
                        ndims=ndims,
                        leaky=leaky
                    )
                )
            lastout = lastout//2

        if self.wave_concat is False:
            self.waveup.append(utils.idwt2d(channels=lastout))
            self.uplayers.append(
                ConcatDoubleConv(
                    lastout*2,
                    lastout,
                    ndims=ndims,
                    leaky=leaky
                )
            )
        else:
            self.waveup.append(utils.idwt2d(channels=lastout, maxgroup=True))
            self.uplayers.append(
                ConcatDoubleConv(
                    lastout*5,
                    lastout,
                    ndims=ndims,
                    leaky=leaky
                )
            )
        lastout = lastout

        if n_classes:
            self.outlayer = OutConv(lastout, n_classes, ndims=ndims)
        else:
            self.outlayer = OutConv(lastout, out_ch, ndims=ndims)

    def forward(self, x):
        # run all the downsampling layers, saving outputs and wavelet
        # decompositions separately
        down_outputs = []
        wt_dec = []
        down_outputs.append(self.inconv(x))
        for i, curlayer in enumerate(self.downlayers):
            cur_wv = self.wavedown[i]
            wt_dec.append(cur_wv(down_outputs[i]))
            down_outputs.append(curlayer(wt_dec[i][:, :cur_wv.channels]))

        up_wt_rec = []
        up_outputs = []
        up_outputs.append(down_outputs[-1])

        # run all the upsampling layers, saving wavelet recompsitions
        # and layer outputs separately
        for i, curlayer in enumerate(self.uplayers):
            cur_wv = self.waveup[i]
            if self.wave_concat is False:
                up_wt_rec.append(
                    cur_wv(
                        torch.cat(
                            (up_outputs[i],  wt_dec[-1+-1*i]
                             [:, cur_wv.channels:]),
                            1
                        )
                    )
                )
            else:
                cur_chan = up_outputs[i].shape[1]
                up_wt_rec.append(
                    cur_wv(
                        torch.cat(
                            (up_outputs[i], wt_dec[-1+-1*i][:, cur_chan:]),
                            1
                        )
                    )
                )

            up_outputs.append(curlayer(up_wt_rec[i], down_outputs[-2-i]))

        # apply the final output layer
        y = self.outlayer(up_outputs[-1])

        # residual skipped connection (considering partial Fourier channel)
        if self.resid:
            y = y + x[:, :2, ...]

        # magnitude operation
        if self.comp2mag:
            y = torch.sqrt(y[:, 0, ...]**2 + y[:, 1, ...]**2).unsqueeze(1)

        return y


class DoubleConv(nn.Module):
    """Double convolution.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        mid_ch (int, default=-1): Number of channels for intermediate output.
            If -1, then mid_ch is set to out_ch.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
        leaky (boolean, default=False): Whether to use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, in_ch, out_ch, mid_ch=-1, ndims=2, leaky=False):
        super(DoubleConv, self).__init__()

        if mid_ch is -1:
            mid_ch = out_ch
            self.mid_ch = out_ch

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ndims = ndims
        self.leaky = leaky

        if (ndims == 2):
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_ch),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, mid_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(mid_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True)
                )
        else:
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(mid_ch),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, mid_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(mid_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(mid_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.conv(x)
        return x


class MidConv(nn.Module):
    """Convolution for bottom of U-Net.

    This module applies a convolution that doubles the number of channels, then
    applies a second convolution that outputs the same number of input
    channels.

    Args:
        in_ch (int): Number of input channels.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
        leaky (boolean, default=False): Whether to use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, in_ch, ndims=2, leaky=False):
        super(MidConv, self).__init__()

        self.in_ch = in_ch
        self.ndims = ndims
        self.leaky = leaky

        self.ndims = ndims
        if (ndims == 2):
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch*2),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv2d(in_ch*2, in_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, in_ch*2, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch*2),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(in_ch*2, in_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(in_ch),
                    nn.ReLU(inplace=True)
                )
        else:
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, in_ch*2, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_ch*2),
                    nn.LeakyReLU(inplace=True),
                    nn.Conv3d(in_ch*2, in_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_ch),
                    nn.LeakyReLU(inplace=True)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, in_ch*2, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_ch*2),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(in_ch*2, in_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(in_ch),
                    nn.ReLU(inplace=True)
                )

    def forward(self, x):
        x = self.conv(x)
        return x


class InConv(nn.Module):
    """Input convolution.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
        leaky (boolean, default=False): Whether to use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, in_ch, out_ch, ndims=2, leaky=False):
        super(InConv, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ndims = ndims
        self.leaky = leaky

        if (ndims == 2):
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.LeakyReLU(inplace=True),
                    DoubleConv(out_ch, out_ch, ndims=ndims, leaky=self.leaky)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    DoubleConv(out_ch, out_ch, ndims=ndims, leaky=self.leaky)
                )
        else:
            if self.leaky:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.LeakyReLU(inplace=True),
                    DoubleConv(out_ch, out_ch, ndims=ndims, leaky=self.leaky)
                )
            else:
                self.conv = nn.Sequential(
                    nn.Conv3d(in_ch, out_ch, kernel_size=3, padding=1),
                    nn.BatchNorm3d(out_ch),
                    nn.ReLU(inplace=True),
                    DoubleConv(out_ch, out_ch, ndims=ndims, leaky=self.leaky)
                )

    def forward(self, x):
        y = self.conv(x)
        return y


class DownConv(nn.Module):
    """Down-sampling convolution.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
        leaky (boolean, default=False): Whether to use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, in_ch, out_ch, ndims=2, leaky=False):
        super(DownConv, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ndims = ndims
        self.leaky = leaky

        if (ndims == 2):
            self.mpconv = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                DoubleConv(in_ch, out_ch, ndims=ndims, leaky=self.leaky)
            )
        else:
            self.mpconv = nn.Sequential(
                nn.MaxPool3d(kernel_size=2),
                DoubleConv(in_ch, out_ch, ndims=ndims, leaky=self.leaky)
            )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class ConcatDoubleConv(nn.Module):
    """Concatenate, then apply double convolution.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
        leaky (boolean, default=False): Whether to use leaky ReLUs instead of
            normal ones (not tested).
    """

    def __init__(self, in_ch, out_ch, ndims=2, leaky=False):
        super(ConcatDoubleConv, self).__init__()

        self.in_ch = in_ch
        self.out_ch = out_ch
        self.ndims = ndims
        self.leaky = leaky

        self.conv = DoubleConv(
            in_ch,
            out_ch,
            mid_ch=out_ch*2,
            ndims=ndims,
            leaky=self.leaky
        )

    def forward(self, x1, x2):
        diffX = x2.size()[3] - x1.size()[3]
        diffY = x2.size()[2] - x1.size()[2]
        #x1 = F.upsample(x1, size=x2.size())
        x1 = F.pad(x1, (diffX // 2, int(diffX / 2) + diffX % 2,
                        diffY // 2, int(diffY / 2) + diffY % 2), mode='constant')
        x = torch.cat([x2, x1], dim=1)

        x = self.conv(x)

        return x


class OutConv(nn.Module):
    """Output convolution layer.

    Args:
        in_ch (int): Number of input channels.
        out_ch (int): Number of output channels.
        ndims (int, default=2): Number of dimensions (3 dimensions not tested).
    """

    def __init__(self, in_ch, out_ch, ndims=2):
        super(OutConv, self).__init__()

        self.ndims = ndims
        if (ndims == 2):
            self.conv = nn.Conv2d(
                in_ch,
                out_ch,
                kernel_size=1,
                padding=0,
                groups=1
            )
        else:
            self.conv = nn.Conv3d(
                in_ch,
                out_ch,
                kernel_size=1,
                padding=0,
                groups=1
            )

    def forward(self, x):
        x = self.conv(x)

        return x
