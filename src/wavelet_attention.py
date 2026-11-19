"""Wavelet frequency-domain attention module using Haar DWT/IDWT."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT(nn.Module):
    """Single-level 2D Haar Discrete Wavelet Transform via strided convolution."""
    def __init__(self):
        super().__init__()
        ll = torch.tensor([[1, 1], [1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[-1, -1], [1, 1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[-1, 1], [-1, 1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[1, -1], [-1, 1]], dtype=torch.float32) * 0.5
        kernels = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)  # [4, 1, 2, 2]
        self.register_buffer('kernels', kernels)

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.reshape(b * c, 1, h, w)
        out = F.conv2d(x, self.kernels, stride=2, padding=0)  # [b*c, 4, h/2, w/2]
        _, _, oh, ow = out.shape
        out = out.reshape(b, c, 4, oh, ow)
        ll = out[:, :, 0]
        lh = out[:, :, 1]
        hl = out[:, :, 2]
        hh = out[:, :, 3]
        return ll, lh, hl, hh


class HaarIDWT(nn.Module):
    """Inverse Haar DWT: exact reconstruction from the 4 subbands.

    legacy=True reproduces the placement convention used for the originally
    reported results, in which the inverse filtering is followed by a fixed
    180-degree transposition within each 2x2 block (see the manuscript's
    implementation note). The default is the exact inverse.
    """
    def __init__(self, legacy=False):
        super().__init__()
        self.legacy = legacy

    def forward(self, ll, lh, hl, hh):
        b, c, h, w = ll.shape
        a = (ll - lh - hl + hh) / 2    # top-left
        b_ = (ll - lh + hl - hh) / 2   # top-right
        c_ = (ll + lh - hl - hh) / 2   # bottom-left
        d = (ll + lh + hl + hh) / 2    # bottom-right
        if self.legacy:
            a, b_, c_, d = d, c_, b_, a
        top = torch.stack([a, b_], dim=-1).reshape(b, c, h, w * 2)
        bot = torch.stack([c_, d], dim=-1).reshape(b, c, h, w * 2)
        return torch.stack([top, bot], dim=-2).reshape(b, c, h * 2, w * 2)


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention."""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, 1)
        self.fc2 = nn.Conv2d(channels // reduction, channels, 1)

    def forward(self, x):
        s = self.avg_pool(x)
        s = F.relu(self.fc1(s))
        s = torch.sigmoid(self.fc2(s))
        return x * s


class WaveletAttention(nn.Module):
    """Wavelet frequency-domain attention: DWT -> SE per subband -> IDWT -> residual."""
    def __init__(self, channels, reduction=8, legacy_idwt=False):
        super().__init__()
        self.dwt = HaarDWT()
        self.idwt = HaarIDWT(legacy=legacy_idwt)
        self.se_ll = SEBlock(channels, reduction)
        self.se_lh = SEBlock(channels, reduction)
        self.se_hl = SEBlock(channels, reduction)
        self.se_hh = SEBlock(channels, reduction)
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        ll, lh, hl, hh = self.dwt(x)
        ll = self.se_ll(ll)
        lh = self.se_lh(lh)
        hl = self.se_hl(hl)
        hh = self.se_hh(hh)
        out = self.idwt(ll, lh, hl, hh)
        out = self.bn(self.fuse(out))
        return x + out


class WaveletModuleSEOnly(nn.Module):
    """Parameter-matched SE-only attention: single SE block + fuse/BN + residual.

    One SE with reduction r/4 has the same FC parameters as four SEs with
    reduction r, so total module parameters match WaveletAttention.
    """
    def __init__(self, channels, reduction):
        super().__init__()
        self.se = SEBlock(channels, reduction)
        self.fuse = nn.Conv2d(channels, channels, 1)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = self.se(x)
        out = self.bn(self.fuse(out))
        return x + out
