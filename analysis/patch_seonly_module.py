"""Append WaveletModuleSEOnly to wavelet_attention.py on the server."""

code = '''

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
'''
src = open("code/wavelet_attention.py").read()
if "WaveletModuleSEOnly" not in src:
    src += code
open("code/wavelet_attention.py", "w").write(src)
print("appended SE-only module")
