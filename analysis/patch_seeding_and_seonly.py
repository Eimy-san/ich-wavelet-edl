"""Patch: seed control + parameter-matched SE-only attention variant."""

import re

# ---- train.py: seeding ----
src = open("code/train.py").read()
seed_block = """

def set_seed(seed):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
"""
src = src.replace(
    "from edl_head import get_edl_loss, edl_predict, evidence_to_alpha\n",
    "from edl_head import get_edl_loss, edl_predict, evidence_to_alpha\n" + seed_block,
)
src = src.replace(
    "    cfg = get_config()\n    device = torch.device",
    "    cfg = get_config()\n    set_seed(cfg.seed)\n    device = torch.device",
)
src = src.replace(
    "    results = {\n        'exp_name': cfg.exp_name,",
    "    results = {\n        'exp_name': cfg.exp_name, 'seed': cfg.seed,",
)
open("code/train.py", "w").write(src)

# ---- model.py: SE-only attention (parameter-matched) ----
src = open("code/model.py").read()
src = src.replace(
    "from wavelet_attention import WaveletAttention",
    "from wavelet_attention import WaveletAttention, SEBlock, WaveletModuleSEOnly",
)
src = src.replace(
    "                 wavelet_pos='layer2', se_reduction=8):",
    "                 wavelet_pos='layer2', se_reduction=8, se_only=False):",
)
src = src.replace(
    "        self.wavelet_pos = wavelet_pos\n",
    "        self.wavelet_pos = wavelet_pos\n        self.se_only = se_only\n",
)
src = src.replace(
    """            if use_wavelet:
                if wavelet_pos in ('layer2', 'layer2_layer3'):
                    self.wavelet_l2 = WaveletAttention(512, se_reduction)
                if wavelet_pos in ('layer3', 'layer2_layer3'):
                    self.wavelet_l3 = WaveletAttention(1024, se_reduction)
        elif backbone == 'efficientnet_b0':""",
    """            if use_wavelet:
                if wavelet_pos in ('layer2', 'layer2_layer3'):
                    self.wavelet_l2 = (WaveletModuleSEOnly(512, se_reduction // 4)
                                       if se_only else WaveletAttention(512, se_reduction))
                if wavelet_pos in ('layer3', 'layer2_layer3'):
                    self.wavelet_l3 = (WaveletModuleSEOnly(1024, se_reduction // 4)
                                       if se_only else WaveletAttention(1024, se_reduction))
        elif backbone == 'efficientnet_b0':""",
)
open("code/model.py", "w").write(src)

# ---- config.py: --se_only flag ----
src = open("code/config.py").read()
src = src.replace(
    "    p.add_argument('--wavelet_pos',",
    "    p.add_argument('--se_only', action='store_true', default=False,\n"
    "                   help='parameter-matched SE-only attention (no DWT decomposition)')\n"
    "    p.add_argument('--wavelet_pos',",
)
open("code/config.py", "w").write(src)
print("patched train/config/model")
