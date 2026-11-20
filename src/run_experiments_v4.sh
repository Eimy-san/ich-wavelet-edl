#!/bin/bash
set -e
# activate your Python environment with PyTorch installed, e.g.:
# source /path/to/miniconda3/etc/profile.d/conda.sh && conda activate base
cd "${ICH_CODE_DIR:-$(dirname "$0")}"
EPOCHS=50
# NOTE on the inverse Haar transform (--idwt_legacy):
# The manuscript's primary results (Tables 2, 3, 7, 8 and the multi-seed
# replications) were produced with the original implementation, whose IDWT
# reconstructs each 2x2 block with a fixed permutation of the subband
# contributions. The exact inverse Haar transform is the released default;
# passing --idwt_legacy selects the original reconstruction so that the
# primary numbers reproduce exactly. The SE-only ablation runs and the
# robustness runs reported in Section 2.4 used the exact transform (default,
# no flag). See README.md "Reproducing the experiments" for the mapping.
LEGACY="--idwt_legacy"
echo "============================================"
echo "Experiment 1/5: Baseline (50 epochs, no early stop)"
echo "============================================"
python train.py --exp_name baseline --backbone resnet50 --epochs $EPOCHS --patience 0 $LEGACY
echo "============================================"
echo "Experiment 2/5: Wavelet only (single-scale layer2, reduction=16)"
echo "============================================"
python train.py --exp_name wavelet_only --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2 --se_reduction 16 --epochs $EPOCHS --patience 0 $LEGACY
echo "============================================"
echo "Experiment 3/5: EDL only (log loss, coeff=0.03)"
echo "============================================"
python train.py --exp_name edl_only --backbone resnet50 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0 $LEGACY
echo "============================================"
echo "Experiment 4/5: Full model (single-scale wavelet + EDL log/0.03)"
echo "============================================"
python train.py --exp_name wavelet_edl --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2 --se_reduction 16 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0 $LEGACY
echo "============================================"
echo "Experiment 5/5: SE-only ablation (parameter-matched SE, exact IDWT)"
echo "============================================"
python train.py --exp_name se_only --backbone resnet50 --use_wavelet --se_only \
  --wavelet_pos layer2 --se_reduction 16 --epochs $EPOCHS --patience 0
echo "============================================"
echo "All done. Running evaluation..."
echo "============================================"
python evaluate.py
echo "Done!"
