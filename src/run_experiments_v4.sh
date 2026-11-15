#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/ich_project/code
EPOCHS=50
echo "============================================"
echo "Experiment 1/4: Baseline (50 epochs, no early stop)"
echo "============================================"
python train.py --exp_name baseline --backbone resnet50 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 2/4: Wavelet only (single-scale layer2, reduction=16)"
echo "============================================"
python train.py --exp_name wavelet_only --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2 --se_reduction 16 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 3/4: EDL only (log loss, coeff=0.03)"
echo "============================================"
python train.py --exp_name edl_only --backbone resnet50 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 4/4: Full model (single-scale wavelet + EDL log/0.03)"
echo "============================================"
python train.py --exp_name wavelet_edl --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2 --se_reduction 16 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0
echo "============================================"
echo "All done. Running evaluation..."
echo "============================================"
python evaluate.py
echo "Done!"
