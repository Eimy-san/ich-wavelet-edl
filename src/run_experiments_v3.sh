#!/bin/bash
set -e
source /root/miniconda3/etc/profile.d/conda.sh
conda activate base
cd /root/autodl-tmp/ich_project/code
EPOCHS=40
echo "============================================"
echo "Experiment 1/4: Baseline"
echo "============================================"
python train.py --exp_name baseline --backbone resnet50 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 2/4: Wavelet only (multi-scale)"
echo "============================================"
python train.py --exp_name wavelet_only --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2_layer3 --se_reduction 8 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 3/4: EDL only (log loss, low coeff)"
echo "============================================"
python train.py --exp_name edl_only --backbone resnet50 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0
echo "============================================"
echo "Experiment 4/4: Full model"
echo "============================================"
python train.py --exp_name wavelet_edl --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2_layer3 --se_reduction 8 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience 0
echo "============================================"
echo "All done. Running evaluation..."
echo "============================================"
python evaluate.py
echo "Done!"
