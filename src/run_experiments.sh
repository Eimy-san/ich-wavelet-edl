#!/bin/bash
# Run all 4 ablation experiments
set -e
cd /root/autodl-tmp/ich_project/code

echo "============================================"
echo "Experiment 1/4: Baseline (ResNet-50, no wavelet, no EDL)"
echo "============================================"
python train.py --exp_name baseline --backbone resnet50 --epochs 50

echo "============================================"
echo "Experiment 2/4: Wavelet only (ResNet-50 + wavelet attention)"
echo "============================================"
python train.py --exp_name wavelet_only --backbone resnet50 --use_wavelet --epochs 50

echo "============================================"
echo "Experiment 3/4: EDL only (ResNet-50 + EDL head)"
echo "============================================"
python train.py --exp_name edl_only --backbone resnet50 --use_edl --edl_loss mse --edl_coeff 0.1 --epochs 50

echo "============================================"
echo "Experiment 4/4: Full model (ResNet-50 + wavelet + EDL)"
echo "============================================"
python train.py --exp_name wavelet_edl --backbone resnet50 --use_wavelet --use_edl --edl_loss mse --edl_coeff 0.1 --epochs 50

echo "============================================"
echo "All experiments done. Running evaluation..."
echo "============================================"
python evaluate.py

echo "Done!"
