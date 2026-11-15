#!/bin/bash
# Run all 4 ablation experiments with TUNED hyperparameters
# Key changes from v1:
#   - EDL: coeff 0.1->0.03, loss mse->log (better convergence)
#   - Wavelet: SE reduction 16->8 (less aggressive compression)
#   - Wavelet position: layer2->layer2_layer3 (multi-scale frequency attention)
#   - Early stopping: patience=10 (stop overfitting)
#   - Epochs: 50->40 (early stopping will likely trigger ~25-30)
set -e
cd /root/autodl-tmp/ich_project/code

EPOCHS=40
PATIENCE=10

echo "============================================"
echo "Experiment 1/4: Baseline (ResNet-50, no wavelet, no EDL)"
echo "============================================"
python train.py --exp_name baseline --backbone resnet50 --epochs $EPOCHS --patience $PATIENCE

echo "============================================"
echo "Experiment 2/4: Wavelet only (ResNet-50 + wavelet attention, multi-scale)"
echo "============================================"
python train.py --exp_name wavelet_only --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2_layer3 --se_reduction 8 --epochs $EPOCHS --patience $PATIENCE

echo "============================================"
echo "Experiment 3/4: EDL only (ResNet-50 + EDL head, log loss, low coeff)"
echo "============================================"
python train.py --exp_name edl_only --backbone resnet50 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience $PATIENCE

echo "============================================"
echo "Experiment 4/4: Full model (ResNet-50 + multi-scale wavelet + EDL)"
echo "============================================"
python train.py --exp_name wavelet_edl --backbone resnet50 --use_wavelet \
  --wavelet_pos layer2_layer3 --se_reduction 8 --use_edl \
  --edl_loss log --edl_coeff 0.03 --epochs $EPOCHS --patience $PATIENCE

echo "============================================"
echo "All experiments done. Running evaluation..."
echo "============================================"
python evaluate.py

echo "Done!"
