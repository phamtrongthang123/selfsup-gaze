#!/bin/bash
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate env_torchism/
CUDA_VISIBLE_DEVICES=0 python train.py --config configs/train/sample_cifar.yaml --gpus 0
