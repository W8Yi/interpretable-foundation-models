#!/usr/bin/env bash

# Script to train the PACE-enhanced ViT model on the Color task

# set GPU device
export CUDA_VISIBLE_DEVICES=1
Dataset="Color"

python main.py --train --task $Dataset --name ViT-PACE --num_epochs 1