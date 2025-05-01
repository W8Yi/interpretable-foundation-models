#!/usr/bin/env bash

# Script to train the PACE-enhanced ViT model on the Color task

# set GPU device
export CUDA_VISIBLE_DEVICES=1

# Set your parameters here
TASK="Color"
NAME="ViT-base"
EPOCHS=5
LR="1e-3"
REQUIRE_GRAD="--require_grad"

# Execute training
python main.py \
  --train \
  --task "$TASK" \
  --name "$NAME" \
  --num_epochs $EPOCHS \
  --lr $LR \
  $REQUIRE_GRAD