#!/bin/bash

# Define arrays for widths and learning rates
widths=(128 256 512 1024 2048)
learning_rates=(1e-6 1e-5 1e-4 1e-3 1e-2 1e-1)

# Loop through widths
for width in "${widths[@]}"; do
    # Loop through learning rates
    for lr in "${learning_rates[@]}"; do
        # Run without μP
        python train.py --width $width --lr $lr --n_epochs 1 --use_wandb

        # Run with μP
        python train.py --width $width --lr $lr --n_epochs 1 --use_wandb --use_mup
    done
done