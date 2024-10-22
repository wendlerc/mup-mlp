#!/bin/bash

# Define arrays for widths and learning rates
#widths=(512 1024 2048 4096 9192)
widths=(500 1000 2000 3000 4000 5000 6000 7000 8000 10000)
learning_rates=(0.125)
opt="adam"

# Loop through widths
for width in "${widths[@]}"; do
    # Loop through learning rates
    for lr in "${learning_rates[@]}"; do
        # Run without μP
        python train.py --width $width --lr 6.103e-05 --n_epochs 25 --use_wandb --wandb_project cifar10-$opt --optimizer $opt

        # Run with μP
        python train.py --width $width --lr 0.125 --n_epochs 25 --use_wandb --wandb_project cifar10-$opt --use_mup --optimizer $opt
    done
done