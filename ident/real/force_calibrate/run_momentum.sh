#!/bin/bash

# This script runs pred_momentum.py with different values of K

# for K in 0.1 0.3 0.5 0.8 1 1.2 1.5 2 2.5 3 3.5 4 5 8 10
for K in 8
do
    echo "Running with K=$K"
    python pred_momentum.py --K $K
done
