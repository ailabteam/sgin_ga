#!/bin/bash

# run_experiments.sh
# Tự động chạy train_convergence_analysis.py với các seed khác nhau.

# Danh sách các seed để chạy
SEEDS=(42 101 2023)

echo "Starting experiments for seeds: ${SEEDS[@]}"

for seed in "${SEEDS[@]}"; do
    echo "------------------------------------------"
    echo "RUNNING EXPERIMENT WITH SEED: $seed"
    echo "------------------------------------------"
    # Chạy script Python với tham số --seed
    python train_convergence_analysis.py --seed "$seed"
done

echo "All experiments completed."
echo "Now, run 'python analyze_results.py' to get the final summary table."
