# train_sensitivity_analysis.py
# v1.1: Sử dụng các hàm từ utils.py

import numpy as np
import torch
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import os
import time

# SỬA LỖI: Import mọi thứ từ utils.py
from utils import MLP_Predictor, LSTM_Predictor, prepare_combined_data, train_and_evaluate_convergence

def main():
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    constellations = ['iridium', 'starlink', 'oneweb']
    seq_len = 5
    NUM_EPOCHS = 15
    
    horizons_in_steps = [1, 5, 10, 15]
    
    models_to_test = {
        "MLP (Vector)": {"model_class": MLP_Predictor, "feature_type": "vector"},
        "LSTM (Vector)": {"model_class": LSTM_Predictor, "feature_type": "vector"},
        "LSTM (GA)": {"model_class": LSTM_Predictor, "feature_type": "ga"},
        "LSTM (Hybrid)": {"model_class": LSTM_Predictor, "feature_type": "hybrid"},
    }
    
    print("--- Bắt đầu chuẩn bị dữ liệu cho tất cả các horizons ---")
    all_datasets = {}
    for horizon in horizons_in_steps:
        print(f"\n--- Horizon = {horizon*60}s ({horizon} steps) ---")
        all_datasets[horizon] = {
            'vector': prepare_combined_data(constellations, 'vector', seq_len, horizon),
            'ga': prepare_combined_data(constellations, 'ga', seq_len, horizon),
            'hybrid': prepare_combined_data(constellations, 'hybrid', seq_len, horizon),
        }

    sensitivity_results = {name: [] for name in models_to_test.keys()}
    
    for horizon in horizons_in_steps:
        print(f"\n\n===== BẮT ĐẦU THỰC NGHIỆM VỚI HORIZON = {horizon*60}s =====\n")
        for model_name, config in models_to_test.items():
            feature_type = config['feature_type']
            X, y = all_datasets[horizon][feature_type]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            feat_dim = X.shape[2]
            if config["model_class"] == MLP_Predictor:
                model = MLP_Predictor(input_size=seq_len * feat_dim)
            else:
                model = LSTM_Predictor(input_size=feat_dim)
            
            res = train_and_evaluate_convergence(model, X_train, y_train, X_test, y_test, f"{model_name} @ H={horizon}", NUM_EPOCHS)
            sensitivity_results[model_name].append(res['F1 Score'])

    print("\n--- Generating Figure 6: Sensitivity Analysis on Prediction Horizon ---")
    horizons_in_minutes = [h * 60 / 60 for h in horizons_in_steps]
    
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(12, 8))
    
    markers = ['o', 's', '^', 'D']
    for i, (name, f1_scores) in enumerate(sensitivity_results.items()):
        ax.plot(horizons_in_minutes, f1_scores, marker=markers[i], linestyle='--', markersize=8, label=name)

    ax.set_title('Model Performance Sensitivity to Prediction Horizon', fontsize=18, pad=20, weight='bold')
    ax.set_xlabel('Prediction Horizon (minutes)', fontsize=14)
    ax.set_ylabel('F1 Score on Test Set', fontsize=14)
    ax.set_xticks(horizons_in_minutes)
    ax.tick_params(axis='both', labelsize=12)
    ax.legend(fontsize=12)
    ax.grid(True, which='both', linestyle='--')
    
    plt.tight_layout()
    fig_filename = os.path.join(output_dir, 'fig_sensitivity_analysis.png')
    plt.savefig(fig_filename, dpi=600)
    print(f"Saved Figure 6 to '{fig_filename}'")

if __name__ == '__main__':
    main()
