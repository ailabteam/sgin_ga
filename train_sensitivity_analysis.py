# train_sensitivity_analysis.py (v2.0 - Data Export Only)
# Mục đích: Chạy thực nghiệm phân tích độ nhạy và lưu kết quả ra file CSV.

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Import các công cụ từ utils.py
from utils import (
    MLP_Predictor,
    LSTM_Predictor,
    prepare_combined_data,
    train_and_evaluate_convergence,
    set_seed
)

def main():
    set_seed(42)
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)

    # --- Thiết lập ---
    constellations = ['iridium', 'starlink', 'oneweb']
    seq_len = 5
    NUM_EPOCHS = 15 # Dùng 15 epochs để chạy nhanh hơn
    
    # Các prediction horizon để thử nghiệm
    horizons_in_steps = [1, 5, 10, 15]
    
    models_to_test = {
        "MLP (Vector)": {"model_class": MLP_Predictor, "feature_type": "vector"},
        "LSTM (Vector)": {"model_class": LSTM_Predictor, "feature_type": "vector"},
        "LSTM (GA)": {"model_class": LSTM_Predictor, "feature_type": "ga"},
        "LSTM (Hybrid)": {"model_class": LSTM_Predictor, "feature_type": "hybrid"},
    }
    
    # --- Chuẩn bị Dữ liệu ---
    print("--- Pre-loading all datasets for all horizons ---")
    all_datasets = {}
    feature_types = ['vector', 'ga', 'hybrid']
    for ft in feature_types:
        for horizon in horizons_in_steps:
             X, y = prepare_combined_data(constellations, ft, seq_len, horizon)
             all_datasets[(ft, horizon)] = (X, y)

    # --- Chạy Thực nghiệm ---
    sensitivity_results = []
    
    for horizon in horizons_in_steps:
        print(f"\n\n===== RUNNING SENSITIVITY @ HORIZON = {horizon*60}s =====\n")
        for model_name, config in models_to_test.items():
            feature_type = config['feature_type']
            
            X, y = all_datasets[(feature_type, horizon)]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
            
            feat_dim = X.shape[2]
            if config["model_class"] == MLP_Predictor:
                model = MLP_Predictor(input_size=seq_len * feat_dim)
            else:
                model = LSTM_Predictor(input_size=feat_dim)
            
            res = train_and_evaluate_convergence(model, X_train, y_train, X_test, y_test, f"{model_name} @ H={horizon}", NUM_EPOCHS)
            model_id, feature_id = model_name.replace(')', '').split(' (')
            
            sensitivity_results.append({
                'Horizon (minutes)': horizon,
                'Model': model_id,
                'Features': feature_id,
                'F1_Score': res['F1 Score']
            })

    # --- Lưu Kết quả ---
    print("\n--- Saving Sensitivity Analysis Results ---")
    df_sensitivity = pd.DataFrame(sensitivity_results)
    sensitivity_path = os.path.join(output_dir, 'figure_6_sensitivity_data.csv')
    df_sensitivity.to_csv(sensitivity_path, index=False, float_format='%.4f')
    
    print(f"Saved sensitivity analysis data (for Figure 6) to '{sensitivity_path}'")
    print(df_sensitivity.to_string(index=False))

if __name__ == '__main__':
    main()
