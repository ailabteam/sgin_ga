# train_convergence_analysis.py (v2.0 - Data Export Only)
# Mục đích: Chạy thực nghiệm so sánh chính và lưu kết quả ra file CSV.
# File này không còn vẽ biểu đồ nữa.

import numpy as np
import torch
import pandas as pd
import os
import time
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
    # Sử dụng một seed cố định để đảm bảo kết quả chính có thể tái lặp
    set_seed(42)

    # Tạo thư mục lưu kết quả nếu chưa có
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    start_total_time = time.time()
    
    # --- 1. Thiết lập Thực nghiệm ---
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5
    seq_len = 5
    NUM_EPOCHS = 30
    
    # --- 2. Chuẩn bị Dữ liệu ---
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', seq_len, horizon_steps)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', seq_len, horizon_steps)
    X_hyb, y_hyb = prepare_combined_data(constellations, 'hybrid', seq_len, horizon_steps)
    
    # Chia dữ liệu
    X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.3, random_state=42, stratify=y_vec)
    X_ga_train, X_ga_test, _, _ = train_test_split(X_ga, y_ga, test_size=0.3, random_state=42, stratify=y_ga)
    X_hyb_train, X_hyb_test, _, _ = train_test_split(X_hyb, y_hyb, test_size=0.3, random_state=42, stratify=y_hyb)

    # --- 3. Chạy Thực nghiệm ---
    results = []
    all_histories = {}
    feat_dim = X_vec.shape[2]
    
    # Chạy MLP (Vector)
    model_mlp = MLP_Predictor(input_size=seq_len * feat_dim)
    res_mlp = train_and_evaluate_convergence(model_mlp, X_vec_train, y_train, X_vec_test, y_test, "MLP (Vector)", NUM_EPOCHS)
    all_histories['MLP_Vector'] = res_mlp.pop('History')
    results.append({'Model': 'MLP', 'Features': 'Vector', **res_mlp})

    # Chạy LSTM (Vector)
    model_lstm_vec = LSTM_Predictor(input_size=feat_dim)
    res_lstm_vec = train_and_evaluate_convergence(model_lstm_vec, X_vec_train, y_train, X_vec_test, y_test, "LSTM (Vector)", NUM_EPOCHS)
    all_histories['LSTM_Vector'] = res_lstm_vec.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Vector', **res_lstm_vec})
    
    # Chạy LSTM (GA)
    model_lstm_ga = LSTM_Predictor(input_size=feat_dim)
    res_lstm_ga = train_and_evaluate_convergence(model_lstm_ga, X_ga_train, y_train, X_ga_test, y_test, "LSTM (GA)", NUM_EPOCHS)
    all_histories['LSTM_GA'] = res_lstm_ga.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'GA', **res_lstm_ga})
    
    # Chạy LSTM (Hybrid)
    model_lstm_hyb = LSTM_Predictor(input_size=feat_dim)
    res_lstm_hyb = train_and_evaluate_convergence(model_lstm_hyb, X_hyb_train, y_train, X_hyb_test, y_test, "LSTM (Hybrid)", NUM_EPOCHS)
    all_histories['LSTM_Hybrid'] = res_lstm_hyb.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Hybrid', **res_lstm_hyb})
    
    # --- 4. Lưu Kết quả ra file ---

    # a) Lưu bảng kết quả chính (dữ liệu cho Table 3)
    df_results = pd.DataFrame(results)
    main_results_path = os.path.join(output_dir, 'table_3_main_performance.csv')
    df_results.to_csv(main_results_path, index=False, float_format='%.4f')
    print(f"\nSaved main performance results (for Table 3) to '{main_results_path}'")
    print("--- Main Performance Table ---")
    print(df_results.to_string(index=False))

    # b) Lưu lịch sử hội tụ (dữ liệu cho Figure 5)
    history_dfs = []
    for name, history in all_histories.items():
        model, feature = name.split('_')
        temp_df = pd.DataFrame({
            'Epoch': range(1, NUM_EPOCHS + 1),
            'Validation_F1': history['val_f1'],
            'Model': model,
            'Features': feature
        })
        history_dfs.append(temp_df)
    
    df_history = pd.concat(history_dfs, ignore_index=True)
    convergence_data_path = os.path.join(output_dir, 'figure_5_convergence_data.csv')
    df_history.to_csv(convergence_data_path, index=False, float_format='%.4f')
    print(f"Saved convergence history (for Figure 5) to '{convergence_data_path}'")

    end_total_time = time.time()
    print(f"\nToàn bộ quá trình thực nghiệm hoàn tất sau {(end_total_time - start_total_time) / 60:.2f} phút.")

if __name__ == '__main__':
    main()
