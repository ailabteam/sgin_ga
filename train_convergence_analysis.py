# train_convergence_analysis.py (v3.0 - Batch Mode)

import numpy as np
import torch
import pandas as pd
import os
import time
import argparse  # Dùng để nhận tham số dòng lệnh
from sklearn.model_selection import train_test_split

# Import các công cụ từ utils.py, bao gồm hàm set_seed mới
from utils import (
    set_seed,
    MLP_Predictor,
    LSTM_Predictor,
    prepare_combined_data,
    train_and_evaluate_convergence
)

def main():
    # --- Thiết lập Parser để nhận tham số ---
    parser = argparse.ArgumentParser(description="Chạy thực nghiệm hội tụ với một random seed cụ thể.")
    parser.add_argument('--seed', type=int, default=42, help="Random seed cho lần chạy.")
    args = parser.parse_args()

    # Thiết lập seed cho toàn bộ quá trình
    set_seed(args.seed)
    print(f"===== RUNNING EXPERIMENT WITH SEED: {args.seed} =====")
    
    # Tạo thư mục lưu kết quả nếu chưa có
    output_dir = 'results_multiple_runs'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Các thiết lập thực nghiệm (không đổi) ---
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5; seq_len = 5; NUM_EPOCHS = 30
    
    # --- Chuẩn bị Dữ liệu ---
    # (Lưu ý: prepare_data không bị ảnh hưởng bởi seed, nhưng split thì có)
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', seq_len, horizon_steps)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', seq_len, horizon_steps)
    X_hyb, y_hyb = prepare_combined_data(constellations, 'hybrid', seq_len, horizon_steps)
    
    # Chia dữ liệu với random_state là seed được truyền vào
    X_vec_train, X_vec_test, y_train, y_test = train_test_split(X_vec, y_vec, test_size=0.3, random_state=args.seed, stratify=y_vec)
    X_ga_train, X_ga_test, _, _ = train_test_split(X_ga, y_ga, test_size=0.3, random_state=args.seed, stratify=y_ga)
    X_hyb_train, X_hyb_test, _, _ = train_test_split(X_hyb, y_hyb, test_size=0.3, random_state=args.seed, stratify=y_hyb)

    # --- Chạy Thực nghiệm ---
    results = []
    feat_dim = X_vec.shape[2]
    
    # Chạy 4 mô hình như cũ
    # MLP (Vector)
    model_mlp = MLP_Predictor(input_size=seq_len * feat_dim)
    res_mlp = train_and_evaluate_convergence(model_mlp, X_vec_train, y_train, X_vec_test, y_test, "MLP (Vector)", NUM_EPOCHS)
    res_mlp.pop('History') # Không cần history cho lần chạy này
    results.append({'Model': 'MLP', 'Features': 'Vector', **res_mlp})

    # LSTM (Vector)
    model_lstm_vec = LSTM_Predictor(input_size=feat_dim)
    res_lstm_vec = train_and_evaluate_convergence(model_lstm_vec, X_vec_train, y_train, X_vec_test, y_test, "LSTM (Vector)", NUM_EPOCHS)
    res_lstm_vec.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Vector', **res_lstm_vec})
    
    # LSTM (GA)
    model_lstm_ga = LSTM_Predictor(input_size=feat_dim)
    res_lstm_ga = train_and_evaluate_convergence(model_lstm_ga, X_ga_train, y_train, X_ga_test, y_test, "LSTM (GA)", NUM_EPOCHS)
    res_lstm_ga.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'GA', **res_lstm_ga})
    
    # LSTM (Hybrid)
    model_lstm_hyb = LSTM_Predictor(input_size=feat_dim)
    res_lstm_hyb = train_and_evaluate_convergence(model_lstm_hyb, X_hyb_train, y_train, X_hyb_test, y_test, "LSTM (Hybrid)", NUM_EPOCHS)
    res_lstm_hyb.pop('History')
    results.append({'Model': 'LSTM', 'Features': 'Hybrid', **res_lstm_hyb})
    
    # --- Lưu kết quả của lần chạy này ---
    df_results = pd.DataFrame(results)
    output_path = os.path.join(output_dir, f'results_seed_{args.seed}.csv')
    df_results.to_csv(output_path, index=False)
    
    print(f"\nResults for seed {args.seed} saved to '{output_path}'")

if __name__ == '__main__':
    main()
