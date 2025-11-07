# train_ablation_study.py (v2.0 - Data Export Only)
# Mục đích: Chạy thực nghiệm Ablation Study và lưu kết quả ra file CSV.

import numpy as np
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Import các công cụ từ utils.py
from utils import (
    LSTM_Predictor,
    prepare_combined_data,
    train_and_evaluate_convergence, # Tái sử dụng hàm train mạnh mẽ này
    set_seed
)

def main():
    # Sử dụng cùng một seed để có thể so sánh
    set_seed(42)

    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # --- Thiết lập ---
    constellations = ['iridium', 'starlink', 'oneweb']
    horizon_steps = 5
    seq_len = 5
    NUM_EPOCHS = 15 # Huấn luyện ít hơn cho ablation study
    
    # Các bộ đặc trưng để "mổ xẻ"
    feature_types_for_ablation = ['bivector_only', 'relative_only', 'ga', 'hybrid']
    
    # --- Chuẩn bị Dữ liệu ---
    # Tải dữ liệu một lần và lưu vào dictionary
    datasets = {
        ft: prepare_combined_data(constellations, ft, seq_len, horizon_steps)
        for ft in feature_types_for_ablation
    }

    # --- Chạy Thực nghiệm ---
    ablation_results = []
    
    for ft in feature_types_for_ablation:
        print(f"\n--- Running Ablation: LSTM ({ft}) ---")
        X, y = datasets[ft]
        
        # Chia dữ liệu
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
        
        # Khởi tạo mô hình
        model = LSTM_Predictor(input_size=X.shape[2])
        
        # Huấn luyện và lấy kết quả cuối cùng. Bỏ qua history.
        res = train_and_evaluate_convergence(model, X_train, y_train, X_test, y_test, f"LSTM ({ft})", NUM_EPOCHS)
        
        # Đổi tên feature cho đẹp
        formatted_name = {
            'bivector_only': 'Bivector Only',
            'relative_only': 'Relative Vec. Only',
            'ga': 'GA (Full)',
            'hybrid': 'Hybrid (Full)'
        }[ft]
        
        ablation_results.append({'Feature_Set': formatted_name, 'F1_Score': res['F1 Score']})

    # --- Lưu Kết quả ---
    print("\n--- Saving Ablation Study Results ---")
    df_ablation = pd.DataFrame(ablation_results)
    ablation_path = os.path.join(output_dir, 'figure_4_ablation_data.csv')
    df_ablation.to_csv(ablation_path, index=False, float_format='%.4f')
    
    print(f"Saved ablation study data (for Figure 4) to '{ablation_path}'")
    print(df_ablation.to_string(index=False))

if __name__ == '__main__':
    main()
