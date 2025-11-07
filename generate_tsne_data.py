# generate_tsne_data.py
# Mục đích: Chỉ tính toán tọa độ t-SNE và lưu ra file CSV cho Figure 3.

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import random
import time

# Import các công cụ từ utils.py
try:
    from utils import prepare_combined_data, set_seed
except ImportError:
    print("Lỗi: Không tìm thấy file 'utils.py'.")
    exit()

def compute_tsne_coords(X, y, n_samples=5000, feature_name=""):
    """
    Hàm để chuẩn bị và chạy thuật toán t-SNE, trả về một DataFrame.
    """
    print(f"Bắt đầu tính toán t-SNE cho '{feature_name}'...")
    
    if len(X) > n_samples:
        print(f"Lấy mẫu con {n_samples} từ {len(X)} điểm dữ liệu...")
        indices = np.random.choice(len(X), n_samples, replace=False)
        # Lấy snapshot (không cần chuỗi), nên reshape
        X_sample = X[indices].reshape(n_samples, -1)
        y_sample = y[indices]
    else:
        X_sample = X.reshape(len(X), -1)
        y_sample = y

    print("Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_sample)
    
    print("Đang chạy thuật toán t-SNE... (việc này sẽ mất vài phút)")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    start_time = time.time()
    X_2d = tsne.fit_transform(X_scaled)
    end_time = time.time()
    print(f"t-SNE hoàn tất sau {end_time - start_time:.2f} giây.")
    
    return pd.DataFrame({'dim1': X_2d[:, 0], 'dim2': X_2d[:, 1], 'label': y_sample.flatten()})

def main():
    set_seed(42)
    output_dir = 'results'
    os.makedirs(output_dir, exist_ok=True)
    
    # Chỉ cần một chòm vệ tinh để trực quan hóa
    constellations = ['iridium']
    
    # Chuẩn bị dữ liệu Vector (lấy snapshot, không phải chuỗi)
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', sequence_length=1, prediction_horizon=1)
    df_tsne_vec = compute_tsne_coords(X_vec, y_vec, feature_name="Vector")
    vec_path = os.path.join(output_dir, 'figure_3_tsne_vector_data.csv')
    df_tsne_vec.to_csv(vec_path, index=False)
    print(f"Saved t-SNE data for Vector features to '{vec_path}'\n")
    
    # Chuẩn bị dữ liệu GA
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', sequence_length=1, prediction_horizon=1)
    df_tsne_ga = compute_tsne_coords(X_ga, y_ga, feature_name="GA")
    ga_path = os.path.join(output_dir, 'figure_3_tsne_ga_data.csv')
    df_tsne_ga.to_csv(ga_path, index=False)
    print(f"Saved t-SNE data for GA features to '{ga_path}'")

if __name__ == "__main__":
    main()
