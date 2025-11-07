# generate_tsne_plot.py
# Mục đích: Tạo Figure 4 - Trực quan hóa không gian đặc trưng bằng t-SNE.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import random

# Import các hàm chuẩn bị dữ liệu từ script huấn luyện
# (Để tránh lặp lại code, chúng ta có thể import trực tiếp)
from train_final_comparison import prepare_combined_data

def generate_tsne(X, y, title):
    """Hàm để tính toán và vẽ t-SNE cho một bộ dữ liệu."""
    print(f"--- Đang tạo t-SNE cho: {title} ---")
    
    # t-SNE rất tốn tài nguyên, chúng ta sẽ lấy một mẫu con ngẫu nhiên
    n_samples = 5000
    if len(X) > n_samples:
        print(f"Lấy mẫu con {n_samples} từ {len(X)} điểm dữ liệu...")
        indices = random.sample(range(len(X)), n_samples)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X
        y_sample = y

    # Chuẩn hóa dữ liệu trước khi chạy t-SNE
    scaler = StandardScaler()
    X_sample_flat = X_sample.reshape(len(X_sample), -1)
    X_scaled = scaler.fit_transform(X_sample_flat)
    
    # Chạy t-SNE
    print("Đang chạy TSNE... (việc này có thể mất vài phút)")
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42, verbose=1)
    X_2d = tsne.fit_transform(X_scaled)
    
    # Tạo biểu đồ
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Tách các điểm theo nhãn
    points_0 = X_2d[y_sample.flatten() == 0]
    points_1 = X_2d[y_sample.flatten() == 1]
    
    ax.scatter(points_0[:, 0], points_0[:, 1], c='cornflowerblue', label='No LoS (Class 0)', alpha=0.6, s=10)
    ax.scatter(points_1[:, 0], points_1[:, 1], c='salmon', label='Has LoS (Class 1)', alpha=0.8, s=15)
    
    ax.set_title(title, fontsize=16)
    ax.set_xlabel("t-SNE Dimension 1", fontsize=12)
    ax.set_ylabel("t-SNE Dimension 2", fontsize=12)
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.6)
    
    return fig, ax

def main():
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    # Chỉ cần một ít dữ liệu để trực quan hóa
    constellations = ['iridium'] 
    horizon_steps = 5
    seq_len = 5
    
    # Chuẩn bị 2 bộ dữ liệu (chỉ cần lấy 1 snapshot, không cần chuỗi)
    # Chúng ta sẽ tạm thời sửa sequence_length=1 để lấy snapshot
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', sequence_length=1, prediction_horizon=horizon_steps)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', sequence_length=1, prediction_horizon=horizon_steps)
    
    # Tạo 2 biểu đồ t-SNE
    fig_vec, _ = generate_tsne(X_vec, y_vec, "Feature Space (Traditional Vectors)")
    fig_ga, _ = generate_tsne(X_ga, y_ga, "Feature Space (Geometric Algebra)")

    # Ghép 2 biểu đồ lại thành một Figure duy nhất
    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # Lấy lại các điểm đã tính
    # (Để cho đơn giản, chúng ta sẽ chạy lại TSNE ở đây, trong code thực tế có thể tối ưu hơn)
    print("\n--- Tạo lại biểu đồ để ghép ---")
    # ... chạy lại TSNE cho Vector ...
    n_samples = 5000
    indices_vec = random.sample(range(len(X_vec)), n_samples)
    X_vec_sample = X_vec[indices_vec].reshape(n_samples, -1)
    y_vec_sample = y_vec[indices_vec]
    X_vec_scaled = StandardScaler().fit_transform(X_vec_sample)
    tsne_vec = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(X_vec_scaled)
    
    # ... chạy lại TSNE cho GA ...
    indices_ga = random.sample(range(len(X_ga)), n_samples)
    X_ga_sample = X_ga[indices_ga].reshape(n_samples, -1)
    y_ga_sample = y_ga[indices_ga]
    X_ga_scaled = StandardScaler().fit_transform(X_ga_sample)
    tsne_ga = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42).fit_transform(X_ga_scaled)
    
    # Vẽ vào subplot 1
    ax1.scatter(tsne_vec[y_vec_sample.flatten() == 0, 0], tsne_vec[y_vec_sample.flatten() == 0, 1], c='cornflowerblue', label='No LoS', alpha=0.6, s=10)
    ax1.scatter(tsne_vec[y_vec_sample.flatten() == 1, 0], tsne_vec[y_vec_sample.flatten() == 1, 1], c='salmon', label='Has LoS', alpha=0.8, s=15)
    ax1.set_title("(a) Traditional Vector Features", fontsize=18, pad=15)
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=14); ax1.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax1.legend()

    # Vẽ vào subplot 2
    ax2.scatter(tsne_ga[y_ga_sample.flatten() == 0, 0], tsne_ga[y_ga_sample.flatten() == 0, 1], c='cornflowerblue', label='No LoS', alpha=0.6, s=10)
    ax2.scatter(tsne_ga[y_ga_sample.flatten() == 1, 0], tsne_ga[y_ga_sample.flatten() == 1, 1], c='salmon', label='Has LoS', alpha=0.8, s=15)
    ax2.set_title("(b) Geometric Algebra Features", fontsize=18, pad=15)
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=14); ax2.set_ylabel("t-SNE Dimension 2", fontsize=14)
    ax2.legend()
    
    fig_combined.suptitle("t-SNE Visualization of Feature Spaces for LoS Classification", fontsize=22)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig4_filename = os.path.join(output_dir, 'fig_tsne_feature_space.png')
    plt.savefig(fig4_filename, dpi=600)
    print(f"\nSaved combined Figure 4 to '{fig4_filename}'")

if __name__ == "__main__":
    main()
