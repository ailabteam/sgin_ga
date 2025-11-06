# generate_tsne_plot.py
# Phiên bản v1.1: Sửa lỗi tham số 'n_iter' -> 'max_iter' và tối ưu hóa
# để không phải chạy lại t-SNE.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import os
import random
import time

# Import hàm chuẩn bị dữ liệu từ script chính của chúng ta
try:
    from train_final_comparison import prepare_combined_data
except ImportError:
    print("Lỗi: Không tìm thấy file 'train_final_comparison.py'.")
    print("Hãy đảm bảo file này nằm trong cùng thư mục.")
    exit()

def create_tsne_data(X, y, n_samples=5000):
    """
    Hàm để chuẩn bị và chạy thuật toán t-SNE.
    Trả về tọa độ 2D đã được tính toán.
    """
    print(f"Bắt đầu xử lý với {X.shape} mẫu.")
    
    if len(X) > n_samples:
        print(f"Lấy mẫu con {n_samples} từ {len(X)} điểm dữ liệu...")
        indices = random.sample(range(len(X)), n_samples)
        X_sample = X[indices]
        y_sample = y[indices]
    else:
        X_sample = X; y_sample = y

    print("Đang chuẩn hóa dữ liệu...")
    scaler = StandardScaler()
    X_sample_flat = X_sample.reshape(len(X_sample), -1)
    X_scaled = scaler.fit_transform(X_sample_flat)
    
    print("Đang chạy thuật toán t-SNE... (việc này sẽ mất vài phút)")
    tsne = TSNE(
        n_components=2,
        perplexity=40,
        # SỬA LỖI: Đổi 'n_iter' thành 'max_iter'
        max_iter=1000,
        random_state=42,
        verbose=1
    )
    start_time = time.time()
    X_2d = tsne.fit_transform(X_scaled)
    end_time = time.time()
    print(f"t-SNE hoàn tất sau {end_time - start_time:.2f} giây.")
    
    return X_2d, y_sample

def main():
    output_dir = 'paper_figures'
    os.makedirs(output_dir, exist_ok=True)
    
    constellations = ['iridium'] 
    X_vec, y_vec = prepare_combined_data(constellations, 'vector', sequence_length=1, prediction_horizon=1)
    X_ga, y_ga = prepare_combined_data(constellations, 'ga', sequence_length=1, prediction_horizon=1)
    
    # Chạy t-SNE MỘT LẦN cho mỗi bộ dữ liệu
    tsne_coords_vec, y_sample_vec = create_tsne_data(X_vec, y_vec)
    tsne_coords_ga, y_sample_ga = create_tsne_data(X_ga, y_ga)
    
    print("\n--- Đang tạo biểu đồ so sánh cuối cùng ---")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig_combined, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 9))
    
    colors = {0: 'cornflowerblue', 1: 'salmon'}
    sizes = {0: 15, 1: 25}
    alphas = {0: 0.6, 1: 0.8}
    
    # --- TỐI ƯU HÓA: Vẽ lại từ dữ liệu đã có ---
    # --- Panel (a): Traditional Vector Features ---
    y_flat_vec = y_sample_vec.flatten()
    ax1.scatter(tsne_coords_vec[y_flat_vec == 0, 0], tsne_coords_vec[y_flat_vec == 0, 1], 
                c=colors[0], label='No LoS (Class 0)', alpha=alphas[0], s=sizes[0], edgecolors='k', linewidths=0.2)
    ax1.scatter(tsne_coords_vec[y_flat_vec == 1, 0], tsne_coords_vec[y_flat_vec == 1, 1], 
                c=colors[1], label='Has LoS (Class 1)', alpha=alphas[1], s=sizes[1], edgecolors='k', linewidths=0.2)
    ax1.set_title("(a) Traditional Vector Features", fontsize=20, pad=15, weight='bold')
    ax1.set_xlabel("t-SNE Dimension 1", fontsize=16); ax1.set_ylabel("t-SNE Dimension 2", fontsize=16)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.legend(fontsize=14)

    # --- Panel (b): Geometric Algebra Features ---
    y_flat_ga = y_sample_ga.flatten()
    ax2.scatter(tsne_coords_ga[y_flat_ga == 0, 0], tsne_coords_ga[y_flat_ga == 0, 1], 
                c=colors[0], label='No LoS (Class 0)', alpha=alphas[0], s=sizes[0], edgecolors='k', linewidths=0.2)
    ax2.scatter(tsne_coords_ga[y_flat_ga == 1, 0], tsne_coords_ga[y_flat_ga == 1, 1], 
                c=colors[1], label='Has LoS (Class 1)', alpha=alphas[1], s=sizes[1], edgecolors='k', linewidths=0.2)
    ax2.set_title("(b) Geometric Algebra Features", fontsize=20, pad=15, weight='bold')
    ax2.set_xlabel("t-SNE Dimension 1", fontsize=16); ax2.set_ylabel("t-SNE Dimension 2", fontsize=16)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    ax2.legend(fontsize=14)
    
    fig_combined.suptitle("t-SNE Visualization of Feature Spaces for LoS Classification", fontsize=24, weight='bold')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    fig_filename = os.path.join(output_dir, 'fig_tsne_feature_space.png')
    plt.savefig(fig_filename, dpi=600, bbox_inches='tight')
    
    print(f"\nSaved combined Figure 3 to '{fig_filename}'")

if __name__ == "__main__":
    main()
